"""Two-level fastcache key derivation and persistence.

Two-level cache
---------------
The fastcache now exposes pruning information (already produced during compile) as a first-class lookup so the args
hash can walk *only* paths the kernel reads:

  - L1 (this module's ``make_source_config_key`` + ``load_pruning_info`` / ``store_pruning_info``): keyed by
    source+config only (no args). Stores ``PruningInfo`` — the set of kernel-accessed flat names (e.g.
    ``__qd_state__qd_x``) plus the ``graph_do_while_arg`` (also a kernel-source property).

  - L2 (``make_full_cache_key`` + ``load_full`` / ``store_full``): keyed by L1 key + the *narrow* args hash computed
    with pruning info from L1. Stores the C++ ``frontend_cache_key`` that names the compiled artifact.

Lookup flow on a warm call: L1 lookup → narrow args hash (paths from L1) → L2 lookup → load artifact.

Cold compile flow: L1 miss → cold compile (pass 0 + pass 1) → store L1 → compute narrow args hash → store L2.

Safety implication
------------------
A kernel-unused path's contents (any type, including unrecognised tensor-likes) is *guaranteed* not to affect kernel
codegen, so dropping it from the hash is correct by construction. Paths the kernel *does* read still go through
``args_hasher.stringify_obj_type``; if it encounters an unrecognised type at such a path it fails the call's fastcache
loudly (one-shot ``[FASTCACHE][UNKNOWN_TYPE]`` warning identifying the offending ``type(v).__qualname__``), so a missed
type registration is impossible to miss and cannot serve stale cached results.
"""

import json
import os
import warnings
from typing import Any, Iterable, Sequence

import pydantic
from pydantic import BaseModel

import quadrants
from quadrants import _logging

from .._wrap_inspect import FunctionSourceInfo
from ..kernel_arguments import ArgMetadata
from . import args_hasher, config_hasher, function_hasher
from .args_hasher import FastcacheSkip
from .fast_caching_types import HashedFunctionSourceInfo
from .hash_utils import hash_iterable_strings
from .python_side_cache import PythonSideCache

# Prefix bytes mixed into L1 / L2 keys so they cannot collide even if the underlying inputs happen to hash to the
# same string. The original single-level cache key (kept for backward-compat reads via ``load`` below) had no such
# prefix; the new two-level scheme uses ``l1:`` and ``l2:`` markers so old single-level entries from prior Quadrants
# installs are simply ignored rather than mis-served.
_L1_MARKER = "l1"
_L2_MARKER = "l2"


def make_source_config_key(kernel_source_info: FunctionSourceInfo) -> str:
    """Build the L1 cache key: source + config + version, with no dependence on args.

    Used by ``_try_load_fastcache`` before any args walking. The same key drives ``load_pruning_info`` /
    ``store_pruning_info``; the matching ``make_full_cache_key`` derives the L2 key from this plus the narrow args
    hash.
    """
    kernel_hash = function_hasher.hash_kernel(kernel_source_info)
    config_hash = config_hasher.hash_compile_config()
    return hash_iterable_strings(
        (
            _L1_MARKER,
            quadrants.__version_str__,
            kernel_hash,
            config_hash,
            kernel_source_info.filepath,
            str(kernel_source_info.start_lineno),
            "pruned",
            "kcov" if os.environ.get("QD_KERNEL_COVERAGE") == "1" else "",
        )
    )


def make_full_cache_key(source_config_key: str, narrow_args_hash: str) -> str:
    """Build the L2 cache key from the L1 key + narrow args hash. See module docstring."""
    return hash_iterable_strings((_L2_MARKER, source_config_key, narrow_args_hash))


def compute_narrow_args_hash(
    raise_on_templated_floats: bool,
    kernel_source_info: FunctionSourceInfo,
    args: Sequence[Any],
    arg_metas: Sequence[ArgMetadata],
    pruning_paths: set[str] | None,
) -> str | None:
    """Compute the args hash narrowed by ``pruning_paths`` (or wide if ``pruning_paths is None``).

    Returns ``None`` if a recognised-but-unsupported tensor-like type forces fastcache off — the caller emits
    the appropriate user-visible diagnostic via the ``FastcacheSkip.WARN`` branch.
    """
    args_hash = args_hasher.hash_args(raise_on_templated_floats, args, arg_metas, pruning_paths=pruning_paths)
    if isinstance(args_hash, FastcacheSkip):
        if args_hash is FastcacheSkip.WARN:
            # the bit in caps at start should not be modified without modifying corresponding text
            # freetext bit can be freely modified
            _logging.warn(
                f"[FASTCACHE][INVALID_FUNC] The pure function {kernel_source_info.function_name} could not be "
                "fast cached, because one or more parameter types were invalid"
            )
        return None
    return args_hash


class L1CacheValue(BaseModel):
    """Persisted L1 entry — pruning info that's source-and-config-deterministic (not args-dependent).

    Pruning info is the set of *flat names* (``__qd_<arg>__qd_<child>__qd_…``) that the kernel actually reads.
    Computed during compile (``Pruning.used_vars_by_func_id``); persisted here so subsequent calls can build
    a narrow args hash without having to recompile.

    ``graph_do_while_arg`` is also stored here because it's a property of the kernel source (not of any
    particular arg value).

    ``hashed_function_source_infos`` is the same content-hash list used for L2 validation; an L1 hit is
    rejected if any helper source has changed since the L1 entry was written, even if the kernel source
    itself hasn't (kernel_hash only covers the entry point).
    """

    used_py_dataclass_parameters: set[str]
    hashed_function_source_infos: list[HashedFunctionSourceInfo]
    graph_do_while_arg: str | None = None


def store_pruning_info(
    source_config_key: str,
    function_source_infos: Iterable[FunctionSourceInfo],
    used_py_dataclass_parameters: set[str],
    graph_do_while_arg: str | None = None,
) -> None:
    """Persist the L1 entry after a cold compile. See ``L1CacheValue`` for what's stored / why."""
    if not source_config_key:
        return
    cache = PythonSideCache()
    hashed_function_source_infos = function_hasher.hash_functions(function_source_infos)
    cache_value = L1CacheValue(
        used_py_dataclass_parameters=used_py_dataclass_parameters,
        hashed_function_source_infos=list(hashed_function_source_infos),
        graph_do_while_arg=graph_do_while_arg,
    )
    cache.store(source_config_key, cache_value.model_dump_json())


def load_pruning_info(
    source_config_key: str,
) -> tuple[set[str], str | None] | tuple[None, None]:
    """Look up L1 cache. Returns (pruning_paths, graph_do_while_arg) on hit, (None, None) on miss / invalid.

    Validates ``hashed_function_source_infos`` against the current on-disk source; if any helper has changed
    since the entry was written, the entry is invalid and we treat the lookup as a miss so the caller does a
    cold compile (which will overwrite the stale L1 entry).
    """
    cache = PythonSideCache()
    maybe_value_json = cache.try_load(source_config_key)
    if maybe_value_json is None:
        return None, None
    try:
        cache_value = L1CacheValue.model_validate_json(maybe_value_json)
    except (pydantic.ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
        warnings.warn(f"Failed to parse L1 cache entry: {e}")
        return None, None
    if not function_hasher.validate_hashed_function_infos(cache_value.hashed_function_source_infos):
        return None, None
    return cache_value.used_py_dataclass_parameters, cache_value.graph_do_while_arg


class CacheValue(BaseModel):
    """Persisted L2 entry — frontend cache key for the compiled artifact + source-validation metadata.

    The full pruning info is duplicated here for backward-compat with existing on-disk caches; it's the same
    set that L1 also stores. The L1 set is the source of truth for narrowing the args hash on warm calls.
    """

    frontend_cache_key: str
    hashed_function_source_infos: list[HashedFunctionSourceInfo]
    used_py_dataclass_parameters: set[str]
    graph_do_while_arg: str | None = None


def store(
    frontend_cache_key: str,
    fast_cache_key: str,
    function_source_infos: Iterable[FunctionSourceInfo],
    used_py_dataclass_parameters: set[str],
    graph_do_while_arg: str | None = None,
) -> None:
    """Persist the L2 entry — the C++ frontend cache key that names the compiled artifact for this call.

    ``fast_cache_key`` is the L2 key from ``make_full_cache_key``. The L1 entry has typically been stored
    earlier by ``store_pruning_info`` during the same materialize.
    """
    if not fast_cache_key:
        return
    assert frontend_cache_key is not None
    cache = PythonSideCache()
    hashed_function_source_infos = function_hasher.hash_functions(function_source_infos)
    cache_value_obj = CacheValue(
        frontend_cache_key=frontend_cache_key,
        hashed_function_source_infos=list(hashed_function_source_infos),
        used_py_dataclass_parameters=used_py_dataclass_parameters,
        graph_do_while_arg=graph_do_while_arg,
    )
    cache.store(fast_cache_key, cache_value_obj.model_dump_json())


def _try_load(cache_key: str) -> CacheValue | None:
    cache = PythonSideCache()
    maybe_cache_value_json = cache.try_load(cache_key)
    if maybe_cache_value_json is None:
        return None
    try:
        cache_value_obj = CacheValue.model_validate_json(maybe_cache_value_json)
    except (pydantic.ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
        warnings.warn(f"Failed to parse cache file {e}")
        return None
    return cache_value_obj


def load(cache_key: str) -> tuple[set[str], str, str | None] | tuple[None, None, None]:
    """Look up L2 cache. Returns (used_pruning_paths, frontend_cache_key, graph_do_while_arg) on hit.

    Validates helper-source hashes against the live source; an L2 entry is invalidated if any helper changed.
    """
    cache_value = _try_load(cache_key)
    if cache_value is None:
        return None, None, None
    if function_hasher.validate_hashed_function_infos(cache_value.hashed_function_source_infos):
        return cache_value.used_py_dataclass_parameters, cache_value.frontend_cache_key, cache_value.graph_do_while_arg
    return None, None, None


def dump_stats() -> None:
    print("dump stats")
    args_hasher.dump_stats()
    function_hasher.dump_stats()
