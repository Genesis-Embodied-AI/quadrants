"""Two-level fastcache key derivation and persistence.

Two-level cache
---------------
Pruning information, already produced during compile, is exposed as a first-class lookup so the args hash can walk
*only* the paths the kernel reads:

  - L1 (this module's ``make_source_config_key`` + ``load_pruning_info`` / ``store_pruning_info``): keyed by
    source+config only (no args). Stores ``PruningInfo`` - the set of kernel-accessed flat names (e.g.
    ``__qd_state__qd_x``) plus the ``graph_do_while_levels`` table (also a kernel-source property). The flat-name set
    is the *union* over every specialization compiled so far, not the reads of one specialization - see "Pruning is
    per-specialization" below.

  - L2 (``make_full_cache_key`` + ``load_full`` / ``store_full``): keyed by L1 key + the *narrow* args hash computed
    with pruning info from L1. Stores the C++ ``frontend_cache_key`` that names the compiled artifact.

Lookup flow on a warm call: L1 lookup -> narrow args hash (paths from L1) -> L2 lookup -> load artifact.

Cold compile flow: L1 miss -> cold compile (pass 0 + pass 1) -> store L1 -> compute narrow args hash -> store L2.

Safety implication
------------------
A kernel-unused path's contents (any type, including unrecognised tensor-likes) is *guaranteed* not to affect kernel
codegen, so dropping it from the hash is correct by construction. Paths the kernel *does* read still go through
``args_hasher.stringify_obj_type``; if it encounters an unrecognised type at such a path it fails the call's fastcache
loudly (one-shot ``[FASTCACHE][UNKNOWN_TYPE]`` warning identifying the offending ``type(v).__qualname__``), so a missed
type registration is impossible to miss and cannot serve stale cached results.

Pruning is per-specialization
-----------------------------
Which paths a kernel reads is not a source+config property: ``qd.static(state.flag)`` can make one specialization
read ``state.x`` and another read ``state.y``. One L1 entry is shared by all of them, so it holds the union of their
paths and every L2 key is derived from that union. The invariant this buys: a stored L2 entry's key covers every path
the specialization it names reads, so a dtype / ndim / baked-value change at any of them changes the key. The cost is
over-invalidation - a path read by one specialization enters its siblings' keys too.
"""

import importlib
import json
import os
import warnings
from enum import IntEnum
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

# Mixed into L1 / L2 keys so the two levels cannot collide, and so single-level entries written by older Quadrants
# installs are ignored rather than mis-served.
_L1_MARKER = "l1"
_L2_MARKER = "l2"

# Bumped whenever the persisted L1CacheValue / CacheValue schema changes. Mixed into both L1 and L2 keys so old
# entries from a prior schema are not mis-read on a Quadrants version where __version_str__ hasn't moved.
# v2 replaced the single graph_do_while_arg string with a nested level table. v3 added the AST-resolved flat C++
# arg-ids for qd.graph_do_while conditions and qd.checkpoint(yield_on=...) targets so the launch path can forward them
# directly without per-launch name matching (necessary for @qd.data_oriented member ndarrays). v4 added the per-slot
# `checkpoint_user_label_enum_qualnames` table so an IntEnum cp_id (e.g. `qd.checkpoint(Stage.SIM, ...)`) round-trips
# through fast-cache restore as the original IntEnum member rather than the underlying int. v5 made the L1 flat-name
# set a union over specializations, so a v4 entry can name an artifact whose key omits paths it depends on.
_CACHE_VALUE_SCHEMA_VERSION = "cachevalue-v5-pruning-union"


def _intenum_member_qualname(value: Any) -> str | None:
    """Return ``"module.ClassQualName.MEMBER"`` for an ``IntEnum`` member, else ``None``.

    Stored alongside ``checkpoint_user_labels_by_cp_id`` so that ``_resolve_intenum_member`` can rebuild the original
    enum member on fast-cache restore -- pydantic coerces ``IntEnum`` to plain ``int`` at ``CacheValue`` construction
    time (it sees ``list[int | None]``), which would otherwise silently break the documented contract that
    ``qd.checkpoint(Stage.X, ...)`` round-trips ``Stage.X`` rather than the raw int through ``status.checkpoint``.
    Returns ``None`` for plain ints, ``None`` labels, anonymous enums (no ``__module__``), and other unsupported
    shapes -- the loader falls back to the raw int in those cases.
    """
    if not isinstance(value, IntEnum):
        return None
    cls = type(value)
    module = getattr(cls, "__module__", None)
    qualname = getattr(cls, "__qualname__", None)
    name = getattr(value, "name", None)
    if not module or not qualname or not name:
        return None
    return f"{module}.{qualname}.{name}"


def _resolve_intenum_member(qualname: str | None, fallback: int | None) -> int | IntEnum | None:
    """Inverse of ``_intenum_member_qualname``: look up the enum member by ``"module.ClassQualName.MEMBER"``.

    Returns the resolved ``IntEnum`` member if every step (module import, attribute walk) succeeds AND the member's int
    value matches ``fallback`` (the raw int from ``checkpoint_user_labels_by_cp_id`` we already persisted). Mismatch or
    any failure -- module renamed since the cache was written, enum class refactored, member removed, etc. -- falls back
    to ``fallback`` so the user still gets a usable (if enum-identity-less) label rather than a hard crash. ``None``
    qualname / ``None`` fallback short-circuit to ``fallback`` for the plain-int label case.
    """
    if qualname is None or fallback is None:
        return fallback
    try:
        # qualname is "module.path.Class[.Nested].MEMBER"; the MEMBER tail is always one segment, so rsplit once. The
        # remaining cls_path mixes dotted module path + dotted class qualname; we try progressively shorter module
        # prefixes until one imports, then resolve the rest as attribute chain. This handles top-level enums
        # (``mymod.Stage.LOAD``), enums nested in classes (``mymod.Outer.Inner.MEMBER``), and enums in subpackages
        # (``a.b.Stage.LOAD``) without needing the user to declare which prefix is the module.
        cls_path, _, member_name = qualname.rpartition(".")
        if not cls_path or not member_name:
            return fallback
        module = None
        cls_attr_path = ""
        segments = cls_path.split(".")
        for i in range(len(segments), 0, -1):
            try:
                module = importlib.import_module(".".join(segments[:i]))
                cls_attr_path = ".".join(segments[i:])
                break
            except ImportError:
                continue
        if module is None:
            return fallback
        obj: Any = module
        if cls_attr_path:
            for seg in cls_attr_path.split("."):
                obj = getattr(obj, seg)
        obj = getattr(obj, member_name)
    except (AttributeError, ValueError):
        return fallback
    if isinstance(obj, IntEnum) and int(obj) == int(fallback):
        return obj
    return fallback


def make_source_config_key(kernel_source_info: FunctionSourceInfo) -> str:
    """Build the L1 cache key: source + config + version, with no dependence on args."""
    kernel_hash = function_hasher.hash_kernel(kernel_source_info)
    config_hash = config_hasher.hash_compile_config()
    # In L1 rather than L2: caps change codegen without changing argument identity.
    device_caps_hash = config_hasher.hash_device_caps()
    return hash_iterable_strings(
        (
            _L1_MARKER,
            quadrants.__version_str__,
            kernel_hash,
            config_hash,
            device_caps_hash,
            kernel_source_info.filepath,
            str(kernel_source_info.start_lineno),
            "pruned",
            "kcov" if os.environ.get("QD_KERNEL_COVERAGE") == "1" else "",
            _CACHE_VALUE_SCHEMA_VERSION,
        )
    )


def make_full_cache_key(source_config_key: str, narrow_args_hash: str) -> str:
    """Build the L2 cache key from the L1 key + narrow args hash. See module docstring."""
    return hash_iterable_strings((_L2_MARKER, source_config_key, narrow_args_hash, _CACHE_VALUE_SCHEMA_VERSION))


def compute_narrow_args_hash(
    raise_on_templated_floats: bool,
    kernel_source_info: FunctionSourceInfo,
    args: Sequence[Any],
    arg_metas: Sequence[ArgMetadata],
    pruning_paths: set[str] | None,
) -> str | None:
    """Compute the args hash narrowed by ``pruning_paths`` (or wide if ``pruning_paths is None``), or ``None`` if an
    unsupported type forces fastcache off for this call.
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
    """Persisted L1 entry - pruning info that's args-independent (though not specialization-independent).

    Pruning info is the set of *flat names* (``__qd_<arg>__qd_<child>__qd_...``) that the kernel actually reads.
    Computed during compile (``Pruning.used_vars_by_func_id``); persisted here so subsequent calls can build
    a narrow args hash without having to recompile. The stored set is the union over specializations - see "Pruning is
    per-specialization" in the module docstring.

    ``graph_do_while_levels`` is only a pre-compile seed for the L1-hit / L2-miss path (the AST walk clears and
    rebuilds the table anyway); a fast-cache *restore* takes the table from ``CacheValue`` instead, since a
    ``qd.graph_do_while`` nested inside a ``qd.static`` branch belongs to some specializations and not others. Each
    entry is ``(cond_arg_name, parent_id, cond_cpp_arg_id)``, indexed by level id (outer before inner); see
    ``CacheValue`` for why the launch path needs the AST-resolved ``cond_cpp_arg_id``.

    ``hashed_function_source_infos`` rejects an L1 hit when a helper's source changed, which the key itself cannot
    catch: ``kernel_hash`` covers only the entry point.
    """

    used_py_dataclass_parameters: set[str]
    hashed_function_source_infos: list[HashedFunctionSourceInfo]
    graph_do_while_levels: list[tuple[str, int, int]] | None = None


def store_pruning_info(
    source_config_key: str,
    function_source_infos: Iterable[FunctionSourceInfo],
    used_py_dataclass_parameters: set[str],
    graph_do_while_levels: list[tuple[str, int, int]] | None = None,
) -> None:
    """Persist the L1 entry, or re-persist it with a grown union. See ``L1CacheValue`` for what's stored / why."""
    if not source_config_key:
        return
    cache = PythonSideCache()
    hashed_function_source_infos = function_hasher.hash_functions(function_source_infos)
    cache_value = L1CacheValue(
        used_py_dataclass_parameters=used_py_dataclass_parameters,
        hashed_function_source_infos=list(hashed_function_source_infos),
        graph_do_while_levels=graph_do_while_levels,
    )
    cache.store(source_config_key, cache_value.model_dump_json())


def persist_l1_and_set_l2_key(
    *,
    l1_key: str | None,
    kernel_source_info: FunctionSourceInfo | None,
    used_py_dataclass_parameters: set[str] | None,
    visited_functions: Iterable[FunctionSourceInfo],
    graph_do_while_levels: list[tuple[str, int, int]] | None,
    pruning_paths_from_l1: set[str] | None,
    fast_checksum: str | None,
    raise_on_templated_floats: bool,
    py_args: tuple[Any, ...],
    arg_metas: Sequence[ArgMetadata],
) -> tuple[str | None, bool]:
    """After a successful materialize, persist L1 (if missing or incomplete) and derive the L2 key.

      1. L1 missing (``pruning_paths_from_l1 is None``): store the freshly-computed pruning info so the next call
         from a new process can skip the args-walk warm-up.

      2. L1 present but not listing every path this specialization reads: grow it to the union, and drop the L2 key
         phase 2 derived from the smaller set - that key ignores paths this artifact depends on.

      3. No ``fast_checksum`` at that point (L1 missing, dropped by 2., or phase 2 saw a FIELD-related
         ``FastcacheSkip``): derive the L2 key from the path set L1 now holds.

    Returns ``(new_fast_checksum, generated)``, where ``generated`` means the kernel had no L2 key on entry and now
    has one (it drives the caller's cache-observations counter). Case 2. re-keys with ``generated`` False, so the
    caller must assign ``new_fast_checksum`` either way. ``(None, False)`` means fastcache is inactive for this
    kernel (``l1_key`` falsy / source info missing / used-params not recorded).
    """
    if not l1_key:
        return None, False
    if kernel_source_info is None:
        return fast_checksum, False
    if used_py_dataclass_parameters is None:
        return fast_checksum, False
    had_checksum = fast_checksum is not None
    key_paths = used_py_dataclass_parameters
    if pruning_paths_from_l1 is None:
        store_pruning_info(
            l1_key,
            visited_functions,
            used_py_dataclass_parameters,
            graph_do_while_levels=graph_do_while_levels,
        )
    elif used_py_dataclass_parameters <= pruning_paths_from_l1:
        key_paths = pruning_paths_from_l1
    else:
        # Grow the shared entry so all specializations key on the same superset.
        key_paths = pruning_paths_from_l1 | used_py_dataclass_parameters
        store_pruning_info(
            l1_key,
            visited_functions,
            key_paths,
            graph_do_while_levels=graph_do_while_levels,
        )
        fast_checksum = None
    if fast_checksum is None:
        narrow_args_hash = compute_narrow_args_hash(
            raise_on_templated_floats,
            kernel_source_info,
            py_args,
            arg_metas,
            key_paths,
        )
        if narrow_args_hash is not None:
            return make_full_cache_key(l1_key, narrow_args_hash), not had_checksum
    return fast_checksum, False


def load_pruning_info(
    source_config_key: str,
) -> tuple[set[str], list[tuple[str, int, int]] | None] | tuple[None, None]:
    """Look up L1 cache. Returns (pruning_paths, graph_do_while_levels) on hit, (None, None) on miss.

    A changed helper source invalidates the entry, which is reported as a miss so the caller cold-compiles and
    overwrites it.
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
    return cache_value.used_py_dataclass_parameters, cache_value.graph_do_while_levels


class CacheValue(BaseModel):
    """Persisted L2 entry - frontend cache key for the compiled artifact + source-validation metadata.

    ``used_py_dataclass_parameters`` duplicates what L1 stores, for compatibility with on-disk caches; L1's copy is
    the one that narrows the args hash on warm calls.
    """

    frontend_cache_key: str
    hashed_function_source_infos: list[HashedFunctionSourceInfo]
    used_py_dataclass_parameters: set[str]
    # Nested graph_do_while level table as (cond_arg_name, parent_id, cond_cpp_arg_id) triples, indexed by level id.
    # None / empty for kernels without graph_do_while. ``cond_cpp_arg_id`` is the flat C++ arg-id resolved at AST-build
    # time by ``ASTTransformer._resolve_ndarray_kernel_arg_id`` and is required by the launch path to support
    # `@qd.data_oriented` member conditions (`qd.graph_do_while(self.counter)`) -- name-matching against ``arg_metas``
    # only resolves top-level parameters.
    graph_do_while_levels: list[tuple[str, int, int]] | None = None
    # AST-build-time-resolved checkpoint metadata, indexed by internal cp_id. Empty for kernels without any `with
    # qd.checkpoint(...)` block. See `Kernel.checkpoint_yield_on_args` / `Kernel.checkpoint_yield_on_cpp_arg_ids` /
    # `Kernel.checkpoint_user_labels_by_cp_id` for what each entry means. Restored alongside the C++-side cached
    # kernel so the launch path can forward `yield_on=` arg-ids and translate `from_checkpoint=` labels without
    # re-running the AST transformer.
    checkpoint_yield_on_args: list[str | None] = []
    checkpoint_yield_on_cpp_arg_ids: list[int] = []
    checkpoint_user_labels_by_cp_id: list[int | None] = []
    # Parallel to ``checkpoint_user_labels_by_cp_id``: each entry is the dotted ``module.ClassQualName.MEMBER`` of the
    # original ``IntEnum`` member the user passed as ``cp_id``, or ``None`` if the user passed a plain int (or for
    # implicit auto-wrap checkpoints). On fast-cache restore the loader runs each entry through
    # ``_resolve_intenum_member`` to rebuild the IntEnum, preserving the documented contract that
    # ``qd.checkpoint(Stage.X, ...)`` round-trips ``Stage.X`` (not the underlying int) through ``status.checkpoint`` and
    # ``kernel.resume(from_checkpoint=...)`` -- pydantic coerces IntEnum to int at ``CacheValue`` construction time so
    # the parallel qualname column is what carries the enum identity.
    checkpoint_user_label_enum_qualnames: list[str | None] = []


def store(
    frontend_cache_key: str,
    fast_cache_key: str,
    function_source_infos: Iterable[FunctionSourceInfo],
    used_py_dataclass_parameters: set[str],
    graph_do_while_levels: list[tuple[str, int, int]] | None = None,
    checkpoint_yield_on_args: list[str | None] | None = None,
    checkpoint_yield_on_cpp_arg_ids: list[int] | None = None,
    checkpoint_user_labels_by_cp_id: list[int | None] | None = None,
) -> None:
    """Persist the L2 entry - the C++ frontend cache key that names the compiled artifact for this call.

    Note that unlike other caches, this cache does not store the compiled artifact itself; it stores what we need to
    validate that the cache key is still valid, plus the AST-transformer output that a fast-cache restore skips
    recomputing:
    - we have a cache key, based on args and top level kernel function
    - we want to use this to look up LLVM IR, in C++ side cache
    - however, before doing that, we first want to validate that the source code didn't change
        - i.e. is our cache key still valid?
    - the python side cache contains information we will use to verify that our cache key is valid
        - ie the list of function source infos

    ``checkpoint_user_label_enum_qualnames`` is derived here rather than passed in, because the labels still hold
    their original ``IntEnum`` instances at this point - ``CacheValue.__init__`` coerces them to plain ints.
    """
    if not fast_cache_key:
        return
    assert frontend_cache_key is not None
    cache = PythonSideCache()
    hashed_function_source_infos = function_hasher.hash_functions(function_source_infos)
    labels = checkpoint_user_labels_by_cp_id or []
    enum_qualnames = [_intenum_member_qualname(lbl) for lbl in labels]
    cache_value_obj = CacheValue(
        frontend_cache_key=frontend_cache_key,
        hashed_function_source_infos=list(hashed_function_source_infos),
        used_py_dataclass_parameters=used_py_dataclass_parameters,
        graph_do_while_levels=graph_do_while_levels,
        checkpoint_yield_on_args=checkpoint_yield_on_args or [],
        checkpoint_yield_on_cpp_arg_ids=checkpoint_yield_on_cpp_arg_ids or [],
        checkpoint_user_labels_by_cp_id=labels,
        checkpoint_user_label_enum_qualnames=enum_qualnames,
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


def load(cache_key: str) -> CacheValue | None:
    """Look up the L2 cache: a validated ``CacheValue`` for *cache_key*, or None on miss / stale entry.

    A changed helper source invalidates the entry.
    """
    cache_value = _try_load(cache_key)
    if cache_value is None:
        return None
    if function_hasher.validate_hashed_function_infos(cache_value.hashed_function_source_infos):
        return cache_value
    return None


def dump_stats() -> None:
    print("dump stats")
    args_hasher.dump_stats()
    function_hasher.dump_stats()
