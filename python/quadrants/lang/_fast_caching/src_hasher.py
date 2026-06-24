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

# Bumped whenever the persisted CacheValue schema changes (see create_cache_key). v2 replaced the single
# graph_do_while_arg string with a nested level table. v3 added the AST-resolved flat C++ arg-ids for
# qd.graph_do_while conditions and qd.checkpoint(yield_on=...) targets so the launch path can forward them
# directly without per-launch name matching (necessary for @qd.data_oriented member ndarrays). v4 added the
# per-slot `checkpoint_user_label_enum_qualnames` table so an IntEnum cp_id (e.g. `qd.checkpoint(Stage.SIM, ...)`)
# round-trips through fast-cache restore as the original IntEnum member rather than the underlying int.
_CACHE_VALUE_SCHEMA_VERSION = "cachevalue-v4-intenum-qualnames"


def _intenum_member_qualname(value: Any) -> str | None:
    """Return ``"module.ClassQualName.MEMBER"`` for an ``IntEnum`` member, else ``None``.

    Stored alongside ``checkpoint_user_labels_by_cp_id`` so that ``_resolve_intenum_member`` can rebuild the
    original enum member on fast-cache restore -- pydantic coerces ``IntEnum`` to plain ``int`` at ``CacheValue``
    construction time (it sees ``list[int | None]``), which would otherwise silently break the documented
    contract that ``qd.checkpoint(Stage.X, ...)`` round-trips ``Stage.X`` rather than the raw int through
    ``status.checkpoint``. Returns ``None`` for plain ints, ``None`` labels, anonymous enums (no ``__module__``),
    and other unsupported shapes -- the loader falls back to the raw int in those cases.
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

    Returns the resolved ``IntEnum`` member if every step (module import, attribute walk) succeeds AND the member's
    int value matches ``fallback`` (the raw int from ``checkpoint_user_labels_by_cp_id`` we already persisted).
    Mismatch or any failure -- module renamed since the cache was written, enum class refactored, member removed,
    etc. -- falls back to ``fallback`` so the user still gets a usable (if enum-identity-less) label rather than a
    hard crash. ``None`` qualname / ``None`` fallback short-circuit to ``fallback`` for the plain-int label case.
    """
    if qualname is None or fallback is None:
        return fallback
    try:
        # qualname is "module.path.Class[.Nested].MEMBER"; the MEMBER tail is always one segment, so rsplit once.
        # The remaining cls_path mixes dotted module path + dotted class qualname; we try progressively shorter
        # module prefixes until one imports, then resolve the rest as attribute chain. This handles top-level
        # enums (``mymod.Stage.LOAD``), enums nested in classes (``mymod.Outer.Inner.MEMBER``), and enums in
        # subpackages (``a.b.Stage.LOAD``) without needing the user to declare which prefix is the module.
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


def create_cache_key(
    raise_on_templated_floats: bool,
    kernel_source_info: FunctionSourceInfo,
    args: Sequence[Any],
    arg_metas: Sequence[ArgMetadata],
) -> str | None:
    """
    cache key takes into account:
    - arg types
    - cache value arg values
    - kernel function (but not sub functions)
    - compilation config (which includes arch, and debug)
    """
    args_hash = args_hasher.hash_args(raise_on_templated_floats, args, arg_metas)
    if isinstance(args_hash, FastcacheSkip):
        if args_hash is FastcacheSkip.WARN:
            # the bit in caps at start should not be modified without modifying corresponding text
            # freetext bit can be freely modified
            _logging.warn(
                f"[FASTCACHE][INVALID_FUNC] The pure function {kernel_source_info.function_name} could not be "
                "fast cached, because one or more parameter types were invalid"
            )
        return None
    kernel_hash = function_hasher.hash_kernel(kernel_source_info)
    config_hash = config_hasher.hash_compile_config()
    cache_key = hash_iterable_strings(
        (
            quadrants.__version_str__,
            kernel_hash,
            args_hash,
            config_hash,
            kernel_source_info.filepath,
            str(kernel_source_info.start_lineno),
            "pruned",
            "kcov" if os.environ.get("QD_KERNEL_COVERAGE") == "1" else "",
            # Fast-cache value schema version. Bump when CacheValue's stored fields change so stale entries are not
            # mis-read. v2: graph_do_while single-arg -> nested level table.
            _CACHE_VALUE_SCHEMA_VERSION,
        )
    )
    return cache_key


class CacheValue(BaseModel):
    frontend_cache_key: str
    hashed_function_source_infos: list[HashedFunctionSourceInfo]
    used_py_dataclass_parameters: set[str]
    # Nested graph_do_while level table as (cond_arg_name, parent_id, cond_cpp_arg_id) triples, indexed by level
    # id. None / empty for kernels without graph_do_while. ``cond_cpp_arg_id`` is the flat C++ arg-id resolved at
    # AST-build time by ``ASTTransformer._resolve_ndarray_kernel_arg_id`` and is required by the launch path to
    # support `@qd.data_oriented` member conditions (`qd.graph_do_while(self.counter)`) -- name-matching against
    # ``arg_metas`` only resolves top-level parameters.
    graph_do_while_levels: list[tuple[str, int, int]] | None = None
    # AST-build-time-resolved checkpoint metadata, indexed by internal cp_id. Empty for kernels without any
    # `with qd.checkpoint(...)` block. See `Kernel.checkpoint_yield_on_args` /
    # `Kernel.checkpoint_yield_on_cpp_arg_ids` / `Kernel.checkpoint_user_labels_by_cp_id` for what each entry means.
    # Restored alongside the C++-side cached kernel so the launch path can forward `yield_on=` arg-ids and
    # translate `from_checkpoint=` labels without re-running the AST transformer.
    checkpoint_yield_on_args: list[str | None] = []
    checkpoint_yield_on_cpp_arg_ids: list[int] = []
    checkpoint_user_labels_by_cp_id: list[int | None] = []
    # Parallel to ``checkpoint_user_labels_by_cp_id``: each entry is the dotted ``module.ClassQualName.MEMBER`` of
    # the original ``IntEnum`` member the user passed as ``cp_id``, or ``None`` if the user passed a plain int (or
    # for implicit auto-wrap checkpoints). On fast-cache restore the loader runs each entry through
    # ``_resolve_intenum_member`` to rebuild the IntEnum, preserving the documented contract that
    # ``qd.checkpoint(Stage.X, ...)`` round-trips ``Stage.X`` (not the underlying int) through
    # ``status.checkpoint`` and ``kernel.resume(from_checkpoint=...)`` -- pydantic coerces IntEnum to int at
    # ``CacheValue`` construction time so the parallel qualname column is what carries the enum identity.
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
    # `checkpoint_user_label_enum_qualnames` is derived from `checkpoint_user_labels_by_cp_id` here (rather than
    # being plumbed through a separate kwarg from `Kernel.materialize`) so callers never have to think about the
    # parallel column: they pass the live label list (which still holds the original ``IntEnum`` instances at
    # store time, before pydantic's int-coercion strips identity in ``CacheValue.__init__``), and the qualname
    # snapshot is recorded once here for the loader to consume.
    """
    Note that unlike other caches, this cache is not going to store the actual value we want.
    This cache is only used for verification that our cache key is valid. Big picture:
    - we have a cache key, based on args and top level kernel function
    - we want to use this to look up LLVM IR, in C++ side cache
    - however, before doing that, we first want to validate that the source code didn't change
        - i.e. is our cache key still valid?
    - the python side cache contains information we will use to verify that our cache key is valid
        - ie the list of function source infos

    Update! We are now going to store parameter pruning infomation, which is:
    - used_py_dataclass_parameters: set[str]

    Update 2: we are going to store the cache key used by the c++ kernel cache, so that we can use that
    to retrieve the immutable cached c++ kernel later, rather than, before, we were storing the c++
    cached kernel using the fast cache key, leading to bugs, when cached kernel file then had to be mutable.
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
    """Load a validated ``CacheValue`` for *cache_key* if one exists and its source hashes still match, else None.

    Returns the full ``CacheValue`` (rather than the historical 3-tuple) so callers can pick off the
    AST-transformer-produced metadata (graph_do_while levels, checkpoint tables) without the loader having to grow
    a new return slot every time we cache a new piece of AST output.
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
