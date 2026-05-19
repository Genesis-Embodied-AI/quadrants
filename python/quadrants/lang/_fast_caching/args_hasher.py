import dataclasses
import enum
import numbers
import time
from typing import Any, Sequence

import numpy as np

from quadrants import _logging, _tensor_wrapper
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants._tensor_wrapper import Tensor as _TensorWrapper
from quadrants.types.annotations import Template

from .._dataclass_util import create_flat_name
from .._ndarray import ScalarNdarray
from .._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from ..field import ScalarField
from ..kernel_arguments import ArgMetadata
from ..matrix import MatrixField, MatrixNdarray, VectorNdarray
from ..util import is_data_oriented, is_dataclass_instance
from .hash_utils import hash_iterable_strings

_FIELD_TYPES = (ScalarField, MatrixField)

try:
    import torch

    torch_type = torch.Tensor
except ImportError:
    torch_type = ()


g_num_calls = 0
g_num_args = 0
g_hashing_time = 0
g_repr_time = 0
g_num_ignored_calls = 0


FIELD_METADATA_CACHE_VALUE = "add_value_to_cache_key"

_DC_REPR_NONE = object()


# Sentinel returned by ``stringify_obj_type`` whenever fastcache cannot safely hash a value:
#   - Recognised-but-unsupported tensor-like type (``ScalarField`` / ``MatrixField``).
#   - Unrecognised type at a kernel-read path (no qualname fallback — see rules in fastcache.md).
#
# Containers (``dataclass_to_repr``, ``data_oriented`` branch, top-level ``hash_args`` loop) must propagate it upward
# — fastcache is disabled for the whole call and the caller writes the appropriate diagnostic.
class _FailFastcache:
    """Singleton sentinel; identity-compared."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_FAIL_FASTCACHE = _FailFastcache()


class FastcacheSkip(enum.Enum):
    """Why fastcache does not apply to this call."""

    FIELD_VIA_TENSOR = "field_via_tensor"
    WARN = "warn"


# Set when the fastcache skip is something callers should warn about (as opposed to a ``Field`` arriving through a
# ``qd.Tensor`` annotation, which is a normal silent path). Reset at the start of each ``hash_args`` call.
_should_warn = False


# Set of ``type(v).__qualname__`` strings we've already emitted the "unknown type at a kernel-read path"
# warning for. Lets the loop run thousands of times without spamming the log while still telling the user once
# that fastcache encountered an unrecognised type. Cleared by ``reset_unknown_type_warn_state`` (called from
# ``qd.init``) so each new test sees a clean log.
_warned_unknown_types: set[str] = set()


def reset_unknown_type_warn_state() -> None:
    """Clear the once-per-process warned-unknown-types set. Called from test setup / ``qd.init``."""
    _warned_unknown_types.clear()


def _mark_warn_if_not_tensor_annotation(arg_meta: ArgMetadata | None) -> None:
    """Flag that a warning is needed if the Field didn't arrive through a qd.Tensor annotation."""
    global _should_warn  # pylint: disable=global-statement
    if arg_meta is not None and arg_meta.annotation is not _TensorWrapper:
        _should_warn = True


def _mark_should_warn() -> None:
    global _should_warn  # pylint: disable=global-statement
    _should_warn = True


def _fail_unknown_type(obj: object, path: tuple[str, ...]) -> _FailFastcache:
    """Disable fastcache for the call when an unrecognised type appears at a kernel-read path.

    Two rules at work here (see ``docs/source/user_guide/fastcache.md`` "Pruning-driven argument hashing"):

      1. The fastcache key may *only* contain contributions from kernel-pruned paths — never a
         ``type(v).__qualname__`` fallback for an unrecognised type, because that hash captures type identity
         only and would silently mask a value-affecting change (e.g. a new tensor-like type whose dtype matters).

      2. We may not silently *discard* something at a kernel-read path on the basis that it's unrecognised —
         that would let unrecognised but codegen-affecting values escape the cache key and serve stale results.

    The only way to honour both rules is to fail the call's fastcache loudly, with a one-shot warning per type
    so the user can add explicit handling in ``stringify_obj_type``.
    """
    t = type(obj)
    qualname = f"{getattr(t, '__module__', '')}.{getattr(t, '__qualname__', t.__name__)}"
    if qualname not in _warned_unknown_types:
        _warned_unknown_types.add(qualname)
        _logging.warn(
            f"[FASTCACHE][UNKNOWN_TYPE] Unrecognised type {qualname} reached at kernel-read path {path}. "
            f"Fastcache is disabled for this call. Add explicit handling for this type to "
            f"``quadrants/lang/_fast_caching/args_hasher.py::stringify_obj_type``, or refactor the kernel "
            f"so it does not read this member."
        )
    _mark_should_warn()
    return _FAIL_FASTCACHE


def _child_flat(parent_flat: str | None, child_name: str) -> str | None:
    """Compute the flat name a kernel parameter would have if it pointed at this container's child.

    For a top-level arg ``state`` with child ``x``: ``__qd_state__qd_x``.
    For a deeper child ``state.dofs.x``: ``__qd_state__qd_dofs__qd_x`` (built incrementally).

    ``parent_flat`` is the *kernel-side* representation of this container's root:
      - top-level arg of a kernel: ``arg_meta.name`` (e.g. ``"state"``, ``"self"``) — no ``__qd_`` prefix.
      - any nested level: the already-computed ``__qd_…`` flat name.

    Returns ``None`` when ``parent_flat`` itself is ``None``, indicating "no path info available" — the caller
    must walk the child unconditionally (i.e. ignore ``pruning_paths`` for this branch).
    """
    if parent_flat is None:
        return None
    return create_flat_name(parent_flat, child_name)


def _is_path_used(pruning_paths: set[str] | None, child_flat: str | None) -> bool:
    """Return True if a child at ``child_flat`` should be hashed.

    - ``pruning_paths is None``: pre-pruning-info compile — hash everything.
    - ``child_flat is None``: caller could not compute a flat-name path (no parent_flat available) — hash
      everything as well, so we never accidentally drop a child we couldn't classify.
    - both non-None: only hash children whose flat name is in the set. Pruning's prefix-expansion step in
      ``Kernel.materialize`` guarantees that if any descendant of ``__qd_a__qd_b`` is used, ``__qd_a__qd_b``
      itself is also in the set, so this single membership check is sufficient to decide whether to descend.
    """
    if pruning_paths is None or child_flat is None:
        return True
    return child_flat in pruning_paths


def dataclass_to_repr(
    raise_on_templated_floats: bool,
    path: tuple[str, ...],
    arg: Any,
    pruning_paths: set[str] | None = None,
    parent_flat: str | None = None,
) -> str | _FailFastcache:
    """Hash a dataclass instance, optionally narrowed by pruning information.

    Returns ``_FAIL_FASTCACHE`` if any field's subtree hits a recognised-but-unsupported tensor type (``ScalarField`` /
    ``MatrixField``); otherwise a string.

    Pruning: if ``pruning_paths`` is non-None, only descend into fields whose flat name is in the set. Pruning's
    prefix-expansion step ensures the set already contains all ancestors of used leaves, so checking the immediate
    child's flat name is sufficient.
    """
    # PERF: For frozen dataclasses the repr never changes. Cache it on the instance to avoid repeated
    # ``dataclasses.fields()`` calls (which are slow due to extra runtime checks — see _template_mapper_hotpath.py
    # module docstring). The cache is stored as ``_qd_dc_repr`` via ``object.__setattr__`` to bypass frozen guards. A
    # cached ``_DC_REPR_NONE`` sentinel distinguishes "computed but not fast-cacheable" from "not yet computed".
    #
    # The cache is keyed by ``(is_frozen, pruning_paths is None)`` because a frozen dataclass's pruned repr depends on
    # the pruning_paths set — we use separate cache slots for pruned vs unpruned to avoid serving the wrong narrowing.
    cache_attr = "_qd_dc_repr" if pruning_paths is None else "_qd_dc_repr_narrow"
    is_frozen = type(arg).__hash__ is not None
    if is_frozen:
        cached = getattr(arg, cache_attr, None)
        if cached is _DC_REPR_NONE:
            return _FAIL_FASTCACHE
        if cached is not None and pruning_paths is None:
            # Narrow cache may be stale if pruning_paths set changed; only reuse the unpruned cache.
            return cached
    repr_l = []
    for field in dataclasses.fields(arg):
        child_flat = _child_flat(parent_flat, field.name)
        if not _is_path_used(pruning_paths, child_flat):
            continue
        child_value = getattr(arg, field.name)
        _repr = stringify_obj_type(
            raise_on_templated_floats,
            path + (field.name,),
            child_value,
            arg_meta=None,
            pruning_paths=pruning_paths,
            parent_flat=child_flat,
        )
        if _repr is _FAIL_FASTCACHE:
            if isinstance(child_value, _FIELD_TYPES) and field.type is not _TensorWrapper:
                _mark_should_warn()
            if is_frozen:
                try:
                    object.__setattr__(arg, cache_attr, _DC_REPR_NONE)
                except AttributeError:
                    pass
            return _FAIL_FASTCACHE
        full_repr = f"{field.name}: ({_repr})"
        if field.metadata.get(FIELD_METADATA_CACHE_VALUE, False):
            full_repr += f" = {child_value}"
        repr_l.append(full_repr)
    result = "[" + ",".join(repr_l) + "]"
    if is_frozen and pruning_paths is None:
        try:
            object.__setattr__(arg, cache_attr, result)
        except AttributeError:
            pass
    return result


def _is_template(arg_meta: ArgMetadata | None) -> bool:
    if arg_meta is None:
        return False
    annot = arg_meta.annotation
    return annot is Template or isinstance(annot, Template)


def stringify_obj_type(
    raise_on_templated_floats: bool,
    path: tuple[str, ...],
    obj: object,
    arg_meta: ArgMetadata | None,
    pruning_paths: set[str] | None = None,
    parent_flat: str | None = None,
) -> str | _FailFastcache:
    """Convert ``obj`` into a deterministic string that contributes to the fastcache key.

    Return contract:
      - ``str``: hashable; the returned string contributes to the cache key.
      - ``_FAIL_FASTCACHE``: fastcache cannot safely hash this value — caller must propagate upward and
        disable fastcache for the whole call. Triggered by:
          * Recognised-but-unsupported tensor-like type (``ScalarField`` / ``MatrixField``).
          * Unrecognised type at this kernel-read path (see ``_fail_unknown_type``).

    Two rules from ``docs/source/user_guide/fastcache.md`` "Pruning-driven argument hashing" govern this function:

      1. The cache key may *only* include contributions from paths that pruning has marked kernel-accessed
         (``pruning_paths``). Container walkers (dataclass + data_oriented) check ``_is_path_used`` per child and
         skip non-pruned subtrees — kernel-unread paths are *guaranteed* not to affect codegen so this is safe by
         construction.

      2. At paths the kernel *does* read, unrecognised types must not be silently dropped or hashed by type-name —
         fastcache fails the call (loudly, with a one-shot warning) so the gap can be closed.

    Parameters:
      - ``arg_meta``: non-``None`` only for top-level kernel args and for ``@qd.data_oriented`` members. Determines
        whether primitive values are baked into the cache key (template-position primitives and all primitive members
        of data-oriented containers).
      - ``pruning_paths``: optional set of kernel-accessed flat names from L1 cache. When provided,
        ``dataclass_to_repr`` and the ``data_oriented`` branch below descend only into children whose flat name is in
        the set. Pruning info is populated by ``ASTTransformer.build_Name`` / ``build_Attribute`` (kernel-arg-rooted
        chains) plus ``Pruning.fold_struct_nd_paths`` (ndarray accesses through data_oriented containers).
      - ``parent_flat``: the flat-name prefix for ``obj``'s children (e.g. ``__qd_self`` if ``obj`` is the ``self``
        arg of a data_oriented kernel). Used together with ``pruning_paths`` to compute each child's flat name for
        the narrow-walk lookup.
    """
    # ``qd.Tensor`` wrappers passed as struct fields. The top-level kernel-arg unwrap hook in ``Kernel.__call__``
    # strips wrappers off positional / keyword args before the fastcache hasher sees them, but the dataclass /
    # data-oriented walkers below do raw ``getattr`` to fetch struct fields, so a wrapper stored as a struct field
    # arrives here un-stripped. Without this branch the hasher would hash the wrapper as an unknown type instead of
    # unwrapping to the recognised impl. See ``perso_hugh/doc/quadrants-tensor.md`` §8.14.
    #
    # PERF-CRITICAL: the ``_any_tensor_constructed`` guard makes this check zero-cost when no ``qd.Tensor`` has been
    # created. ``type(obj) in _TENSOR_WRAPPER_TYPES`` is used instead of ``isinstance`` because it is a pointer
    # comparison (~10 ns) vs an MRO walk (~100–200 ns). Do not replace with isinstance or remove the guard.
    if (
        _tensor_wrapper._any_tensor_constructed and type(obj) in _TENSOR_WRAPPER_TYPES
    ):  # pyright: ignore[reportOptionalMemberAccess]
        obj = obj._unwrap()  # pyright: ignore[reportAttributeAccessIssue]
    arg_type = type(obj)
    _layout = getattr(obj, "_qd_layout", None)
    _layout_tag = "" if _layout is None else f"-L{_layout!r}"
    if isinstance(obj, ScalarNdarray):
        return f"[nd-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, VectorNdarray):
        return f"[ndv-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, ScalarField):
        # Recognised-but-unsupported: shape/dtype affect kernel codegen but fastcache doesn't yet hash them. Disable
        # fastcache for the whole call.
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        return _FAIL_FASTCACHE
    if isinstance(obj, MatrixNdarray):
        return f"[ndm-{obj.m}-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, torch_type):
        return f"[pt-{obj.dtype}-{obj.ndim}]"  # type: ignore
    if isinstance(obj, np.ndarray):
        return f"[np-{obj.dtype}-{obj.ndim}]"
    if isinstance(obj, MatrixField):
        # Recognised-but-unsupported, same as ScalarField above.
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        return _FAIL_FASTCACHE
    if is_dataclass_instance(obj):
        return dataclass_to_repr(
            raise_on_templated_floats, path, obj, pruning_paths=pruning_paths, parent_flat=parent_flat
        )
    if is_data_oriented(obj):
        # Walk the data_oriented container's members, narrowed by pruning info — the kernel-compile path records
        # every kernel-accessed attribute chain (ndarrays via ``_promote_ndarray_if_declared`` +
        # ``Pruning.fold_struct_nd_paths``; primitives, opaque members, nested structs via
        # ``ASTTransformer.build_Attribute``'s ``_qd_arg_chain`` propagation calling ``pruning.mark_used``). Members
        # not in ``pruning_paths`` are *guaranteed* not to affect kernel codegen because the kernel cannot read them.
        # Dropping them from the hash satisfies rule 1 (cache only pruned paths).
        child_repr_l = ["da"]
        try:
            _asdict = getattr(obj, "_asdict")
            _dict = _asdict()
        except AttributeError:
            _dict = obj.__dict__
        for k, v in _dict.items():
            # Skip Quadrants method-descriptor cache entries. ``QuadrantsCallable.__get__`` stashes the per-instance
            # ``BoundQuadrantsCallable`` on ``instance.__dict__`` so subsequent ``instance.method`` lookups skip the
            # descriptor allocation; those entries are not data and must not invalidate the fastcache key.
            v_type = type(v)
            if v_type is QuadrantsCallable or v_type is BoundQuadrantsCallable:
                continue
            child_flat = _child_flat(parent_flat, k)
            if not _is_path_used(pruning_paths, child_flat):
                continue
            _child_repr = stringify_obj_type(
                raise_on_templated_floats,
                (*path, k),
                v,
                ArgMetadata(Template, ""),
                pruning_paths=pruning_paths,
                parent_flat=child_flat,
            )
            if _child_repr is _FAIL_FASTCACHE:
                return _FAIL_FASTCACHE
            child_repr_l.append(f"{k}: {_child_repr}")
        return ", ".join(child_repr_l)
    if issubclass(arg_type, (numbers.Number, np.number)):
        if _is_template(arg_meta):
            if raise_on_templated_floats and isinstance(obj, float):
                raise ValueError("Floats should not be used in template parameters.")
            # cache value too
            return f"{arg_type}={obj}"
        return str(arg_type)
    if arg_type is np.bool_:
        # np is deprecating bool. Treat specially/carefully
        if _is_template(arg_meta):
            # cache value too
            return f"np.bool_={obj}"
        return "np.bool_"
    if isinstance(obj, enum.Enum):
        return f"enum-{obj.name}-{obj.value}"
    # Unrecognised type at a kernel-read path — fail fastcache loudly. See ``_fail_unknown_type``.
    return _fail_unknown_type(obj, path)


def hash_args(
    raise_on_templated_floats: bool,
    args: Sequence[Any],
    arg_metas: Sequence[ArgMetadata | None],
    pruning_paths: set[str] | None = None,
) -> str | FastcacheSkip:
    """Return the args hash string, or a ``FastcacheSkip`` explaining why hashing failed.

    Parameters:
      - ``pruning_paths``: optional set of kernel-accessed flat names from the L1 cache (or freshly populated
        after a cold compile). When provided, the container walkers skip children whose flat name is not in
        the set; this is what keeps the cache key narrow and brittleness-free (no opaque-typed member can
        affect the key unless the kernel actually reads it).

    Fastcache is disabled (``FastcacheSkip`` returned) when either:
      - a recognised-but-unsupported tensor-like type (``ScalarField`` / ``MatrixField``) is encountered at a
        kernel-read path, OR
      - an unrecognised type is encountered at a kernel-read path (see ``_fail_unknown_type``).

    Both cases are loud: ``FastcacheSkip.WARN`` triggers an ``[INVALID_FUNC]`` log line and the unknown-type
    branch additionally emits a one-shot ``[UNKNOWN_TYPE]`` warning identifying the offending type.
    """
    global g_num_calls, g_num_args, g_hashing_time, g_repr_time, g_num_ignored_calls, _should_warn  # pylint: disable=global-statement
    _should_warn = False
    g_num_calls += 1
    g_num_args += len(args)
    hash_l = []
    if len(args) != len(arg_metas):
        raise RuntimeError(
            f"Number of args passed in {len(args)} doesnt match number of declared args {len(arg_metas)}"
        )
    for i_arg, arg in enumerate(args):
        start = time.time()
        arg_meta = arg_metas[i_arg]
        # Top-level arg flat name: matches the kernel-side ``arg_meta.name`` (no ``__qd_`` prefix at the root).
        # Used by the narrow walk to construct child flat names compatible with ``pruning.used_vars_by_func_id``.
        top_flat = arg_meta.name if arg_meta is not None else None
        _hash = stringify_obj_type(
            raise_on_templated_floats,
            (str(i_arg),),
            arg,
            arg_meta,
            pruning_paths=pruning_paths,
            parent_flat=top_flat,
        )
        g_repr_time += time.time() - start
        if _hash is _FAIL_FASTCACHE:
            g_num_ignored_calls += 1
            return FastcacheSkip.WARN if _should_warn else FastcacheSkip.FIELD_VIA_TENSOR
        # All other return values are valid strings (qualname fallback handles unrecognised types).
        hash_l.append(_hash)
    start = time.time()
    res = hash_iterable_strings(hash_l)
    g_hashing_time += time.time() - start
    return res


def dump_stats() -> None:
    print("args hasher dump stats")
    print("total calls", g_num_calls)
    print("ignored calls", g_num_ignored_calls)
    print("total args", g_num_args)
    print("hashing time", g_hashing_time)
    print("arg representation time", g_repr_time)
