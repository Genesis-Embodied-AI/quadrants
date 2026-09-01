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
from .._final_dataclass_fields import (
    final_field_names,
    final_scalar_key,
    subtree_has_final_fields,
)
from .._ndarray import ScalarNdarray
from .._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from ..field import ScalarField
from ..kernel_arguments import ArgMetadata
from ..matrix import MatrixField, MatrixNdarray, VectorNdarray
from ..util import is_data_oriented, is_dataclass_instance, wants_runtime_primitives
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

# arg_meta used when walking the children of a ``@qd.data_oriented(template_primitives=False)`` object. Its annotation
# is non-Template (so primitive members contribute their *type* only, not their value, since they are lifted to runtime
# scalar args rather than baked into the kernel) and non-Tensor (so a stray ``qd.field`` child still triggers the
# warn-and-disable path, exactly as for a normal data_oriented object).
_NON_TEMPLATE_CHILD_META = ArgMetadata(None, "")


# Returned by ``stringify_obj_type`` when a value cannot be safely hashed (unsupported tensor-like type, or an
# unrecognised type at a kernel-read path). Every container walker has to propagate it upward: fastcache is then off for
# the whole call and the caller writes the diagnostic.
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


# Set when the skip is worth warning about, as opposed to a ``Field`` arriving through a ``qd.Tensor`` annotation, which
# is a normal silent path. Reset at the start of each ``hash_args`` call.
_should_warn = False


# ``type(v).__qualname__`` strings already warned about, so a hot launch loop reports an unrecognised type once instead
# of thousands of times.
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

    Neither alternative is safe: a ``__qualname__`` fallback captures type identity only, and so hides a
    value-affecting change, while skipping the value lets a codegen-affecting one escape the key entirely. See
    ``docs/source/user_guide/fastcache.md`` "Pruning-driven argument hashing".
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

    ``state`` + ``x`` -> ``__qd_state__qd_x``; ``state.dofs`` + ``x`` -> ``__qd_state__qd_dofs__qd_x``. A ``None``
    ``parent_flat`` means no path info is available, and propagates so the caller walks the child unconditionally.
    """
    if parent_flat is None:
        return None
    return create_flat_name(parent_flat, child_name)


def _is_path_used(pruning_paths: set[str] | None, child_flat: str | None) -> bool:
    """Return True if a child at ``child_flat`` should be hashed. Unknown (``None``) means hash it.

    Membership of the child alone also decides whether to descend into it: ``Kernel.materialize``'s
    prefix-expansion step puts every ancestor of a used leaf in the set.
    """
    if pruning_paths is None or child_flat is None:
        return True
    return child_flat in pruning_paths


def dataclass_to_repr(
    raise_on_templated_floats: bool,
    path: tuple[str, ...],
    arg: Any,
    annotated_type: type | None = None,
    pruning_paths: set[str] | None = None,
    parent_flat: str | None = None,
) -> str | _FailFastcache:
    """Hash a dataclass instance, descending only into fields listed in ``pruning_paths`` (when given).

    Returns ``_FAIL_FASTCACHE`` if any visited subtree holds an unsupported tensor type.
    """
    # PERF: a frozen dataclass's repr never changes, so cache it on the instance (``dataclasses.fields()`` is slow, see
    # the _template_mapper_hotpath.py module docstring). ``_DC_REPR_NONE`` caches a failure verdict.
    #
    # Hash the *declared* type's fields (``annotated_type``, when given) rather than the runtime subclass's, so a
    # subclass carrying extra app-only fields still fast-caches on the base's field set. Only the runtime type's own
    # unpruned view is cacheable: the repr/verdict depend on the visited (pruning) set and the active field set, so a
    # narrower pruning set - or a subclass passed under a different base - must not inherit another view's verdict.
    # Final-bearing subtrees are excluded too (their repr recomputes each launch to re-run ``final_scalar_key``).
    field_source = annotated_type if annotated_type is not None else type(arg)
    is_frozen = type(arg).__hash__ is not None
    final_names = final_field_names(field_source)
    cacheable = (
        is_frozen and field_source is type(arg) and pruning_paths is None and not subtree_has_final_fields(field_source)
    )
    if cacheable:
        cached = getattr(arg, "_qd_dc_repr", None)
        if cached is _DC_REPR_NONE:
            return _FAIL_FASTCACHE
        if cached is not None:
            return cached
    repr_l = []
    for field in dataclasses.fields(field_source):
        child_value = getattr(arg, field.name)
        if field.name in final_names:
            # ``final_scalar_key`` rather than ``stringify_obj_type``: the latter has no bare-``str`` case, so a
            # ``Final[str]`` would disable fastcache for the whole arg. Included even when ``pruning_paths`` does not
            # mention the field - the value is baked into the generated code, so a key that omits it could serve a
            # kernel baked with a different constant.
            repr_l.append(f"{field.name}: (final) = {final_scalar_key(child_value)}")
            continue
        child_flat = _child_flat(parent_flat, field.name)
        if not _is_path_used(pruning_paths, child_flat):
            continue
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
            if cacheable:
                try:
                    object.__setattr__(arg, "_qd_dc_repr", _DC_REPR_NONE)
                except AttributeError:
                    pass
            return _FAIL_FASTCACHE
        full_repr = f"{field.name}: ({_repr})"
        if field.metadata.get(FIELD_METADATA_CACHE_VALUE, False):
            full_repr += f" = {child_value}"
        repr_l.append(full_repr)
    result = "[" + ",".join(repr_l) + "]"
    if cacheable:
        try:
            object.__setattr__(arg, "_qd_dc_repr", result)
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

    Returns ``_FAIL_FASTCACHE`` for a value that cannot be safely hashed, which callers must propagate upward to
    disable fastcache for the whole call.

    Parameters:
      - ``arg_meta``: non-``None`` only for top-level kernel args and for ``@qd.data_oriented`` members. Determines
        whether primitive values are baked into the cache key (template-position primitives and all primitive members
        of data-oriented containers).
      - ``pruning_paths``: kernel-accessed flat names from the L1 cache. When provided, the container walkers descend
        only into children whose flat name is in the set - a path the kernel cannot read cannot affect codegen. See
        ``docs/source/user_guide/fastcache.md`` "Pruning-driven argument hashing".
      - ``parent_flat``: flat-name prefix for ``obj``'s children (e.g. ``__qd_self``), used to build those names.
    """
    # ``Kernel.__call__`` unwraps ``qd.Tensor`` for positional / keyword args, but the walkers below reach struct fields
    # by raw ``getattr``, so a wrapper stored as a field arrives here un-stripped.
    #
    # PERF-CRITICAL: the ``_any_tensor_constructed`` guard keeps this free for programs that use no ``qd.Tensor``, and
    # the ``type(obj) in ...`` test is a pointer comparison rather than an MRO walk. Don't switch it to isinstance.
    if (
        _tensor_wrapper._any_tensor_constructed and type(obj) in _TENSOR_WRAPPER_TYPES
    ):  # pyright: ignore[reportOptionalMemberAccess]
        obj = obj._unwrap()  # pyright: ignore[reportAttributeAccessIssue]
    arg_type = type(obj)
    _layout = getattr(obj, "_qd_layout", None)
    _layout_tag = "" if _layout is None else f"-L{_layout!r}"
    # The grad tags below are not cosmetic: ``insert_ndarray_param`` bakes grad-presence into the parameter struct
    # layout, so two ndarrays alike but for ``needs_grad`` must hash distinctly or the artifact is launched against a
    # slot of the other shape (silent miscomputation or OOB).
    if isinstance(obj, ScalarNdarray):
        _grad_tag = "-g" if obj.grad is not None else ""
        return f"[nd-{obj.dtype}-{len(obj.shape)}{_layout_tag}{_grad_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, VectorNdarray):
        _grad_tag = "-g" if obj.grad is not None else ""
        return f"[ndv-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}{_grad_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, ScalarField):
        # disabled for now, because we need to think about how to handle field offset
        # etc
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        return _FAIL_FASTCACHE
    if isinstance(obj, MatrixNdarray):
        _grad_tag = "-g" if obj.grad is not None else ""
        return f"[ndm-{obj.m}-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}{_grad_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, torch_type):
        return f"[pt-{obj.dtype}-{obj.ndim}]"  # type: ignore
    if isinstance(obj, np.ndarray):
        return f"[np-{obj.dtype}-{obj.ndim}]"
    if isinstance(obj, MatrixField):
        # disabled for now, because we need to think about how to handle field offset
        # etc
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        return _FAIL_FASTCACHE
    if is_dataclass_instance(obj):
        # Pass the declared dataclass type so an annotated subclass hashes via the base's fields. Nested / data_oriented
        # children have no dataclass annotation here (arg_meta is None or Template), so they fall back to runtime type.
        annotated_type = None
        if arg_meta is not None:
            ann = arg_meta.annotation
            if isinstance(ann, type) and dataclasses.is_dataclass(ann) and isinstance(obj, ann):
                annotated_type = ann
        return dataclass_to_repr(
            raise_on_templated_floats, path, obj, annotated_type, pruning_paths=pruning_paths, parent_flat=parent_flat
        )
    if is_data_oriented(obj):
        # Narrowed by pruning info: a member the kernel cannot read cannot affect codegen, so it stays out of the key.
        child_repr_l = ["da"]
        try:
            _asdict = getattr(obj, "_asdict")
            _dict = _asdict()
        except AttributeError:
            _dict = obj.__dict__
        # A normal @qd.data_oriented bakes primitive members into the kernel (value in the cache key); one declared
        # with template_primitives=False lifts them to runtime args (type only - value must NOT enter the key, or it
        # would recompile on every value change, defeating the feature). Decide per object, since the flag is per class.
        child_meta = _NON_TEMPLATE_CHILD_META if wants_runtime_primitives(obj) else ArgMetadata(Template, "")
        for k, v in _dict.items():
            # ``QuadrantsCallable.__get__`` stashes bound callables on ``instance.__dict__``; they are not data.
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
                child_meta,
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
    if obj is None:
        # ``None`` is a singleton, so its type fully determines its value and a constant tag is a complete cache key.
        return "None"
    return _fail_unknown_type(obj, path)


def hash_args(
    raise_on_templated_floats: bool,
    args: Sequence[Any],
    arg_metas: Sequence[ArgMetadata | None],
    pruning_paths: set[str] | None = None,
) -> str | FastcacheSkip:
    """Return the args hash string, or a ``FastcacheSkip`` explaining why hashing failed.

    ``pruning_paths`` are the kernel-accessed flat names from the L1 cache; children outside the set are skipped, so
    an opaque-typed member cannot affect the key unless the kernel reads it. A skip is always reported: the caller
    logs ``[INVALID_FUNC]`` for ``FastcacheSkip.WARN``, and unrecognised types also warn from ``_fail_unknown_type``.
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
        # Root flat name carries no ``__qd_`` prefix, matching ``pruning.used_vars_by_func_id``.
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
