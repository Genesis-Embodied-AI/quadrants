# pyright: reportAttributeAccessIssue=false

"""
This function '_extract_arg' is called so often during physics simulation with Genesis that it becomes a major
bottleneck for simple scenes running faster than 10M FPS. In practice, it adds about 100% overhead when running 20M
FPS, and things get worst as FPS increases. At this scale, it is necessary to chase not just us (microseconds) but
also ns (nanoseconds). This requires special optimization technics, to name a few:
* Avoid attribute lookup as much as possible, indirectly for submodules, instance methods and class attributes.
* Do not define local-scope function, because the definition itself is costly, and called methods that are not in
  global scope is slowe
* Avoid function indirection as much as possible, especially for very short method
* Prefer list comprehension over tuple + generator
* Prefer using 'in' operator of set if possible, otherwise tuple, instead of list
* Avoid redundant operations by inlining complementary methods, i.e. 'dataclasses.is_dataclass' in conjunction with
  'dataclasses.fields'.
* Prefer using 'arg_type = type(arg)' plus 'issubclass' over 'isinstance' when doing many checks successively
* Prefer 'is' operator over '==', 'isinstance' and 'issubclass' whenever it is applicable
* Order branches by hit probability
* Guard complex manually debug checks with 'if __debug__ and __builtins__["__debug__"]' to allow disabling them at
  runtime instead of compile time only
* Use 'getattr' on class rather than instances for static properties

A direct consequence of this breaking type checking because pyright is not able to understand that 'arg_type' is
immutably bound to 'type(arg)'. Moreover, some privates fields of standard module 'dataclass' had to be imported as
a consequence of inlining 'is_dataclass' and 'fields'.
"""

import dataclasses
import sys
import weakref
from dataclasses import _FIELD, _FIELDS
from typing import Any, Union

import numpy as np

from quadrants import _tensor_wrapper
from quadrants._lib import core as _qd_core
from quadrants._tensor import (
    _TENSOR_T_FIELD_MARKER,
    _TENSOR_T_NDARRAY_MARKER,
)
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang._dataclass_util import create_flat_name
from quadrants.lang._final_dataclass_fields import (
    final_field_names,
    final_scalar_key,
    subtree_has_final_fields,
)
from quadrants.lang._ndarray import Ndarray
from quadrants.lang.any_array import AnyArray
from quadrants.lang.buffer_view import BufferView as BufferViewInstance
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import MatrixType
from quadrants.lang.snode import SNode
from quadrants.lang.util import (
    is_data_oriented,
    is_dataclass_instance,
    to_quadrants_type,
)
from quadrants.types import (
    buffer_view_type,
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
)

# Default ndarray annotation for Tensor-resolved-as-ndarray. Defining at module scope avoids re-allocating per call.
# boundary defaults to UNSAFE (the same default a bare ``qd.types.ndarray()`` would produce).
_TENSOR_T_NDARRAY_ANNOTATION = ndarray_type.NdarrayType()

AnnotationType = Union[
    template,
    ndarray_type.NdarrayType,
    sparse_matrix_builder,
    Any,
]


def annotation_has_final_subtree(annotation: Any) -> bool:
    """True iff ``annotation`` is a dataclass type whose transitive subtree declares a ``Final`` field. The ``_FIELDS``
    probe is needed because ``subtree_has_final_fields`` would raise on the non-dataclass annotation shapes a kernel
    arg can carry. ``TemplateMapper.lookup`` uses it to disable its instance-keyed cache for a Final-bearing mapper.
    """
    return getattr(annotation, _FIELDS, None) is not None and subtree_has_final_fields(annotation)


_ExprCxx = _qd_core.ExprCxx
_composite_mutable_types = {list, dict, set}
# ``type(None)`` belongs here so ``None`` arguments are treated like other primitives and excluded from the
# weakref-based spec-key cache tracker in ``TemplateMapper.lookup``: ``weakref.ref(None)`` raises ``TypeError``, and
# ``None`` is an immortal singleton that never needs lifetime tracking. Without it, the spec-key cache entry for a
# ``None`` argument is dropped and every such launch re-runs full spec-key extraction.
_primitive_types = {int, float, bool, type(None)}
_np_ndarray = np.ndarray


def _is_external_array(arg: Any) -> bool:
    if isinstance(arg, _np_ndarray):
        return True
    torch = sys.modules.get("torch")
    if torch is not None:
        return isinstance(arg, torch.Tensor)
    return False


# Fallback for ``__slots__`` classes, which have no ``__dict__`` to hold the per-instance path cache that
# ``_struct_nd_paths_for`` normally stashes. Per-class caching cannot express an attribute structure that differs per
# instance, so such classes keep that limitation.
_struct_nd_paths_cache: dict[type, list[tuple]] = {}


def _build_struct_nd_paths(obj: Any, prefix: tuple, out: list, _seen: "set[int] | None" = None) -> None:
    # ``_seen`` is required, not defensive: Genesis object graphs have cross-references (``solver -> scene -> sim ->
    # solver``).
    if _seen is None:
        _seen = {id(obj)}
    if is_dataclass_instance(obj):
        children = ((f.name, getattr(obj, f.name)) for f in dataclasses.fields(obj))
    else:
        # ``NamedTuple`` (decorated as ``@qd.data_oriented``) has no instance ``__dict__`` — fall back to ``_asdict()``
        # which materialises a dict view of the named fields. Mirrors the same fallback in
        # ``args_hasher.stringify_obj_type`` so the per-class path cache here picks up ndarray members on NamedTuples
        # too (regression covered by ``test_args_hasher_named_tuple``).
        try:
            children = obj._asdict().items()
        except AttributeError:
            children = obj.__dict__.items()
    for k, v in children:
        chain = prefix + (k,)
        if type(v) in _TENSOR_WRAPPER_TYPES:
            v = v._unwrap()
        v_type = type(v)
        if issubclass(v_type, Ndarray):
            out.append(chain)
        elif is_data_oriented(v) or is_dataclass_instance(v):
            v_id = id(v)
            if v_id in _seen:
                continue
            _seen.add(v_id)
            _build_struct_nd_paths(v, chain, out, _seen)


def _struct_nd_paths_for(arg: Any) -> list[tuple]:
    """Return the attribute paths (tuples of attr names) at which ``Ndarray`` instances are reachable from ``arg``,
    caching them on the instance itself as ``_qd_nd_paths``.

    The cache has to be per instance: ``@qd.data_oriented`` classes can have different attribute sets across
    instances of one class (Genesis ``DataManager`` with vs without ``requires_grad``), and a ``qd.Tensor`` member
    can swap backends within one instance's lifetime.

    Limitation: an ndarray attribute attached *after* the instance's first kernel call is not tracked until the cache
    is dropped (``del arg.__dict__['_qd_nd_paths']``).
    """
    # ``__dict__[...]`` rather than ``getattr``: some third-party metaclasses (e.g. Pydantic) recurse infinitely on
    # probe-style ``getattr`` of an unknown name.
    try:
        return arg.__dict__["_qd_nd_paths"]
    except (AttributeError, KeyError):
        pass
    cls = type(arg)
    paths = _struct_nd_paths_cache.get(cls)
    if paths is not None:
        return paths
    paths = []
    _build_struct_nd_paths(arg, (), paths)
    try:
        object.__setattr__(arg, "_qd_nd_paths", paths)
    except AttributeError:
        # No ``__dict__`` to stash in - degrade to per-class caching.
        _struct_nd_paths_cache[cls] = paths
    return paths


def chain_has_mutable_container(args, template_arg_idx, attr_chain) -> bool:
    """Return True if any container along ``attr_chain`` from ``args[template_arg_idx]`` down to (but excluding) the
    leaf ndarray attribute is mutable in a way that lets it rebind its child attribute. Such a parent makes
    ``id(args[template_arg_idx])`` alone insufficient to uniquely identify the leaf, so the leaf id must be folded
    into the launch-context cache key.

    A container is "mutable" here iff:
    - its type has ``__hash__ is None`` (Python sets this for non-frozen ``@dataclass(eq=True)`` types), or
    - it is a ``@qd.data_oriented`` instance (these inherit ``object.__hash__`` so the ``__hash__ is None`` check
      misses them; they support normal attribute assignment).

    Walks all parents from the root down to ``attr_chain[:-1]`` — the final entry is the leaf itself, whose own
    mutability does not affect rebinding by its parent. Returns on the first mutable parent.
    """
    cur = args[template_arg_idx]
    if type(cur).__hash__ is None or is_data_oriented(cur):
        return True
    for attr_name in attr_chain[:-1]:
        cur = getattr(cur, attr_name)
        if type(cur).__hash__ is None or is_data_oriented(cur):
            return True
    return False


def _collect_struct_nd_descriptors(arg: Any, out: list) -> None:
    """Emit per-ndarray shape descriptors ``(joined-path, element_type, ndim, needs_grad, layout)`` for every ndarray
    reachable from ``arg``. Used by the template-mapper to refine the spec key for ``@qd.data_oriented`` args holding
    ndarrays — see the data_oriented branch in ``_extract_arg``.
    """
    # A cached path's leaf can stop being an ``Ndarray`` mid-lifetime (a ``qd.Tensor`` impl swapped for a
    # ``MatrixField``), hence the skip below rather than reading Ndarray-only accessors off it.
    for chain in _struct_nd_paths_for(arg):
        v = arg
        for a in chain:
            v = getattr(v, a)
        if type(v) in _TENSOR_WRAPPER_TYPES:
            v = v._unwrap()
        if not isinstance(v, Ndarray):
            continue
        # ``shape`` is legitimately ``None`` for an uninitialised ``_physical_shape``.
        shape = v.shape
        if shape is None:
            continue
        type_id = id(v.element_type)
        element_type = type_id if type_id in primitive_types.type_ids else v.element_type
        out.append((".".join(chain), element_type, len(shape), v.grad is not None, v._qd_layout))


def _extract_arg(raise_on_templated_floats: bool, arg: Any, annotation: AnnotationType, arg_name: str) -> Any:
    # ``qd.Tensor`` wrappers passed as struct fields. Top-level kernel-arg unwrap in ``Kernel.__call__`` covers direct
    # args, but the dataclass-field recursion at the bottom of this function walks struct attributes via raw
    # ``getattr``, so a wrapper stored as a struct field arrives here un-stripped with its declared annotation (e.g.
    # ``qd.types.NDArray[qd.f32, 2]``). Without this unwrap the function falls through to the "external arrays" path
    # (line ~149) which technically reads ``.shape`` off the wrapper but produces a meaningless cache key. See
    # ``perso_hugh/doc/quadrants-tensor.md`` §8.14. Idempotent for top-level args.
    #
    # PERF-CRITICAL: The _any_tensor_constructed guard makes this check zero-cost when no qd.Tensor has been created.
    # This function runs on *every* argument of *every* kernel invocation. ``type(arg) in _TENSOR_WRAPPER_TYPES`` is
    # used instead of ``isinstance`` because it is a pointer comparison (~10 ns) vs an MRO walk (~100–200 ns). Do not
    # replace with isinstance or remove the guard.
    if (
        _tensor_wrapper._any_tensor_constructed and type(arg) in _TENSOR_WRAPPER_TYPES
    ):  # pyright: ignore[reportOptionalMemberAccess]
        arg = arg._unwrap()
    annotation_type = type(annotation)
    # qd.Tensor: value-dispatch. Ndarray-shaped values flow through the ndarray feature path; everything else falls
    # through to the template path (Field, SNode, primitives). Both branches are salted with a marker so cache keys
    # disambiguate. The annotation is the wrapper *class* (``qd.Tensor``); ``arg`` is always a bare impl by the time
    # we get here (``Kernel.__call__`` unwraps ``Tensor`` instances).
    if annotation is _TensorClass:
        if type(arg) in _TENSOR_WRAPPER_TYPES:
            arg = arg._unwrap()
        arg_type = type(arg)
        if issubclass(arg_type, (Ndarray, AnyArray)) or _is_external_array(arg):
            return (_TENSOR_T_NDARRAY_MARKER,) + tuple(
                _extract_arg(
                    raise_on_templated_floats,
                    arg,
                    _TENSOR_T_NDARRAY_ANNOTATION,
                    arg_name,
                )
            )
        # Fall through to the template path below by retargeting the annotation. Wrap the result with a field marker
        # so its cache entry is distinct from the ndarray branch above.
        annotation = template
        annotation_type = type(template)
        return (_TENSOR_T_FIELD_MARKER,) + (_extract_arg(raise_on_templated_floats, arg, template, arg_name),)
    arg_type = type(arg)
    if annotation is template or annotation_type is template:
        if arg_type is SNode:
            return arg.ptr
        if arg_type is Expr:
            return arg.ptr.get_underlying_ptr_address()
        if arg_type is _ExprCxx:
            return arg.get_underlying_ptr_address()
        if issubclass(arg_type, tuple):  # Handle all tuple-based containers, incl. NamedTuple
            return tuple([_extract_arg(raise_on_templated_floats, item, annotation, arg_name) for item in arg])
        if issubclass(arg_type, Ndarray):
            raise QuadrantsRuntimeTypeError(
                "Ndarray shouldn't be passed in via `qd.template()`, please annotate your kernel using `qd.types.ndarray(...)` instead"
            )
        if arg_type in _composite_mutable_types:
            # [Composite arguments] Return weak reference to the object
            # Quadrants kernel will cache the extracted arguments, thus we can't simply return the original argument.
            # Instead, a weak reference to the original value is returned to avoid memory leak.

            # TODO(zhanlue): replacing "tuple(args)" with "hash of argument values"
            # This can resolve the following issues:
            # 1. Invalid weak-ref will leave a dead(dangling) entry in both caches: "self.mapping" and "self.compiled_functions"
            # 2. Different argument instances with same type and same value, will get templatized into separate kernels.
            return weakref.ref(arg)
        if is_data_oriented(arg):
            # Same memory-leak avoidance as above — keep ``weakref.ref(arg)`` so the spec key never holds a strong
            # reference to user state. But for data_oriented containers that hold ``Ndarray`` members, the live
            # ``weakref`` alone is too coarse: same instance with ``state.x = other_ndarray`` of a different dtype/ndim
            # would re-use the previously-compiled kernel, which was specialised for the old shape. Walk the reachable
            # ndarrays and prepend their shape descriptors so dtype/ndim changes trigger re-specialisation. Mirrors what
            # the dataclass branch below does via ``annotation_fields``.
            #
            # Containers with no ndarrays keep the original short-path (one spec per instance via weakref) so this is
            # a no-op for the existing data_oriented + qd.field workloads (genesis field-backend).
            if type(arg).__dict__.get("_qd_stable_members"):
                return weakref.ref(arg)
            nd_descriptors: list = []
            _collect_struct_nd_descriptors(arg, nd_descriptors)
            if nd_descriptors:
                return (id(type(arg)), tuple(nd_descriptors), weakref.ref(arg))
            return weakref.ref(arg)

        # Return value directly for other types, i.e. primitive types and all qd.Field-derived classes
        if raise_on_templated_floats and arg_type is float:
            raise ValueError("Floats not allowed as templated types.")
        return arg
    if annotation_type is buffer_view_type.BufferViewType:
        if not isinstance(arg, BufferViewInstance):
            raise QuadrantsRuntimeTypeError(f"Argument {arg_name} expects a BufferView, got {type(arg).__name__}")
        inner = arg.get_ndarray()
        assert isinstance(inner, Ndarray)
        assert inner.shape is not None
        if __debug__ and __builtins__["__debug__"] and annotation.dtype is not None:
            annotation.ndarray_type.check_matched(inner.get_type(), arg_name)
        type_id = id(inner.element_type)
        element_type = type_id if type_id in primitive_types.type_ids else inner.element_type
        return element_type, len(inner.shape), False, annotation.ndarray_type.boundary
    if annotation_type is ndarray_type.NdarrayType:
        if isinstance(arg, Ndarray):
            # Allow deferring '__debug__' evaluation at runtime
            if __debug__ and __builtins__["__debug__"]:
                annotation.check_matched(arg.get_type(), arg_name)
            assert arg.shape is not None
            needs_grad = annotation.needs_grad
            if needs_grad is None:
                needs_grad = arg.grad is not None
            # Convert singleton primitive dtype to int. This will dramatically speed up hashing later on.
            type_id = id(arg.element_type)
            element_type = type_id if type_id in primitive_types.type_ids else arg.element_type
            # Optional tensor layout (None for legacy / identity).
            #
            # PERF-CRITICAL: arg._qd_layout uses direct attribute access (not getattr) because Ndarray has a class-level
            # _qd_layout=None default. This avoids getattr(..., default) overhead on every kernel arg. Do not replace
            # with getattr().
            return element_type, len(arg.shape), needs_grad, annotation.boundary, arg._qd_layout
        if isinstance(arg, AnyArray):
            ty = arg.get_type()
            if __debug__ and __builtins__["__debug__"]:
                annotation.check_matched(ty, arg_name)
            assert arg.shape is not None
            return ty.element_type, len(arg.shape), ty.needs_grad, annotation.boundary, arg._qd_layout
        # external arrays
        shape = getattr(arg, "shape", None)
        if shape is None:
            raise QuadrantsRuntimeTypeError(f"Invalid type for argument {arg_name}, got {arg}")
        shape = tuple(shape)
        element_shape: tuple[int, ...] = ()
        dtype = to_quadrants_type(arg.dtype)
        if isinstance(annotation.dtype, MatrixType):
            if annotation.ndim is not None:
                if len(shape) != annotation.dtype.ndim + annotation.ndim:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has ndim={annotation.ndim} "
                        f"element_dim={annotation.dtype.ndim}, array with {len(shape)} dimensions is provided"
                    )
            else:
                if len(shape) < annotation.dtype.ndim:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required element_dim={annotation.dtype.ndim}, "
                        f"array with {len(shape)} dimensions is provided"
                    )
            element_shape = shape[-annotation.dtype.ndim :]
            anno_element_shape = annotation.dtype.get_shape()
            if None not in anno_element_shape and element_shape != anno_element_shape:
                raise ValueError(
                    f"Invalid value for argument {arg_name} - required element_shape={anno_element_shape}, "
                    f"array with element shape of {element_shape} is provided"
                )
        elif annotation.dtype is not None:
            # User specified scalar dtype
            if annotation.dtype != dtype:
                raise ValueError(
                    f"Invalid value for argument {arg_name} - required array has dtype={annotation.dtype.to_string()}, "
                    f"array with dtype={dtype.to_string()} is provided"
                )

            if annotation.ndim is not None and len(shape) != annotation.ndim:
                raise ValueError(
                    f"Invalid value for argument {arg_name} - required array has ndim={annotation.ndim}, "
                    f"array with {len(shape)} dimensions is provided"
                )
        needs_grad = getattr(arg, "requires_grad", False) if annotation.needs_grad is None else annotation.needs_grad
        if element_shape:
            element_type = _qd_core.get_type_factory_instance().get_tensor_type(element_shape, dtype)
        else:
            element_type = arg.dtype
        return element_type, len(shape) - len(element_shape), needs_grad, annotation.boundary, None
    # Inlining `dataclasses.is_dataclass` and `dataclasess.fields`, which are very slow due to extra runtime checks
    annotation_fields = getattr(annotation, _FIELDS, None)
    if annotation_fields is not None:
        # Some dataclasses may be declared as "frozen", which means that changing pointers of fields is not allowed.
        # This property is sufficient to guarantee that its quadrants "key" will never change and therefore can be stored
        # as a static attribute, much like its hash which is computed once and for all during instantiation.
        # Instead of strictly requiring being frozen, we only require the dataclass to be hashable. Any frozen dataclass
        # is hashable, but a user can enforce a dataclass to be consider frozen for a user perspective without being
        # truly frozen by specifying 'unsafe_hash=True'. If a user is doing this on purpose, it makes sense to honor it.
        is_frozen = annotation.__hash__ is not None
        # Serve the ``_qd_spec_key`` cache only in the default (non-strict) mode: the cached key is
        # setting-independent, but the ``Final[float]`` guard below is not, and an early return would also skip the
        # recursion that runs it on nested frozen dataclasses. A Final-bearing subtree is never served either - the
        # write below refuses to store one, so this read falls through and re-runs validation each launch.
        if is_frozen and not raise_on_templated_floats:
            try:
                # Instance-level (not class-level): instances of one class may differ in memory layout. The reserved
                # ``_qd_`` prefix avoids collision with a user field named ``_key``. Absent under ``slots=True``.
                return arg._qd_spec_key
            except AttributeError:
                pass
        # ``Final[T]`` fields drive the key by *value*; other fields keep the recursive ``_extract_arg``. PERF: with no
        # Final fields (the common case) we take the original comprehension verbatim.
        final_names = final_field_names(annotation)
        if final_names:
            key_parts = []
            for field in annotation_fields.values():
                if field._field_type is not _FIELD:
                    continue
                field_value = getattr(arg, field.name)
                if field.name in final_names:
                    # A ``Final[float]`` drives specialisation, so honour ``raise_on_templated_floats`` like a
                    # ``qd.template()`` float. The ``isinstance`` runs only in strict mode; ``bool`` is not a ``float``.
                    if raise_on_templated_floats and (
                        type(field_value) is float or isinstance(field_value, (float, np.floating))
                    ):
                        raise ValueError(
                            f"Floats not allowed as templated types: {annotation.__name__}.{field.name} is "
                            f"``Final[float]``, so its value is baked into the compiled kernel and each distinct "
                            f"value compiles a separate kernel. Drop the ``Final`` to make it an ordinary runtime "
                            f"field, or unset ``raise_on_templated_floats``."
                        )
                    # ``live=True``: in-process key, so a subclass/enum is keyed by class identity. The offline
                    # fastcache passes ``live=False`` (see ``final_scalar_key`` / ``args_hasher``).
                    key_parts.append(final_scalar_key(field_value, live=True))
                else:
                    key_parts.append(
                        _extract_arg(
                            raise_on_templated_floats,
                            field_value,
                            field.type,
                            create_flat_name(arg_name, field.name),
                        )
                    )
            key = tuple(key_parts)
        else:
            key = tuple(
                [
                    _extract_arg(
                        raise_on_templated_floats,
                        getattr(arg, field.name),
                        field.type,
                        create_flat_name(arg_name, field.name),
                    )
                    for field in annotation_fields.values()
                    if field._field_type is _FIELD
                ]
            )
        # Store the cache only for a ``Final``-free subtree; a Final-bearing one must revalidate every launch, so we
        # never store one (making the read above fall through).
        if is_frozen and not subtree_has_final_fields(annotation):
            try:
                object.__setattr__(arg, "_qd_spec_key", key)
            except AttributeError:
                # Impossible to store _qd_spec_key at instance-level if 'slots=True'. It is recomputed each time.
                pass
        return key
    if annotation_type is sparse_matrix_builder:
        return arg.dtype
    # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
    return "#"
