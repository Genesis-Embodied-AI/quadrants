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
import weakref
from dataclasses import _FIELD, _FIELDS
from typing import Any, Union

from quadrants import _tensor_wrapper
from quadrants._lib import core as _qd_core
from quadrants._tensor import (
    _TENSOR_T_FIELD_MARKER,
    _TENSOR_T_NDARRAY_MARKER,
)
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang._dataclass_util import create_flat_name
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


_ExprCxx = _qd_core.ExprCxx
_composite_mutable_types = {list, dict, set}
_primitive_types = {int, float, bool}


# Per-class cache: ``type(arg) -> list[tuple[str, ...]]`` of attribute paths whose values are ``Ndarray`` instances at
# first observation. Populated lazily by ``_struct_nd_paths_for`` on the first call with each new data_oriented (or
# nested dataclass) class. Empty list means "this class holds no ndarrays anywhere", in which case subsequent calls
# pay only a dict-lookup per arg. Non-empty list short-circuits the full ``vars()`` recursion and just resolves each
# cached path via ``getattr`` chains. Critical for the genesis field-backend hot path: the ``@qd.data_oriented``
# Solver is passed as ``self`` to most kernels and holds dozens of attributes, so a full per-call ``vars()`` walk
# costs >100ns per kernel and trashed FPS until this cache was added.
_struct_nd_paths_cache: dict[type, list[tuple]] = {}


def _build_struct_nd_paths(obj: Any, prefix: tuple, out: list, _seen: "set[int] | None" = None) -> None:
    # Cycle-safe walker. Genesis object graphs have cross-references (e.g. ``solver -> scene -> sim -> solver``) and
    # Pydantic-options-style children. ``_seen`` tracks ``id(obj)`` for the current traversal to avoid re-entering a
    # node we've already expanded. Cheap (one ``set`` op per frame, only allocated when we actually start recursing)
    # and bounds the walk to a finite depth regardless of the graph shape.
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
    """Return the cached attribute paths (each a tuple of attr-name strings) at which ``Ndarray`` instances are
    reachable from ``arg`` of type ``type(arg)``. First call for a class walks ``arg`` once via
    ``_build_struct_nd_paths``; subsequent calls are dict-lookups.

    Trades freshness for speed: assumes the *set* of ndarray-holding attribute paths is stable across instances of
    the same class. The genesis Solver and similar ``@qd.data_oriented`` containers satisfy this — their ndarray
    members are declared in ``__init__`` and not added later. If you need to add an ndarray attribute after the first
    kernel launch on an instance of a given class, the new attribute won't be tracked. Call ``invalidate_struct_nd_
    paths_for`` (below) or restart the program.
    """
    cls = type(arg)
    paths = _struct_nd_paths_cache.get(cls)
    if paths is None:
        paths = []
        _build_struct_nd_paths(arg, (), paths)
        _struct_nd_paths_cache[cls] = paths
    return paths


def _collect_struct_nd_descriptors(arg: Any, out: list) -> None:
    """Emit per-ndarray shape descriptors ``(joined-path, element_type, ndim, needs_grad, layout)`` for every ndarray
    reachable from ``arg``. Used by the template-mapper to refine the spec key for ``@qd.data_oriented`` args holding
    ndarrays — see the data_oriented branch in ``_extract_arg``.
    """
    # The path cache is keyed on ``type(arg)`` and assumes the *set* of ndarray-reachable attribute chains is stable
    # across instances of the same class. That holds for the typical ``@qd.data_oriented`` container, but Genesis
    # ``FEMSolver`` / ``MPMSolver`` / ``SPHSolver`` and similar can hold polymorphic children (e.g. ``self.material``
    # of a different concrete subclass) or swap a ``qd.Tensor``'s underlying impl between an ``Ndarray`` and a
    # ``MatrixField``. When the leaf at a cached path is no longer an ``Ndarray`` we silently skip it:
    # ``v.element_type`` / ``v.shape`` / ``v._qd_layout`` are Ndarray-only accessors. The per-instance ``weakref(arg)``
    # part of the spec key still ensures correct cache discrimination across instances.
    for chain in _struct_nd_paths_for(arg):
        v = arg
        for a in chain:
            v = getattr(v, a)
        if type(v) in _TENSOR_WRAPPER_TYPES:
            v = v._unwrap()
        if not isinstance(v, Ndarray):
            continue
        # ``Ndarray.shape`` can legitimately be ``None`` (uninitialised ``_physical_shape``); such an instance
        # has no meaningful spec contribution, so skip it rather than crashing on ``len(None)``.
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
        if issubclass(arg_type, (Ndarray, AnyArray)):
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
            #
            # Opt-out: ``_qd_stable_members = True`` on the class (or ``@qd.data_oriented(stable_members=True)``)
            # skips the per-call descriptor walk.
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
        # This property is sufficient to guarantee that its taichi "key" will never change and therefore can be stored
        # as a static attribute, much like its hash which is computed once and for all during instantiation.
        # Instead of strictly requiring being frozen, we only require the dataclass to be hashable. Any frozen dataclass
        # is hashable, but a user can enforce a dataclass to be consider frozen for a user perspective without being
        # truly frozen by specifying 'unsafe_hash=True'. If a user is doing this on purpose, it makes sense to honor it.
        is_frozen = annotation.__hash__ is not None
        if is_frozen:
            try:
                # Note that it is necessary to store the key at instance-level instead of class-level because because
                # multiple instances of the same class may have different memory layout (although unusual).
                # One limitation is that storing '_key' is then impossible for dataclasses enforcing 'slots=True',
                # but this not the default option and almost never used in practice because of other limitations.
                return arg._key
            except AttributeError:
                pass
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
        if is_frozen:
            try:
                object.__setattr__(arg, "_key", key)
            except AttributeError:
                # Impossible to store _key at instance-level if 'slots=True'. It will be recomputed systematically.
                pass
        return key
    if annotation_type is sparse_matrix_builder:
        return arg.dtype
    # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
    return "#"
