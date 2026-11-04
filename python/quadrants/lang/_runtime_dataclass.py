"""Resolve Python dataclass values to the field descriptions consumed by the existing flattening pipeline.

Descriptions are private dataclass-shaped types, never user objects. Keeping them immutable and interning them by
structure lets compilation, inlined calls, dispatch and launch packing share precisely the same field order/types
without modifying the user's class or its annotations.
"""

import copy
import dataclasses
from contextlib import contextmanager
from typing import Any, TypeAlias

import numpy as np

from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._tensor_wrapper import Tensor
from quadrants.lang import impl
from quadrants.lang._dataclass_util import create_flat_name
from quadrants.lang._deferred_dataclass import DeferredMemberType, deferred_member
from quadrants.lang._final_dataclass_fields import final_field_names
from quadrants.lang._ndarray import Ndarray
from quadrants.lang._runtime_dataclass_final import (
    runtime_final_plan,
    runtime_final_type,
)
from quadrants.lang.any_array import AnyArray
from quadrants.lang.buffer_view import BufferView
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.field import Field
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.matrix import Matrix, MatrixType, VectorType
from quadrants.lang.struct import Struct, StructType
from quadrants.lang.util import is_data_oriented, to_quadrants_type
from quadrants.types import buffer_view_type, ndarray_type, primitive_types
from quadrants.types.annotations import RuntimeTyped, Template

InferredMemberType: TypeAlias = (
    type | DataTypeCxx | ndarray_type.NdarrayType | buffer_view_type.BufferViewType | MatrixType | StructType
)

# Generated schema classes, keyed by source class identity, stability, and ordered member names/type identities.
_schemas: dict[tuple[int, bool, tuple[tuple[str, int], ...]], type] = {}
# Matrix/vector type descriptions, keyed by rows, columns, dimensionality, and scalar dtype.
_matrix_types: dict[tuple[int, int, int, DataTypeCxx], MatrixType] = {}
# Inferred struct type descriptions, keyed by ordered member names and type identities.
_struct_types: dict[tuple[tuple[str, int], ...], StructType] = {}
# Shared buffer-view type description; contains no instance data.
_buffer_annotation = buffer_view_type.BufferViewType()
# Shared array type description; actual dtype and rank are checked during specialization.
_array_annotation = ndarray_type.NdarrayType()


def is_dataclass_annotation(annotation: Any) -> bool:
    """Recognize RuntimeTyped and concrete dataclass annotations without inferring member types from them."""
    return annotation is RuntimeTyped or isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def infer_member_type(
    py_obj: Any, path: str, visiting: frozenset[int], final_parent: str | None = None, validated_keys=None
) -> InferredMemberType | DeferredMemberType:
    """Infer a member type from its value, or defer an unsupported-value error until the member is used."""
    if isinstance(py_obj, Tensor):
        py_obj = py_obj._unwrap()
    if hasattr(type(py_obj), "__dataclass_fields__"):
        return infer_dataclass_type(py_obj, path, visiting, final_parent, validated_keys)
    if is_data_oriented(py_obj):
        return Template
    if py_obj is None:
        return Tensor
    if isinstance(py_obj, Field):
        return Tensor
    if isinstance(py_obj, (Ndarray, AnyArray, np.ndarray)):
        return _array_annotation
    # Avoid importing torch on the inference path when it is not already in use.
    import sys  # pylint: disable=import-outside-toplevel

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(py_obj, torch.Tensor):
        return _array_annotation
    if isinstance(py_obj, BufferView):
        return _buffer_annotation
    if isinstance(py_obj, (bool, np.bool_)):
        return primitive_types.u1
    if isinstance(py_obj, np.generic):
        return to_quadrants_type(py_obj.dtype)
    if isinstance(py_obj, int):
        return impl.get_runtime().default_ip
    if isinstance(py_obj, float):
        return impl.get_runtime().default_fp
    if isinstance(py_obj, Matrix):
        dtype = to_quadrants_type(np.asarray(py_obj.to_numpy()).dtype)
        key = (py_obj.n, py_obj.m, py_obj.ndim, dtype)
        if key not in _matrix_types:
            _matrix_types[key] = (
                VectorType(py_obj.n, dtype) if py_obj.ndim == 1 else MatrixType(py_obj.n, py_obj.m, py_obj.ndim, dtype)
            )
        return _matrix_types[key]
    if isinstance(py_obj, Struct):
        declared = getattr(py_obj, "_qd_struct_type", None)
        if declared is not None:
            return declared
        members = {name: infer_member_type(child, f"{path}.{name}", visiting) for name, child in py_obj.items}
        key = tuple((name, id(dtype)) for name, dtype in members.items())
        if key not in _struct_types:
            _struct_types[key] = StructType(**members)
        return _struct_types[key]
    return deferred_member(QuadrantsRuntimeTypeError, f"unsupported runtime type {type(py_obj).__name__}")


def infer_dataclass_type(
    py_obj: Any, path: str, visiting: frozenset[int] = frozenset(), final_parent: str | None = None, validated_keys=None
) -> type:
    """Build or reuse a private schema class; the caller must verify that py_obj is a dataclass instance."""
    if id(py_obj) in visiting:
        raise QuadrantsRuntimeTypeError(f"Dataclass member {path!r} contains a cycle")
    visiting = visiting | {id(py_obj)}
    source = type(py_obj)
    runtime = impl.get_runtime()
    cache_key = (id(source), id(runtime.default_ip), id(runtime.default_fp), final_parent)
    cached = getattr(py_obj, "_qd_runtime_schemas", {}).get(cache_key)
    if cached is not None and cached._qd_runtime_origin is source:
        return cached
    finals, final_errors = runtime_final_plan(source)
    params = getattr(source, "__dataclass_params__")
    stable = params.frozen and not finals and not final_errors
    fields = {}
    has_deferred = False
    for field in dataclasses.fields(source):
        resolved = copy.copy(field)
        if field.name in final_errors:
            resolved.type = final_errors[field.name]
        elif field.name in finals:
            resolved.type = runtime_final_type(
                field.type, getattr(py_obj, field.name), final_parent, field.name, validated_keys
            )
        else:
            child = getattr(py_obj, field.name)
            child_final_parent = None
            if not (params.frozen or params.unsafe_hash):
                child_final_parent = f"{source.__name__}.{field.name}"
            elif final_parent is not None:
                child_final_parent = f"{final_parent}.{field.name}"
            resolved.type = infer_member_type(
                child, f"{path}.{field.name}", visiting, child_final_parent, validated_keys
            )
            if dataclasses.is_dataclass(resolved.type):
                stable &= getattr(resolved.type, "_qd_runtime_stable")
            elif child is not None and not isinstance(child, (bool, int, float, np.generic, Ndarray, Field)):
                stable = False
        has_deferred |= isinstance(resolved.type, DeferredMemberType) or getattr(
            resolved.type, "_qd_runtime_has_deferred", False
        )
        fields[field.name] = resolved
    stable &= not has_deferred
    key = (id(source), stable, tuple((name, id(field.type)) for name, field in fields.items()))
    schema = _schemas.get(key)
    if schema is None:
        schema = type(
            source.__name__,
            (),
            {
                "__module__": source.__module__,
                "__qualname__": source.__qualname__,
                "__dataclass_fields__": fields,
                "__dataclass_params__": params,
                "__hash__": source.__hash__,
                "_qd_runtime_origin": source,
                "_qd_runtime_stable": stable,
                "_qd_runtime_has_deferred": has_deferred,
                # Cache eligibility is not part of the compiled interface: a Tensor wrapper and its bare ndarray
                # must share a specialization, even though the wrapper can change its underlying implementation.
                "_qd_runtime_key": (
                    id(source),
                    tuple(
                        (name, getattr(field.type, "_qd_runtime_key", id(field.type))) for name, field in fields.items()
                    ),
                ),
            },
        )
        # Validate Final-bearing paths against the inferred nesting, including children annotated as Any/object.
        final_field_names(schema)
        _schemas[key] = schema
    if stable:
        cache = getattr(py_obj, "_qd_runtime_schemas", None)
        if cache is None:
            cache = {}
            try:
                object.__setattr__(py_obj, "_qd_runtime_schemas", cache)
            except AttributeError:
                pass
        cache[cache_key] = schema
    return schema


def infer_kernel_dataclass_arg_types(
    arg_metas: list[ArgMetadata], py_objs: tuple[Any, ...], validated_keys=None
) -> list[ArgMetadata]:
    """Replace kernel dataclass parameter annotations with schemas inferred from their arriving values."""
    resolved_arg_metas = []
    for arg_meta, py_obj in zip(arg_metas, py_objs):
        if hasattr(type(py_obj), "__dataclass_fields__"):
            arg_meta = ArgMetadata(
                infer_dataclass_type(py_obj, arg_meta.name, validated_keys=validated_keys),
                arg_meta.name,
                arg_meta.default,
                optional=arg_meta.optional,
            )
        elif is_dataclass_annotation(arg_meta.annotation):
            raise QuadrantsRuntimeTypeError(
                f"Argument {arg_meta.name!r} requires a Python dataclass instance, got {type(py_obj).__name__}"
            )
        resolved_arg_metas.append(arg_meta)
    return resolved_arg_metas


def prepare_kernel_dataclass_args_for_mapper(kernel, py_objs):
    """Update dataclass argument types, template slots, cache eligibility, and deferred-member status for a launch.

    Return Final keys already validated during this call so specialization can reuse them.
    """
    validated_keys = {}
    kernel.arg_metas = infer_kernel_dataclass_arg_types(kernel.orig_arguments, py_objs, validated_keys)
    kernel.mapper.arguments = kernel.arg_metas
    has_deferred = False
    unstable = False
    for meta in kernel.arg_metas:
        schema = meta.annotation
        if hasattr(schema, "_qd_runtime_origin"):
            has_deferred |= schema._qd_runtime_has_deferred
            unstable |= not schema._qd_runtime_stable
    kernel._runtime_has_deferred = has_deferred
    kernel.template_slot_locations = [
        i
        for i in kernel.original_template_slot_locations
        if not hasattr(kernel.arg_metas[i].annotation, "_qd_runtime_origin")
    ]
    kernel.mapper.template_slot_locations = kernel.template_slot_locations
    # A dataclass's member types can change without changing the identity of the outer object.
    kernel.mapper._mapping_cache_disabled = unstable or None

    return validated_keys


def runtime_template_members(schema, py_obj, path=()):
    """Yield template-valued subobjects with paths rooted at the original kernel argument."""
    if not hasattr(schema, "_qd_runtime_origin"):
        return
    for field in dataclasses.fields(schema):
        child = getattr(py_obj, field.name)
        child_path = (*path, field.name)
        if field.type is Template:
            yield child_path, child
        elif hasattr(field.type, "_qd_runtime_origin"):
            yield from runtime_template_members(field.type, child, child_path)


def is_runtime_member_name(arg_metas: list[ArgMetadata], name: str) -> bool:
    """Whether a flattened AST name belongs to an inferred dataclass argument (including inlined functions)."""
    for arg_meta in arg_metas:
        if not hasattr(arg_meta.annotation, "_qd_runtime_origin"):
            continue
        for field in dataclasses.fields(arg_meta.annotation):
            child_name = create_flat_name(arg_meta.name, field.name)
            if name == child_name:
                return True
            if is_runtime_member_name([ArgMetadata(field.type, child_name)], name):
                return True
    return False


@contextmanager
def dataclass_call_arguments(func, args, keywords, bound: bool = False):
    """Carry a caller's inferred structures across the pre-flattening boundary of an inlined function call."""
    wrapper = func.wrapper
    original_arg_metas = wrapper.arg_metas
    expanded_arg_metas = wrapper.arg_metas_expanded
    declared_arg_metas = wrapper.orig_arguments
    arg_metas = list(declared_arg_metas)
    offset = int(bound)
    py_objs = {meta.name: arg.ptr for meta, arg in zip(arg_metas[offset:], args)}
    py_objs.update((kw.arg, kw.ptr[kw.arg]) for kw in keywords)
    for i, meta in enumerate(arg_metas):
        if meta.name not in py_objs:
            if hasattr(type(meta.default), "__dataclass_fields__"):
                raise QuadrantsRuntimeTypeError(
                    f"Omitted dataclass defaults are not supported in @qd.func. Pass {meta.name!r} explicitly."
                )
            continue
        py_obj = py_objs[meta.name]
        if isinstance(py_obj, type) and hasattr(py_obj, "_qd_runtime_origin"):
            schema = py_obj
        elif hasattr(type(py_obj), "__dataclass_fields__"):
            schema = infer_dataclass_type(py_obj, meta.name)
        elif is_dataclass_annotation(meta.annotation):
            raise QuadrantsRuntimeTypeError(
                f"Argument {meta.name!r} requires a Python dataclass instance, got {type(py_obj).__name__}"
            )
        else:
            continue
        arg_metas[i] = ArgMetadata(schema, meta.name, meta.default, optional=meta.optional)
    wrapper.arg_metas = arg_metas
    wrapper.orig_arguments = arg_metas
    try:
        yield
    finally:
        wrapper.arg_metas = original_arg_metas
        wrapper.orig_arguments = declared_arg_metas
        wrapper.arg_metas_expanded = expanded_arg_metas
