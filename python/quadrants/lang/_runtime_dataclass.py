"""Resolve Python dataclass values to the field descriptions consumed by the existing flattening pipeline.

Descriptions are private dataclass-shaped types, never user objects. Keeping them immutable and interning them by
structure lets compilation, inlined calls, dispatch and launch packing share precisely the same field order/types
without modifying the user's class or its annotations.
"""

import copy
import dataclasses
from contextlib import contextmanager
from typing import Any

import numpy as np

from quadrants._tensor_wrapper import Tensor
from quadrants.lang import impl
from quadrants.lang._dataclass_util import create_flat_name
from quadrants.lang._final_dataclass_fields import final_field_names
from quadrants.lang._ndarray import Ndarray
from quadrants.lang._runtime_dataclass_guards import FrozenSchemaEntry
from quadrants.lang.any_array import AnyArray
from quadrants.lang.buffer_view import BufferView
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.field import Field
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.matrix import Matrix, MatrixType, VectorType
from quadrants.lang.struct import Struct, StructType
from quadrants.lang.util import is_data_oriented, to_quadrants_type
from quadrants.types import buffer_view_type, ndarray_type, primitive_types
from quadrants.types.annotations import RuntimeTyped, template

_schemas: dict[tuple, type] = {}
_field_plans: dict[int, tuple] = {}
_direct_final_types: dict[int, type] = {}
_matrix_types: dict[tuple, MatrixType] = {}
_struct_types: dict[tuple, StructType] = {}
_buffer_annotation = buffer_view_type.BufferViewType()
_array_annotation = ndarray_type.NdarrayType()


def _direct_final_names(source: type) -> frozenset[str]:
    """Validate direct Final declarations without following annotation-only nested dataclass types.

    Nested paths are validated later against the actual values. Ordinary annotations must not introduce phantom
    children (or reject a valid value) just because they happen to name a Final-bearing dataclass.
    """
    direct = _direct_final_types.get(id(source))
    if direct is None:
        fields = {}
        for field in dataclasses.fields(source):
            copied = copy.copy(field)
            if isinstance(field.type, type) and dataclasses.is_dataclass(field.type):
                copied.type = object
            fields[field.name] = copied
        direct = type(
            source.__name__,
            (),
            {
                "__module__": source.__module__,
                "__annotations__": dict(source.__annotations__),
                "__dataclass_fields__": fields,
                "__dataclass_params__": source.__dataclass_params__,
                "_qd_runtime_origin": source,
            },
        )
        _direct_final_types[id(source)] = direct
    return final_field_names(direct)


def is_dataclass_annotation(annotation: Any) -> bool:
    return annotation is RuntimeTyped or isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def infer_member_type(value: Any, path: str | tuple[str, ...], visiting: frozenset[int]) -> Any:
    if isinstance(path, str):
        path = (path,)
    if isinstance(value, Tensor):
        value = value._unwrap()
    if hasattr(type(value), "__dataclass_fields__"):
        return resolve_dataclass(value, path, visiting)
    if is_data_oriented(value):
        return template
    if value is None:
        return Tensor
    if isinstance(value, Field):
        return Tensor
    if isinstance(value, (Ndarray, AnyArray, np.ndarray)):
        return _array_annotation
    # Avoid importing torch on the inference path when it is not already in use.
    import sys  # pylint: disable=import-outside-toplevel

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return _array_annotation
    if isinstance(value, BufferView):
        return _buffer_annotation
    if isinstance(value, (bool, np.bool_)):
        return primitive_types.u1
    if isinstance(value, np.generic):
        return to_quadrants_type(value.dtype)
    if isinstance(value, int):
        return impl.get_runtime().default_ip
    if isinstance(value, float):
        return impl.get_runtime().default_fp
    if isinstance(value, Matrix):
        dtype = to_quadrants_type(np.asarray(value.to_numpy()).dtype)
        key = (value.n, value.m, value.ndim, dtype)
        if key not in _matrix_types:
            _matrix_types[key] = (
                VectorType(value.n, dtype) if value.ndim == 1 else MatrixType(value.n, value.m, value.ndim, dtype)
            )
        return _matrix_types[key]
    if isinstance(value, Struct):
        declared = getattr(value, "_qd_struct_type", None)
        if declared is not None:
            return declared
        members = {name: infer_member_type(child, path + (name,), visiting) for name, child in value.items}
        key = tuple((name, id(dtype)) for name, dtype in members.items())
        if key not in _struct_types:
            _struct_types[key] = StructType(**members)
        return _struct_types[key]
    raise QuadrantsRuntimeTypeError(
        f"Dataclass member {'.'.join(path)!r} has unsupported runtime type {type(value).__name__}"
    )


def resolve_dataclass(value: Any, path: str | tuple[str, ...], visiting: frozenset[int] = frozenset()) -> type:
    if isinstance(path, str):
        path = (path,)
    if not hasattr(type(value), "__dataclass_fields__"):
        raise QuadrantsRuntimeTypeError(
            f"Argument {'.'.join(path)!r} requires a Python dataclass instance, got {type(value).__name__}"
        )
    if id(value) in visiting:
        raise QuadrantsRuntimeTypeError(f"Dataclass member {'.'.join(path)!r} contains a cycle")
    visiting = visiting | {id(value)}
    source = type(value)
    runtime = impl.get_runtime()
    cache_key = (id(source), id(runtime.default_ip), id(runtime.default_fp))
    cached = getattr(value, "_qd_runtime_schemas", {}).get(cache_key)
    if cached is not None and cached._qd_runtime_origin is source:
        return cached
    # Guarded entries reuse schema metadata only. They retain _qd_runtime_stable=False so specialization and
    # launch binding still inspect current arrays and template values.
    guarded = getattr(value, "_qd_runtime_guarded_schemas", {}).get(cache_key)
    if guarded is not None and guarded.schema._qd_runtime_origin is source:
        if guarded.matches(path, visiting, infer_member_type):
            return guarded.schema
    plan = _field_plans.get(id(source))
    if plan is None:
        finals = _direct_final_names(source)
        plan = (
            source,
            source.__dataclass_params__,
            bool(finals),
            tuple((field.name, field, field.name in finals) for field in dataclasses.fields(source)),
        )
        _field_plans[id(source)] = plan
    _, params, has_finals, field_plan = plan
    stable = params.frozen and not has_finals
    member_types = []
    for name, field, is_final in field_plan:
        member_type = field.type
        if not is_final:
            child = getattr(value, name)
            member_type = infer_member_type(child, path + (name,), visiting)
            if dataclasses.is_dataclass(member_type):
                stable &= getattr(member_type, "_qd_runtime_stable")
            elif child is not None and not isinstance(child, (bool, int, float, np.generic, Ndarray, Field)):
                stable = False
        member_types.append(member_type)
    key = (id(source), stable, tuple(map(id, member_types)))
    schema = _schemas.get(key)
    if schema is None:
        # Revalidate values on every uncached launch, but allocate field metadata only for a new inferred schema.
        # The class plan retains the source and field order; only runtime member types and stability vary here.
        fields = {}
        for (name, field, _), member_type in zip(field_plan, member_types):
            resolved = copy.copy(field)
            resolved.type = member_type
            fields[name] = resolved
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
        cache = getattr(value, "_qd_runtime_schemas", None)
        if cache is None:
            cache = {}
            try:
                object.__setattr__(value, "_qd_runtime_schemas", cache)
            except AttributeError:
                pass
        cache[cache_key] = schema
    elif params.frozen:
        cache = getattr(value, "_qd_runtime_guarded_schemas", None)
        if cache is None:
            cache = {}
            try:
                object.__setattr__(value, "_qd_runtime_guarded_schemas", cache)
            except AttributeError:
                return schema
        cache[cache_key] = FrozenSchemaEntry(value, schema, field_plan, member_types)
    return schema


def resolve_arguments(arguments: list[ArgMetadata], values: tuple[Any, ...]) -> list[ArgMetadata]:
    return [
        (
            ArgMetadata(
                resolve_dataclass(value, arg.name),
                arg.name,
                arg.default,
                optional=arg.optional,
            )
            if hasattr(type(value), "__dataclass_fields__") or is_dataclass_annotation(arg.annotation)
            else arg
        )
        for arg, value in zip(arguments, values)
    ]


def prepare_runtime_arguments(kernel, values):
    """Prepare inferred metadata and dispatch-cache eligibility for one kernel launch."""
    kernel.arg_metas = resolve_arguments(kernel.orig_arguments, values)
    kernel.mapper.arguments = kernel.arg_metas
    kernel.template_slot_locations = [
        i
        for i in kernel.original_template_slot_locations
        if not hasattr(kernel.arg_metas[i].annotation, "_qd_runtime_origin")
    ]
    kernel.mapper.template_slot_locations = kernel.template_slot_locations
    # A dataclass's member types can change without changing the identity of the outer object.
    kernel.mapper._mapping_cache_disabled = (
        any(
            hasattr(meta.annotation, "_qd_runtime_origin") and not meta.annotation._qd_runtime_stable
            for meta in kernel.arg_metas
        )
        or None
    )


def runtime_template_members(schema, value, path=()):
    """Yield template-valued subobjects with paths rooted at the original kernel argument."""
    if not hasattr(schema, "_qd_runtime_origin"):
        return
    for field in dataclasses.fields(schema):
        child = getattr(value, field.name)
        child_path = (*path, field.name)
        if field.type is template:
            yield child_path, child
        elif hasattr(field.type, "_qd_runtime_origin"):
            yield from runtime_template_members(field.type, child, child_path)


def is_runtime_member_name(arguments: list[ArgMetadata], name: str) -> bool:
    """Whether a flattened AST name belongs to an inferred dataclass argument (including inlined functions)."""
    for arg in arguments:
        if not hasattr(arg.annotation, "_qd_runtime_origin"):
            continue
        for field in dataclasses.fields(arg.annotation):
            child_name = create_flat_name(arg.name, field.name)
            if name == child_name:
                return True
            if is_runtime_member_name([ArgMetadata(field.type, child_name)], name):
                return True
    return False


@contextmanager
def dataclass_call_arguments(func, args, keywords, bound: bool = False):
    """Carry a caller's inferred structures across the pre-flattening boundary of an inlined function call."""
    wrapper = func.wrapper
    original = wrapper.arg_metas
    expanded = wrapper.arg_metas_expanded
    declared = wrapper.orig_arguments
    metas = list(declared)
    offset = int(bound)
    values = {meta.name: arg.ptr for meta, arg in zip(metas[offset:], args)}
    values.update((kw.arg, kw.ptr[kw.arg]) for kw in keywords)
    for i, meta in enumerate(metas):
        if meta.name not in values:
            continue
        value = values[meta.name]
        if isinstance(value, type) and hasattr(value, "_qd_runtime_origin"):
            schema = value
        elif hasattr(type(value), "__dataclass_fields__") or is_dataclass_annotation(meta.annotation):
            schema = resolve_dataclass(value, meta.name)
        else:
            continue
        metas[i] = ArgMetadata(schema, meta.name, meta.default, optional=meta.optional)
    wrapper.arg_metas = metas
    wrapper.orig_arguments = metas
    try:
        yield
    finally:
        wrapper.arg_metas = original
        wrapper.orig_arguments = declared
        wrapper.arg_metas_expanded = expanded
