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
from quadrants.lang._final_dataclass_fields import final_field_names
from quadrants.lang._ndarray import Ndarray
from quadrants.lang.any_array import AnyArray
from quadrants.lang.buffer_view import BufferView
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.field import Field
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.matrix import Matrix, MatrixType
from quadrants.lang.util import to_quadrants_type
from quadrants.types import buffer_view_type, primitive_types
from quadrants.types.annotations import RuntimeTyped

_schemas: dict[tuple, type] = {}
_direct_final_types: dict[int, type] = {}
_matrix_types: dict[tuple, MatrixType] = {}
_buffer_annotation = buffer_view_type.BufferViewType()


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


def infer_member_type(value: Any, path: str, visiting: frozenset[int]) -> Any:
    if isinstance(value, Tensor):
        value = value._unwrap()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return resolve_dataclass(value, path, visiting)
    if isinstance(value, (Ndarray, AnyArray, Field, np.ndarray)):
        return Tensor
    # Avoid importing torch on the inference path when it is not already in use.
    import sys  # pylint: disable=import-outside-toplevel

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return Tensor
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
        dtype = to_quadrants_type(value.to_numpy().dtype)
        key = (value.n, value.m, value.ndim, dtype)
        if key not in _matrix_types:
            _matrix_types[key] = MatrixType(value.n, value.m, value.ndim, dtype)
        return _matrix_types[key]
    raise QuadrantsRuntimeTypeError(f"Dataclass member {path!r} has unsupported runtime type {type(value).__name__}")


def resolve_dataclass(
    value: Any, path: str, visiting: frozenset[int] = frozenset(), declared_type: type | None = None
) -> type:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise QuadrantsRuntimeTypeError(
            f"Argument {path!r} requires a Python dataclass instance, got {type(value).__name__}"
        )
    if id(value) in visiting:
        raise QuadrantsRuntimeTypeError(f"Dataclass member {path!r} contains a cycle")
    visiting = visiting | {id(value)}
    source = declared_type if declared_type is not None else type(value)
    if not isinstance(value, source):
        raise QuadrantsRuntimeTypeError(f"Argument {path!r} requires {source.__name__}, got {type(value).__name__}")
    runtime = impl.get_runtime()
    cache_key = (id(source), id(runtime.default_ip), id(runtime.default_fp))
    cached = getattr(value, "_qd_runtime_schemas", {}).get(cache_key)
    if cached is not None and cached._qd_runtime_origin is source:
        return cached
    finals = _direct_final_names(source)
    stable = source.__dataclass_params__.frozen and not finals
    fields = {}
    for field in dataclasses.fields(source):
        resolved = copy.copy(field)
        if field.name not in finals:
            child = getattr(value, field.name)
            resolved.type = infer_member_type(child, f"{path}.{field.name}", visiting)
            if dataclasses.is_dataclass(resolved.type):
                stable &= resolved.type._qd_runtime_stable
            elif not isinstance(child, (bool, int, float, np.generic, Ndarray, Field)):
                stable = False
        fields[field.name] = resolved
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
                "__dataclass_params__": source.__dataclass_params__,
                "__hash__": source.__hash__,
                "_qd_runtime_origin": source,
                "_qd_runtime_stable": stable,
            },
        )
        # Validate Final-bearing paths against the inferred nesting, including children annotated as Any/object.
        final_field_names(schema)
        _schemas[key] = schema
    if stable:
        try:
            cache = value._qd_runtime_schemas
        except AttributeError:
            cache = {}
            try:
                object.__setattr__(value, "_qd_runtime_schemas", cache)
            except AttributeError:
                pass
        cache[cache_key] = schema
    return schema


def resolve_arguments(arguments: list[ArgMetadata], values: tuple[Any, ...]) -> list[ArgMetadata]:
    return [
        (
            ArgMetadata(
                resolve_dataclass(
                    value, arg.name, declared_type=None if arg.annotation is RuntimeTyped else arg.annotation
                ),
                arg.name,
                arg.default,
                optional=arg.optional,
            )
            if is_dataclass_annotation(arg.annotation)
            else arg
        )
        for arg, value in zip(arguments, values)
    ]


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
        if not is_dataclass_annotation(meta.annotation) or meta.name not in values:
            continue
        value = values[meta.name]
        if isinstance(value, type) and hasattr(value, "_qd_runtime_origin"):
            schema = value
        else:
            schema = resolve_dataclass(value, meta.name)
        metas[i] = ArgMetadata(schema, meta.name, meta.default, optional=meta.optional)
    wrapper.arg_metas = metas
    wrapper.orig_arguments = metas
    try:
        yield
    finally:
        wrapper.arg_metas = original
        wrapper.orig_arguments = declared
        wrapper.arg_metas_expanded = expanded
