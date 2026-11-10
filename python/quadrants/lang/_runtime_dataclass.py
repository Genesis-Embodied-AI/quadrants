"""Resolve Python dataclass values to the field descriptions consumed by the existing flattening pipeline.

Descriptions are private dataclass-shaped types, never user objects. Keeping them immutable and interning them by
structure lets compilation, inlined calls, dispatch and launch packing share precisely the same field order/types
without modifying the user's class or its annotations.
"""

import ast
import copy
import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeAlias, cast
from weakref import ref

import numpy as np

from quadrants import _tensor_wrapper
from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES, Tensor
from quadrants.lang import impl
from quadrants.lang._deferred_errors import (
    DeferredMemberType,
    create_deferred_member_type_for_invalid_dataclass_member,
)
from quadrants.lang._final_dataclass_fields import final_field_names
from quadrants.lang._name_utils import create_flat_name
from quadrants.lang._ndarray import Ndarray
from quadrants.lang._runtime_dataclass_cache import (
    _runtime_kernels,
    invalidate_tensor_bindings,
)
from quadrants.lang._runtime_dataclass_final import (
    get_final_field_names,
    infer_final_field_annotation,
)
from quadrants.lang._runtime_dataclass_guards import FrozenSchemaEntry
from quadrants.lang._weak_identity_cache import WeakIdentityCache
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

_tensor_wrapper._dataclass_binding_invalidator = invalidate_tensor_bindings

if TYPE_CHECKING:
    from quadrants.lang._func_base import FuncBase
    from quadrants.lang._quadrants_callable import (
        BoundQuadrantsCallable,
        QuadrantsCallable,
    )
    from quadrants.lang.kernel import Kernel


class DataclassInstance(Protocol):
    """Describe the field metadata shared by Python dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


# Object identity maps to the retained object and its type-tagged Final specialization key.
FinalValueKeyCache: TypeAlias = dict[int, tuple[Any, tuple[Any, ...]]]

InferredMemberType: TypeAlias = (
    type | DataTypeCxx | ndarray_type.NdarrayType | buffer_view_type.BufferViewType | MatrixType | StructType
)

# Per-instance schemas, separated by numeric defaults and ancestor validation context; keys are held weakly.
InferenceContextKey: TypeAlias = tuple[int, int, int, str | None]
_schema_by_context_by_instance: WeakIdentityCache[dict[InferenceContextKey, type | FrozenSchemaEntry]] = (
    WeakIdentityCache()
)

# These caches retain dataclass classes, assuming applications define relatively few of them.
# Repeatedly generating new dataclass classes will grow these caches for the lifetime of the process.
# Generated schema classes, keyed by source class identity, stability, and ordered member names/type identities.
_schemas: dict[tuple[int, bool, tuple[tuple[str, int], ...]], type] = {}
# Source classes and their ordered field descriptions, reused across schema inference calls.
_field_plans: dict[int, tuple[type, tuple[tuple[str, dataclasses.Field[Any], bool], ...]]] = {}
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
    py_obj: Any,
    path: str,
    ancestor_ids: frozenset[int],
    non_frozen_ancestor_path: str | None = None,
    final_value_key_by_value_id: FinalValueKeyCache | None = None,
) -> InferredMemberType | DeferredMemberType:
    """Infer a member type from its value, or defer an unsupported-value error until the member is used."""
    if isinstance(py_obj, Tensor):
        py_obj = py_obj._unwrap()
    if hasattr(type(py_obj), "__dataclass_fields__"):
        return infer_dataclass_type(py_obj, path, ancestor_ids, non_frozen_ancestor_path, final_value_key_by_value_id)
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
        members = {name: infer_member_type(child, f"{path}.{name}", ancestor_ids) for name, child in py_obj.items}
        key = tuple((name, id(dtype)) for name, dtype in members.items())
        if key not in _struct_types:
            _struct_types[key] = StructType(**members)
        return _struct_types[key]
    return create_deferred_member_type_for_invalid_dataclass_member(
        QuadrantsRuntimeTypeError, f"unsupported runtime type {type(py_obj).__name__}", py_obj
    )


def infer_dataclass_type(
    dataclass_instance: DataclassInstance,
    path: str,
    ancestor_ids: frozenset[int] = frozenset(),
    non_frozen_ancestor_path: str | None = None,
    final_value_key_by_value_id: FinalValueKeyCache | None = None,
) -> type:
    """Build or reuse a private schema class; the caller must verify that dataclass_instance is a dataclass instance."""
    if id(dataclass_instance) in ancestor_ids:
        raise QuadrantsRuntimeTypeError(f"Dataclass member {path!r} contains a cycle")
    ancestor_ids = ancestor_ids | {id(dataclass_instance)}
    dataclass_type = type(dataclass_instance)
    runtime = impl.get_runtime()
    # The same instance can need different schemas under different numeric defaults or non-frozen ancestor paths.
    # These affect member types and Final validation, so each instance has a dictionary keyed by inference context.
    inference_context_key = (
        id(dataclass_type),
        id(runtime.default_ip),
        id(runtime.default_fp),
        non_frozen_ancestor_path,
    )
    schema_by_context = _schema_by_context_by_instance.get(dataclass_instance)
    cached_schema = schema_by_context.get(inference_context_key) if schema_by_context is not None else None
    if isinstance(cached_schema, FrozenSchemaEntry):
        if cached_schema.matches(dataclass_instance, path, ancestor_ids, infer_member_type):
            return cached_schema.schema
    elif cached_schema is not None:
        return cached_schema
    finals = get_final_field_names(dataclass_type)
    plan = _field_plans.get(id(dataclass_type))
    if plan is None:
        plan = (
            dataclass_type,
            tuple((field.name, field, field.name in finals) for field in dataclasses.fields(dataclass_type)),
        )
        _field_plans[id(dataclass_type)] = plan
    _, field_plan = plan
    params = getattr(dataclass_type, "__dataclass_params__")
    container_prevents_rebinding = params.frozen or params.unsafe_hash
    final_non_frozen_path = non_frozen_ancestor_path
    if not container_prevents_rebinding:
        final_non_frozen_path = dataclass_type.__name__
    stable = params.frozen and not finals
    if stable:
        try:
            ref(dataclass_instance)
        except TypeError:
            # Tensor rebinding must be able to find and invalidate this instance's downstream caches.
            stable = False
    member_types = []
    has_deferred = False
    for name, field, is_final in field_plan:
        if is_final:
            member_type = infer_final_field_annotation(
                getattr(dataclass_instance, name), final_non_frozen_path, name, final_value_key_by_value_id
            )
        else:
            child = getattr(dataclass_instance, name)
            child_non_frozen_ancestor_path = None
            if not container_prevents_rebinding:
                child_non_frozen_ancestor_path = f"{dataclass_type.__name__}.{name}"
            elif non_frozen_ancestor_path is not None:
                child_non_frozen_ancestor_path = f"{non_frozen_ancestor_path}.{name}"
            member_type = infer_member_type(
                child, f"{path}.{name}", ancestor_ids, child_non_frozen_ancestor_path, final_value_key_by_value_id
            )
            if dataclasses.is_dataclass(member_type):
                stable &= getattr(member_type, "_qd_runtime_stable")
            elif (
                child is not None
                and type(child) not in _TENSOR_WRAPPER_TYPES
                and not isinstance(child, (bool, int, float, np.generic, Ndarray, Field))
            ):
                stable = False
        has_deferred |= isinstance(member_type, DeferredMemberType) or getattr(
            member_type, "_qd_runtime_has_deferred", False
        )
        member_types.append(member_type)
    # Deferred descriptions depend on debug mode. Re-infer them each launch instead of caching them by instance.
    stable &= not has_deferred
    key = (
        id(dataclass_type),
        stable,
        tuple((name, id(member_type)) for (name, _, _), member_type in zip(field_plan, member_types)),
    )
    schema = _schemas.get(key)
    if schema is None:
        # Revalidate py_objs on every uncached launch, but allocate field metadata only for a new inferred schema.
        # The class plan retains the dataclass_type and field order; only runtime member types and stability vary here.
        fields = {}
        for (name, field, _), member_type in zip(field_plan, member_types):
            resolved = copy.copy(field)
            resolved.type = member_type
            fields[name] = resolved
        schema = type(
            dataclass_type.__name__,
            (),
            {
                "__module__": dataclass_type.__module__,
                "__qualname__": dataclass_type.__qualname__,
                "__dataclass_fields__": fields,
                "__dataclass_params__": params,
                "__hash__": dataclass_type.__hash__,
                "_qd_runtime_origin": dataclass_type,
                "_qd_runtime_stable": stable,
                "_qd_runtime_has_deferred": has_deferred,
                # Cache eligibility is not part of the compiled interface: a Tensor wrapper and its bare ndarray
                # must share a specialization, even though the wrapper can change its underlying implementation.
                "_qd_runtime_key": (
                    id(dataclass_type),
                    tuple(
                        (name, getattr(field.type, "_qd_runtime_key", id(field.type))) for name, field in fields.items()
                    ),
                ),
            },
        )
        # Validate Final-bearing paths against the inferred nesting, including children annotated as Any/object.
        final_field_names(schema)
        _schemas[key] = schema
    # Guard reuse must not bypass deferred diagnostics or validation of Final-bearing paths.
    if stable or (params.frozen and not has_deferred and not finals and non_frozen_ancestor_path is None):
        if schema_by_context is None:
            schema_by_context = {}
            if not _schema_by_context_by_instance.set(dataclass_instance, schema_by_context):
                return schema
        schema_by_context[inference_context_key] = (
            schema if stable else FrozenSchemaEntry(dataclass_instance, schema, field_plan, member_types)
        )
    return schema


def infer_kernel_dataclass_arg_types(
    arg_metas: list[ArgMetadata],
    py_objs: tuple[Any, ...],
    final_value_key_by_value_id: FinalValueKeyCache | None = None,
) -> list[ArgMetadata]:
    """Replace kernel dataclass parameter annotations with schemas inferred from their arriving values."""
    resolved_arg_metas = []
    for arg_meta, py_obj in zip(arg_metas, py_objs):
        if hasattr(type(py_obj), "__dataclass_fields__"):
            arg_meta = ArgMetadata(
                infer_dataclass_type(py_obj, arg_meta.name, final_value_key_by_value_id=final_value_key_by_value_id),
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


def prepare_kernel_dataclass_args_for_mapper(
    kernel: "Kernel", py_objs: tuple[Any, ...], args_hash: tuple[int, ...] | None = None
) -> FinalValueKeyCache | None:
    """Update dataclass argument types, template slots, cache eligibility, and deferred-member status for a launch.

    Reuse metadata when argument identities and numeric defaults match a live mapper cache entry.
    Return Final keys already validated during this call so specialization can reuse them.
    """
    defaults = (id(kernel.runtime.default_ip), id(kernel.runtime.default_fp))
    if (
        args_hash is not None
        and kernel._runtime_metadata_args_hash == args_hash
        and kernel._runtime_metadata_defaults == defaults
        and kernel.mapper._mapping_cache_tracker.get(args_hash)
    ):
        return None
    _runtime_kernels.add(kernel)
    if kernel._runtime_metadata_defaults is not None and kernel._runtime_metadata_defaults != defaults:
        kernel.mapper._mapping_cache.clear()
        kernel.mapper._mapping_cache_tracker.clear()
    # Per-call cache: id(value) -> (value, specialization key); retain the value to verify its identity.
    final_value_key_by_value_id: FinalValueKeyCache = {}
    kernel.arg_metas = infer_kernel_dataclass_arg_types(kernel.orig_arguments, py_objs, final_value_key_by_value_id)
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
    kernel._runtime_metadata_args_hash = args_hash if not unstable and not has_deferred else None
    kernel._runtime_metadata_defaults = defaults

    return final_value_key_by_value_id


def runtime_template_members(
    schema: type, dataclass_instance: DataclassInstance, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Yield template-valued subobjects with paths rooted at the original kernel argument."""
    if not hasattr(schema, "_qd_runtime_origin"):
        return
    for field in dataclasses.fields(schema):
        child = getattr(dataclass_instance, field.name)
        child_path = (*path, field.name)
        if field.type is Template:
            yield child_path, child
        elif hasattr(field.type, "_qd_runtime_origin"):
            yield from runtime_template_members(cast(type, field.type), child, child_path)


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
def dataclass_call_arguments(
    func: "QuadrantsCallable | BoundQuadrantsCallable",
    args: Sequence[ast.expr],
    keywords: Sequence[ast.keyword],
    bound: bool = False,
) -> Iterator[None]:
    """Carry a caller's inferred structures across the pre-flattening boundary of an inlined function call."""
    wrapper = cast("FuncBase", func.wrapper)
    original_arg_metas = wrapper.arg_metas
    expanded_arg_metas = wrapper.arg_metas_expanded
    declared_arg_metas = wrapper.orig_arguments
    arg_metas = list(declared_arg_metas)
    offset = int(bound)
    py_objs = {meta.name: getattr(arg, "ptr") for meta, arg in zip(arg_metas[offset:], args)}
    py_objs.update((kw.arg, getattr(kw, "ptr")[kw.arg]) for kw in keywords)
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
            schema = infer_dataclass_type(cast(DataclassInstance, py_obj), meta.name)
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
