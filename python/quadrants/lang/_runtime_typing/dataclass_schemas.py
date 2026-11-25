"""Resolve Python dataclass values to the field descriptions consumed by the existing flattening pipeline.

Descriptions are private dataclass-shaped types, never user objects. Keeping them immutable and interning them by
structure lets compilation, inlined calls, dispatch and launch packing share precisely the same field order/types
without modifying the user's class or its annotations.
"""

import dataclasses
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import _FIELD
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from quadrants import _warnings
from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants._tensor_wrapper import Tensor
from quadrants.lang import impl
from quadrants.lang._final.final_annotation import dataclass_field_is_final
from quadrants.lang._final.final_dataclass_fields import infer_final_field_annotation
from quadrants.lang._ndarray import Ndarray
from quadrants.lang._runtime_typing.deferred_errors import (
    create_deferred_member_type_for_invalid_dataclass_member,
)
from quadrants.lang._runtime_typing.runtime_dataclass_guards import (
    SchemaNeedingMemberChecks,
)
from quadrants.lang._runtime_typing.types import (
    DataclassInstance,
    DeferredMemberType,
    FinalValueKeyCache,
    InferenceContextKey,
    InferredMemberType,
    RuntimeDataclassSchema,
    create_dataclass_schema,
    is_frozen_dataclass,
    is_runtime_dataclass_schema,
)
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
from quadrants.types.annotations import RuntimeTyped

if TYPE_CHECKING:
    from typing import _SpecialForm

    from quadrants.lang._func_base import FuncBase
    from quadrants.lang._quadrants_callable import (
        BoundQuadrantsCallable,
        QuadrantsCallable,
    )
    from quadrants.lang.kernel import Kernel


# Schema lookup uses three cache layers:
# - Dataclass instance, then compiler settings and ancestor path -> schema or SchemaNeedingMemberChecks.
# - Dataclass type -> field names and Final flags.
# - Dataclass type and inferred member types -> schema.
#
# Arriving dataclass instance
# |
# +-- Already in the ancestor chain?
# |   +-- Yes -> Raise cycle error
# |   `-- No
# |
# +-- Look up instance and context
# |   +-- Schema -> Return it
# |   +-- SchemaNeedingMemberChecks
# |   |   +-- Checks pass -> Return its schema
# |   |   `-- Checks fail -> Continue below
# |   `-- Cache miss -> Continue below
# |
# +-- Look up field names and Final flags by dataclass type
# |   +-- Hit -> Use cached names and flags
# |   `-- Miss -> Collect names and flags; cache them
# |
# +-- Read current member values and infer their types
# |
# +-- Look up schema by dataclass type and member types
# |   +-- Hit -> Use cached schema
# |   `-- Miss -> Create schema; cache it
# |
# +-- Decide what to store for this instance and context
# |   +-- No checks needed -> Store schema
# |   +-- Checks needed and caching allowed
# |   |   `-- Store SchemaNeedingMemberChecks
# |   `-- Caching not allowed -> Store nothing
# |
# `-- Return schema
#
# Instance-cache writes require weak-reference support; otherwise return the schema without storing it.
#
# Per-instance schemas, separated by compiler settings and ancestor validation context; keys are held weakly.
_schema_by_dataclass_instance_then_compiler_settings_and_ancestor_path: WeakIdentityCache[
    dict[InferenceContextKey, type[RuntimeDataclassSchema] | SchemaNeedingMemberChecks]
] = WeakIdentityCache()

# These caches retain dataclass classes, assuming applications define relatively few of them.
# Repeatedly generating new dataclass classes will grow these caches for the lifetime of the process.
# Generated schema classes, keyed by source class identity, stability, and ordered member names/type identities.
_schema_by_dataclass_type_and_member_types: dict[
    tuple[int, bool, tuple[tuple[str, int], ...]], type[RuntimeDataclassSchema]
] = {}
# Source classes, ordered field names and Final flags, and whether any fields are Final.
_field_names_and_finality_by_dataclass_type: dict[int, tuple[type, tuple[tuple[str, bool], ...], bool]] = {}
# Matrix/vector type descriptions, keyed by rows, columns, dimensionality, and scalar dtype.
_matrix_types: dict[tuple[int, int, int, DataTypeCxx], MatrixType] = {}
# Inferred struct type descriptions, keyed by ordered member names and type identities.
_struct_types: dict[tuple[tuple[str, int], ...], StructType] = {}
# Shared buffer-view type description; contains no instance data.
_buffer_annotation = buffer_view_type.BufferViewType()
# Shared array type description; actual dtype and rank are checked during specialization.
_array_annotation = ndarray_type.NdarrayType()


def _cached_get_field_names_and_finality(
    dataclass_type: type[DataclassInstance],
) -> tuple[tuple[tuple[str, bool], ...], bool]:
    """Return cached (field name, is_final) pairs and whether any field is Final."""
    _, field_names_and_final, has_final_fields = _field_names_and_finality_by_dataclass_type.get(
        id(dataclass_type), (None, None, False)
    )
    if field_names_and_final is None:
        field_names_and_final = tuple(
            (field.name, dataclass_field_is_final(dataclass_type, field.name, field.type))
            for field in dataclasses.fields(dataclass_type)
        )
        has_final_fields = any(is_final for _, is_final in field_names_and_final)
        _field_names_and_finality_by_dataclass_type[id(dataclass_type)] = (
            dataclass_type,
            field_names_and_final,
            has_final_fields,
        )
    return field_names_and_final, has_final_fields


def is_dataclass_annotation(annotation: Any) -> bool:
    """Recognize RuntimeTyped and concrete dataclass annotations without inferring member types from them."""
    return annotation is RuntimeTyped or isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def infer_member_type(
    py_obj: Any,
    path: str,
    ancestor_ids: frozenset[int],
    non_frozen_ancestor_path: str | None = None,
    final_value_key_by_value_id: FinalValueKeyCache | None = None,
) -> tuple[InferredMemberType | DeferredMemberType, bool]:
    """Return (member_type, needs_member_check) inferred from the arriving value.

    member_type is an inferred type or a DeferredMemberType describing an unsupported value. needs_member_check
    indicates whether the member must be rechecked before reusing its containing dataclass's schema.
    """
    # FIXME: Generalize runtime type inference to all kernel arguments, not just dataclass members.
    if isinstance(py_obj, Tensor):
        member_type, _ = infer_member_type(
            py_obj._unwrap(), path, ancestor_ids, non_frozen_ancestor_path, final_value_key_by_value_id
        )
        if is_runtime_dataclass_schema(member_type):
            return member_type, member_type._qd_needs_member_check

        # Tensor subclasses must also keep their backing type invariant.
        return member_type, False

    if hasattr(type(py_obj), "__dataclass_fields__"):
        schema = cached_infer_dataclass_schema(
            py_obj, path, ancestor_ids, non_frozen_ancestor_path, final_value_key_by_value_id
        )
        return schema, schema._qd_needs_member_check

    if is_data_oriented(py_obj):
        return (
            create_deferred_member_type_for_invalid_dataclass_member(
                QuadrantsRuntimeTypeError,
                "Data-oriented objects inside Python dataclass fields are unsupported. "
                "Store the dataclass on the data-oriented object instead.",
                py_obj,
            ),
            False,
        )

    if py_obj is None:
        return Tensor, False

    if isinstance(py_obj, Field):
        return Tensor, False

    if isinstance(py_obj, (Ndarray, np.ndarray, AnyArray)):
        return _array_annotation, False

    # Avoid importing torch on the inference path when it is not already in use.
    import sys  # pylint: disable=import-outside-toplevel

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(py_obj, torch.Tensor):
        return _array_annotation, False

    if isinstance(py_obj, BufferView):
        return _buffer_annotation, False

    if isinstance(py_obj, (bool, np.bool_)):
        return primitive_types.u1, False

    if isinstance(py_obj, np.generic):
        return to_quadrants_type(py_obj.dtype), False

    if isinstance(py_obj, int):
        return impl.get_runtime().default_ip, False

    if isinstance(py_obj, float):
        return impl.get_runtime().default_fp, False

    if isinstance(py_obj, Matrix):
        dtype = py_obj.dtype
        key = (py_obj.n, py_obj.m, py_obj.ndim, dtype)
        if key not in _matrix_types:
            _matrix_types[key] = (
                VectorType(py_obj.n, dtype) if py_obj.ndim == 1 else MatrixType(py_obj.n, py_obj.m, py_obj.ndim, dtype)
            )
        return _matrix_types[key], False

    if isinstance(py_obj, Struct):
        declared = getattr(py_obj, "_qd_struct_type", None)
        if declared is not None:
            return declared, False

        members = {name: infer_member_type(child, f"{path}.{name}", ancestor_ids)[0] for name, child in py_obj.items}
        key = tuple((name, id(dtype)) for name, dtype in members.items())
        if key not in _struct_types:
            _struct_types[key] = StructType(**members)
        return _struct_types[key], True

    return (
        create_deferred_member_type_for_invalid_dataclass_member(
            QuadrantsRuntimeTypeError, f"unsupported runtime type {type(py_obj).__name__}", py_obj
        ),
        False,
    )


def _infer_dataclass_runtime_member_types(
    dataclass_instance: DataclassInstance,
    field_names_and_final: tuple[tuple[str, bool], ...],
    path: str,
    ancestor_ids: frozenset[int],
    non_frozen_ancestor_path: str | None,
    final_value_key_by_value_id: FinalValueKeyCache | None,
) -> tuple[
    "list[InferredMemberType | DeferredMemberType | _SpecialForm]",
    bool,
    bool,
    list[tuple[str, InferredMemberType | DeferredMemberType]],
]:
    """Read members and infer their types, returning (member_types, needs_member_check, has_deferred_error, member_guards).

    needs_member_check means inferred types may need checking before reuse. has_deferred_error marks errors raised only
    if a member is used during compilation. member_guards records names and inferred types needing checks.
    """
    dataclass_type = type(dataclass_instance)
    frozen = is_frozen_dataclass(dataclass_type)
    final_non_frozen_path = non_frozen_ancestor_path
    if not frozen:
        final_non_frozen_path = dataclass_type.__name__
    needs_member_check = not frozen
    member_types: list[InferredMemberType | DeferredMemberType | _SpecialForm] = []
    has_deferred_error = False
    member_guards: list[tuple[str, InferredMemberType | DeferredMemberType]] = []
    for name, is_final in field_names_and_final:
        member = getattr(dataclass_instance, name)
        if is_final:
            needs_member_check = True
            member_type = infer_final_field_annotation(member, final_non_frozen_path, name, final_value_key_by_value_id)
        else:
            member_non_frozen_ancestor_path = None
            if not frozen:
                member_non_frozen_ancestor_path = f"{dataclass_type.__name__}.{name}"
            elif non_frozen_ancestor_path is not None:
                member_non_frozen_ancestor_path = f"{non_frozen_ancestor_path}.{name}"
            member_type, _needs_member_check = infer_member_type(
                member, f"{path}.{name}", ancestor_ids, member_non_frozen_ancestor_path, final_value_key_by_value_id
            )
            needs_member_check |= _needs_member_check
            if _needs_member_check:
                member_guards.append((name, member_type))
        if isinstance(member_type, DeferredMemberType):
            has_deferred_error = True
        elif is_runtime_dataclass_schema(member_type):
            has_deferred_error |= member_type._qd_has_deferred_errors
        member_types.append(member_type)
    return member_types, needs_member_check, has_deferred_error, member_guards


def _create_dataclass_field(
    name: str,
    member_type: "InferredMemberType | DeferredMemberType | _SpecialForm",
) -> dataclasses.Field[Any]:
    """Create a dataclass field with the given name and inferred type."""
    field = dataclasses.field()
    field.name = name
    field.type = member_type
    # dataclasses.fields() includes only entries marked as normal fields.
    field._field_type = _FIELD
    return field


def _store_schema_in_instance_cache(
    dataclass_instance: DataclassInstance,
    schema_by_compiler_settings_and_ancestor_path: dict[
        InferenceContextKey, type[RuntimeDataclassSchema] | SchemaNeedingMemberChecks
    ] | None,
    compiler_settings_and_ancestor_path: InferenceContextKey,
    schema: type[RuntimeDataclassSchema],
    needs_member_check: bool,
    member_guards: list[tuple[str, InferredMemberType | DeferredMemberType]],
) -> None:
    """Cache a schema or its member-check wrapper when the instance supports weak references."""
    if schema_by_compiler_settings_and_ancestor_path is None:
        schema_by_compiler_settings_and_ancestor_path = {}
        try:
            _schema_by_dataclass_instance_then_compiler_settings_and_ancestor_path[dataclass_instance] = (
                schema_by_compiler_settings_and_ancestor_path
            )
        except TypeError as error:
            # Slotted dataclasses may lack weak-reference support; infer without an instance cache.
            _warnings.warn_uncacheable_weakref(error)
            return

    schema_by_compiler_settings_and_ancestor_path[compiler_settings_and_ancestor_path] = (
        schema
        if not needs_member_check
        else SchemaNeedingMemberChecks(schema, member_guards)
    )


def cached_infer_dataclass_schema(
    dataclass_instance: DataclassInstance,
    path: str,
    ancestor_ids: frozenset[int] = frozenset(),
    non_frozen_ancestor_path: str | None = None,
    final_value_key_by_value_id: FinalValueKeyCache | None = None,
) -> type[RuntimeDataclassSchema]:
    """Build or reuse a private schema class, validating mutable members before cache reuse when needed.

    The caller must verify that dataclass_instance is a dataclass instance.
    """
    if id(dataclass_instance) in ancestor_ids:
        raise QuadrantsRuntimeTypeError(f"Dataclass member {path!r} contains a cycle")

    ancestor_ids = ancestor_ids | {id(dataclass_instance)}
    dataclass_type = type(dataclass_instance)
    runtime = impl.get_runtime()
    schema_by_compiler_settings_and_ancestor_path = (
        _schema_by_dataclass_instance_then_compiler_settings_and_ancestor_path.get(dataclass_instance)
    )
    # Numeric defaults affect member types, debug affects diagnostics, and ancestor paths affect Final validation.
    compiler_settings_and_ancestor_path = (
        id(runtime.default_ip),
        id(runtime.default_fp),
        impl.current_cfg().debug,
        non_frozen_ancestor_path,
    )
    cached_schema_entry = (
        schema_by_compiler_settings_and_ancestor_path.get(compiler_settings_and_ancestor_path)
        if schema_by_compiler_settings_and_ancestor_path is not None
        else None
    )
    if cached_schema_entry is not None and not isinstance(cached_schema_entry, SchemaNeedingMemberChecks):
        return cached_schema_entry

    if cached_schema_entry is not None and cached_schema_entry.do_member_check(
        dataclass_instance, path, ancestor_ids, infer_member_type
    ):
        return cached_schema_entry.schema

    field_names_and_final, has_final_fields = _cached_get_field_names_and_finality(dataclass_type)
    member_types, needs_member_check, has_deferred_error, member_guards = _infer_dataclass_runtime_member_types(
        dataclass_instance,
        field_names_and_final,
        path,
        ancestor_ids,
        non_frozen_ancestor_path,
        final_value_key_by_value_id,
    )
    dataclass_type_and_member_types = (
        id(dataclass_type),
        not needs_member_check,
        tuple((name, id(member_type)) for (name, _), member_type in zip(field_names_and_final, member_types)),
    )
    schema = _schema_by_dataclass_type_and_member_types.get(dataclass_type_and_member_types)
    if schema is None:
        fields = {
            name: _create_dataclass_field(name, member_type)
            for (name, _), member_type in zip(field_names_and_final, member_types)
        }
        schema = create_dataclass_schema(dataclass_type, fields, needs_member_check, has_deferred_error)
        _schema_by_dataclass_type_and_member_types[dataclass_type_and_member_types] = schema
    # Deferred descriptions are reusable because debug is part of the instance-cache key.
    # Final-bearing paths still require validation.
    if not needs_member_check or (
        is_frozen_dataclass(dataclass_type) and not has_final_fields and non_frozen_ancestor_path is None
    ):
        _store_schema_in_instance_cache(
            dataclass_instance,
            schema_by_compiler_settings_and_ancestor_path,
            compiler_settings_and_ancestor_path,
            schema,
            needs_member_check,
            member_guards,
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
                cached_infer_dataclass_schema(
                    py_obj, arg_meta.name, final_value_key_by_value_id=final_value_key_by_value_id
                ),
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

    if kernel._runtime_metadata_defaults is not None and kernel._runtime_metadata_defaults != defaults:
        kernel.mapper._mapping_cache.clear()
        kernel.mapper._mapping_cache_tracker.clear()
    kernel._launch_arg_schemas = ()
    # Per-call cache: id(value) -> (value, specialization signature); retain the value to verify its identity.
    final_value_key_by_value_id: FinalValueKeyCache = {}
    kernel.arg_metas = infer_kernel_dataclass_arg_types(kernel.orig_arguments, py_objs, final_value_key_by_value_id)
    kernel.mapper.arguments = kernel.arg_metas
    has_deferred_error = False
    unstable = False
    for meta in kernel.arg_metas:
        schema = meta.annotation
        if hasattr(schema, "_qd_underlying_dataclass"):
            has_deferred_error |= schema._qd_has_deferred_errors
            unstable |= schema._qd_needs_member_check
    kernel._runtime_has_deferred = has_deferred_error
    kernel.template_slot_locations = [
        i
        for i in kernel.original_template_slot_locations
        if not hasattr(kernel.arg_metas[i].annotation, "_qd_underlying_dataclass")
    ]
    kernel.mapper.template_slot_locations = kernel.template_slot_locations
    # A dataclass's member types can change without changing the identity of the outer object.
    kernel.mapper._mapping_cache_disabled = unstable or None
    kernel._runtime_metadata_args_hash = args_hash if not unstable and not has_deferred_error else None
    kernel._runtime_metadata_defaults = defaults

    return final_value_key_by_value_id


@contextmanager
def dataclass_call_arguments(
    func: "QuadrantsCallable | BoundQuadrantsCallable",
    args: Sequence[Any],
    keywords: dict[str, Any],
    bound: bool = False,
) -> Iterator[None]:
    """Use the arriving dataclasses' schemas while compiling an inlined call."""
    wrapper = cast("FuncBase", func.wrapper)
    original_arg_metas = wrapper.arg_metas
    declared_arg_metas = wrapper.orig_arguments
    arg_metas = list(declared_arg_metas)
    offset = int(bound)
    py_objs = {meta.name: arg for meta, arg in zip(arg_metas[offset:], args)}
    py_objs.update(keywords)
    for i, meta in enumerate(arg_metas):
        if meta.name not in py_objs:
            if hasattr(type(meta.default), "__dataclass_fields__"):
                raise QuadrantsRuntimeTypeError(
                    f"Omitted dataclass defaults are not supported in @qd.func. Pass {meta.name!r} explicitly."
                )
            continue
        py_obj = py_objs[meta.name]
        if isinstance(py_obj, type) and hasattr(py_obj, "_qd_underlying_dataclass"):
            schema = py_obj
        elif hasattr(type(py_obj), "__dataclass_fields__"):
            schema = cached_infer_dataclass_schema(cast(DataclassInstance, py_obj), meta.name)
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
