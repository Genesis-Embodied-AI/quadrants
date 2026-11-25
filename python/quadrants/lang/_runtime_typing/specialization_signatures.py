"""Prepare runtime dataclass arguments and build their specialization signatures."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from quadrants.lang._runtime_typing.dataclass_schemas import (
    infer_kernel_dataclass_arg_types,
)
from quadrants.lang._runtime_typing.types import (
    FinalValueKeyCache,
    RuntimeDataclassSchema,
)
from quadrants.lang._template_mapper_hotpath import _extract_arg
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import get_member_at_path

if TYPE_CHECKING:
    from quadrants.lang.kernel import Kernel


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


def extract_dataclass_specialization_signatures(
    raise_on_templated_floats: bool,
    py_args: tuple[Any, ...],
    arg_metas: Sequence[ArgMetadata],
    launch_arg_schemas: Sequence[tuple[int, tuple[str, ...], type[RuntimeDataclassSchema]]],
    final_value_key_by_value_id: FinalValueKeyCache | None,
) -> tuple[tuple[int, tuple[str, ...], Any], ...]:
    """Return argument indices, attribute paths, and signatures for dataclasses beneath data-oriented arguments."""
    return tuple(
        (
            index,
            path,
            _extract_arg(
                raise_on_templated_floats,
                get_member_at_path(py_args[index], path),
                schema,
                ".".join((arg_metas[index].name, *path)),
                final_value_key_by_value_id,
            ),
        )
        for index, path, schema in launch_arg_schemas
    )
