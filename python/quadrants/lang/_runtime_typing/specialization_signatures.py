"""Build specialization signatures for runtime-typed dataclasses."""

from collections.abc import Sequence
from typing import Any

from quadrants.lang._runtime_typing.types import (
    FinalValueKeyCache,
    RuntimeDataclassSchema,
)
from quadrants.lang._template_mapper_hotpath import _extract_arg
from quadrants.lang.kernel_arguments import ArgMetadata
from quadrants.lang.util import get_member_at_path


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
