"""Validate mutable descendants before reusing a frozen dataclass's inferred schema."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants.lang._ndarray import Ndarray
from quadrants.lang.field import Field

if TYPE_CHECKING:
    from typing import _SpecialForm

    from quadrants.lang._runtime_typing.types import (
        DataclassInstance,
        DeferredMemberType,
        InferredMemberType,
        RuntimeDataclassSchema,
    )


class SchemaNeedingMemberChecks:
    """Store member names and types, never child objects that could retain the owning dataclass."""

    __slots__ = ("schema", "member_guards")

    def __init__(
        self,
        value: "DataclassInstance",
        schema: "type[RuntimeDataclassSchema]",
        field_names_and_final: Sequence[tuple[str, bool]],
        member_types: "Sequence[InferredMemberType | DeferredMemberType | _SpecialForm]",
    ) -> None:
        """Record guards for members whose runtime representation can change inside a frozen dataclass."""
        self.schema = schema
        self.member_guards = []
        for (name, is_final), member_type in zip(field_names_and_final, member_types):
            if is_final:
                # Final values are still validated by specialization and launch handling; their declared type is fixed.
                continue
            child = getattr(value, name)
            if type(child) in _TENSOR_WRAPPER_TYPES:
                continue
            if hasattr(member_type, "_qd_underlying_dataclass"):
                if not cast("type[RuntimeDataclassSchema]", member_type)._qd_needs_member_check:
                    continue
            elif child is None or isinstance(child, (bool, int, float, np.generic, Ndarray, Field)):
                continue
            self.member_guards.append((name, member_type))

    def do_member_check(
        self,
        dataclass_instance: "DataclassInstance",
        path: str,
        ancestor_ids: frozenset[int],
        infer_member_type: "Callable[[Any, str, frozenset[int]], InferredMemberType | DeferredMemberType]",
    ) -> bool:
        """Check whether the current member types still match the cached schema."""
        for name, member_type in self.member_guards:
            child = getattr(dataclass_instance, name)
            if infer_member_type(child, f"{path}.{name}", ancestor_ids) is not member_type:
                return False
        return True
