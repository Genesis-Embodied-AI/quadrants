"""Validate mutable descendants before reusing a frozen dataclass's inferred schema."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from quadrants.lang._runtime_typing.types import DeferredMemberType

if TYPE_CHECKING:
    from quadrants.lang._runtime_typing.types import (
        DataclassInstance,
        InferredMemberType,
        RuntimeDataclassSchema,
    )


class SchemaNeedingMemberChecks:
    """Store member names and types, never child objects that could retain the owning dataclass."""

    __slots__ = ("schema", "member_guards")

    def __init__(
        self,
        schema: "type[RuntimeDataclassSchema]",
        member_guards: "list[tuple[str, InferredMemberType | DeferredMemberType]]",
    ) -> None:
        """Store the member checks selected during runtime type inference."""
        self.schema = schema
        self.member_guards = member_guards

    def do_member_check(
        self,
        dataclass_instance: "DataclassInstance",
        path: str,
        ancestor_ids: frozenset[int],
        infer_member_type: "Callable[[Any, str, frozenset[int]], tuple[InferredMemberType | DeferredMemberType, bool]]",
    ) -> bool:
        """Check whether the current member types still match the cached schema."""
        for name, member_type in self.member_guards:
            child = getattr(dataclass_instance, name)
            current_member_type, _ = infer_member_type(child, f"{path}.{name}", ancestor_ids)
            if current_member_type is not member_type:
                return False
        return True
