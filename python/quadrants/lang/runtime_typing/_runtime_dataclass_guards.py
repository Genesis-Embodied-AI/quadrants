"""Validate mutable descendants before reusing a frozen dataclass's inferred schema."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from quadrants._tensor_wrapper import Tensor
from quadrants.lang._ndarray import Ndarray
from quadrants.lang.field import Field
from quadrants.types.ndarray_type import NdarrayType

if TYPE_CHECKING:
    import dataclasses
    from typing import _SpecialForm

    from quadrants.lang.runtime_typing._deferred_errors import DeferredMemberType
    from quadrants.lang.runtime_typing._runtime_dataclass import DataclassInstance, InferredMemberType


class FrozenSchemaEntry:
    """Store member names and types, never child objects that could retain the owning dataclass."""

    __slots__ = ("schema", "tensor_guards", "member_guards")

    def __init__(
        self,
        value: "DataclassInstance",
        schema: type,
        field_plan: Sequence[tuple[str, "dataclasses.Field[Any]", bool]],
        member_types: "Sequence[InferredMemberType | DeferredMemberType | _SpecialForm]",
    ) -> None:
        """Record guards for members whose runtime representation can change inside a frozen dataclass."""
        self.schema = schema
        self.tensor_guards = []
        self.member_guards = []
        for (name, _, is_final), member_type in zip(field_plan, member_types):
            if is_final:
                # Final values are still validated by specialization and launch handling; their declared type is fixed.
                continue
            child = getattr(value, name)
            if isinstance(child, Tensor):
                backing = child._unwrap()
                if (isinstance(backing, Ndarray) and isinstance(member_type, NdarrayType)) or (
                    isinstance(backing, Field) and member_type is Tensor
                ):
                    self.tensor_guards.append((name, type(child), type(backing)))
                    continue
            if hasattr(member_type, "_qd_runtime_origin"):
                if getattr(member_type, "_qd_runtime_stable"):
                    continue
            elif child is None or isinstance(child, (bool, int, float, np.generic, Ndarray, Field)):
                continue
            self.member_guards.append((name, member_type))

    def matches(
        self,
        dataclass_instance: "DataclassInstance",
        path: str,
        ancestor_ids: frozenset[int],
        infer_member_type: "Callable[[Any, str, frozenset[int]], InferredMemberType | DeferredMemberType]",
    ) -> bool:
        """Check whether the current members still match the cached schema's wrapper and type guards."""
        for name, wrapper_type, backing_type in self.tensor_guards:
            tensor = getattr(dataclass_instance, name)
            if type(tensor) is not wrapper_type or type(tensor._unwrap()) is not backing_type:
                return False
        for name, member_type in self.member_guards:
            child = getattr(dataclass_instance, name)
            if infer_member_type(child, f"{path}.{name}", ancestor_ids) is not member_type:
                return False
        return True
