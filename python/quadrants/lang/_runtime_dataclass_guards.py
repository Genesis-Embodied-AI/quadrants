"""Validate mutable descendants before reusing a frozen dataclass's inferred schema."""

import numpy as np

from quadrants._tensor_wrapper import Tensor
from quadrants.lang._ndarray import Ndarray
from quadrants.lang.field import Field
from quadrants.types.ndarray_type import NdarrayType


class FrozenSchemaEntry:
    """Keep guards on values owned by the frozen container, without making its launch bindings immutable."""

    __slots__ = ("schema", "tensor_guards", "member_guards")

    def __init__(self, value, schema, field_plan, member_types):
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
                    self.tensor_guards.append((child, type(child), type(backing)))
                    continue
            if hasattr(member_type, "_qd_runtime_origin"):
                if member_type._qd_runtime_stable:
                    continue
            elif child is None or isinstance(child, (bool, int, float, np.generic, Ndarray, Field)):
                continue
            self.member_guards.append((name, child, member_type))

    def matches(self, path, visiting, infer_member_type):
        for tensor, wrapper_type, backing_type in self.tensor_guards:
            if type(tensor) is not wrapper_type or type(tensor._unwrap()) is not backing_type:
                return False
        for name, child, member_type in self.member_guards:
            path.append(name)
            try:
                if infer_member_type(child, path, visiting) is not member_type:
                    return False
            finally:
                path.pop()
        return True
