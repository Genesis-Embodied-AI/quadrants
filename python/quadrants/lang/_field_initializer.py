"""Allocation-time field initialization, replayed after placement is complete."""

from typing import TYPE_CHECKING, Callable, TypeAlias, Union, cast

import numpy as np

if TYPE_CHECKING:
    from quadrants._lib.core.quadrants_python import SNodeCxx
    from quadrants.lang.field import ScalarField
    from quadrants.lang.matrix import Matrix, MatrixField


_FillValue: TypeAlias = Union[bool, int, float, np.bool_, np.integer, np.floating, list, tuple, "Matrix"]


def capture_fill_value(value: _FillValue | None, element_shape=()) -> _FillValue | None:
    """Validate and snapshot an initializer without allocating or materializing any fields."""
    if isinstance(value, np.bool_):
        return bool(value)
    if value is None or isinstance(value, (bool, int, float, np.integer, np.floating)):
        return value
    # Import lazily: matrix.py also uses this module when constructing fields.
    from quadrants.lang.matrix import Matrix  # pylint: disable=C0415

    if isinstance(value, Matrix):
        value = value.to_list()
    if not element_shape or not isinstance(value, (list, tuple)) or len(value) != element_shape[0]:
        raise ValueError(f"fill must be a number or a sequence with element shape {element_shape}")
    result = tuple(capture_fill_value(item, element_shape[1:]) for item in value)
    if any(item is None for item in result):
        raise ValueError("fill components must be numbers")
    if len(element_shape) > 1 and any(not isinstance(item, tuple) for item in result):
        raise ValueError(f"fill must have element shape {element_shape}")
    return result


def snode_belongs_to_tree(node: "SNodeCxx | None", root: "SNodeCxx") -> bool:
    """Compare roots without relying on the default tree ID of an unfinalized tree."""
    if node is None:
        return False
    while node.parent is not None:
        node = node.parent
    return node.id == root.id


class FieldInitializers:
    """Runtime-owned initializers; ordinary field.fill calls never enter this queue."""

    def __init__(self) -> None:
        self.pending: list[tuple["ScalarField | MatrixField", _FillValue]] = []

    def add(self, field: "ScalarField | MatrixField", value: _FillValue | None) -> None:
        if value is not None:
            self.pending.append((field, value))

    def clear(self) -> None:
        self.pending.clear()

    def discard_tree(self, root: "SNodeCxx") -> None:
        """Discard pending fills before their allocation tree is destroyed."""
        self.pending = [
            (field, value)
            for field, value in self.pending
            if not any(snode_belongs_to_tree(var.ptr.snode(), root) for var in field.vars)
        ]

    def run(self) -> None:
        if not self.pending:
            return
        # Fill kernels materialize recursively. Detach the queue before launching any of them.
        pending, self.pending = self.pending, []
        for index, (field, value) in enumerate(pending):
            # Field.fill's inherited annotation only covers scalars. Concrete fills also accept the component values
            # already validated by the constructors.
            fill = cast(Callable[[_FillValue], None], field.fill)
            try:
                fill(value)
            except Exception:
                self.pending = pending[index:] + self.pending
                raise
