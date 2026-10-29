"""Allocation-time field initialization, replayed after placement is complete."""

from typing import TYPE_CHECKING, Callable, cast

from quadrants.lang._fill_value import _FillValue

if TYPE_CHECKING:
    from quadrants.lang.field import ScalarField
    from quadrants.lang.matrix import MatrixField


class FieldInitializers:
    """Runtime-owned initializers; ordinary field.fill calls never enter this queue."""

    def __init__(self) -> None:
        self.pending: list[tuple["ScalarField | MatrixField", _FillValue]] = []

    def add(self, field: "ScalarField | MatrixField", value: _FillValue | None) -> None:
        if value is not None:
            self.pending.append((field, value))

    def clear(self) -> None:
        self.pending.clear()

    def run(self) -> None:
        if not self.pending:
            return
        # Fill kernels materialize recursively. Detach the queue before launching any of them.
        pending, self.pending = self.pending, []
        for field, value in pending:
            # Field.fill's inherited annotation only covers scalars. Concrete fills also accept the component values
            # already validated by the constructors.
            fill = cast(Callable[[_FillValue], None], field.fill)
            fill(value)
