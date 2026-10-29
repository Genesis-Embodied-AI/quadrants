"""Allocation-time fill validation and initialization for fields and ndarrays."""

import numbers
from typing import TYPE_CHECKING, Callable, TypeAlias, TypeVar, Union, cast

import numpy as np

if TYPE_CHECKING:
    import torch

    from quadrants.lang._ndarray import Ndarray
    from quadrants.lang.field import ScalarField
    from quadrants.lang.matrix import Matrix, MatrixField


_FillValue: TypeAlias = Union[int, float, numbers.Number, np.number, list, tuple, "Matrix"]
_Array = TypeVar("_Array", bound="Ndarray | torch.Tensor")


def capture_fill_value(value: _FillValue | None, element_shape=()) -> _FillValue | None:
    """Validate and snapshot an initializer without allocating storage or materializing fields."""
    if value is None or isinstance(value, numbers.Number):
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


def apply_initial_ndarray_fill(array: _Array, fill: _FillValue | None) -> _Array:
    """Apply an optional initial fill to an allocated ndarray."""
    if fill is None:
        return array

    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

    if impl.is_python_backend():
        python_array = cast("torch.Tensor", array)
        captured_fill_value = capture_fill_value(fill, tuple(python_array.size()[len(python_array.shape) :]))
        python_array.copy_(python_array.new_tensor(captured_fill_value))
        return array
    ndarray = cast("Ndarray", array)
    captured_fill_value = capture_fill_value(fill, ndarray.element_shape)
    ndarray.fill(captured_fill_value)
    return array
