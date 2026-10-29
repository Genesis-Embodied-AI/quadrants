"""Allocation-time initialization for ndarray factories."""

from typing import TYPE_CHECKING, TypeVar, cast

from quadrants.lang._field_initializer import _FillValue

if TYPE_CHECKING:
    import torch

    from quadrants.lang._ndarray import Ndarray


_Array = TypeVar("_Array", bound="Ndarray | torch.Tensor")


def apply_initial_ndarray_fill(array: _Array, fill: _FillValue | None) -> _Array:
    """Apply an optional initial fill to an allocated ndarray."""
    if fill is None:
        return array

    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel
    from quadrants.lang._field_initializer import (  # pylint: disable=import-outside-toplevel
        capture_fill_value,
    )

    if impl.is_python_backend():
        python_array = cast("torch.Tensor", array)
        captured_fill_value = capture_fill_value(fill, tuple(python_array.size()[len(python_array.shape) :]))
        python_array.copy_(python_array.new_tensor(captured_fill_value))
        return array
    ndarray = cast("Ndarray", array)
    captured_fill_value = capture_fill_value(fill, ndarray.element_shape)
    ndarray.fill(captured_fill_value)
    return array
