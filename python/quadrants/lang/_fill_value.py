"""Shared validation and snapshots for allocation-time fill values."""

import numbers
from typing import TYPE_CHECKING, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    from quadrants.lang.matrix import Matrix


_FillValue: TypeAlias = Union[int, float, numbers.Number, np.number, list, tuple, "Matrix"]


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

