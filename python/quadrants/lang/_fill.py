"""Constant construction shared by field and ndarray fill kernels."""

from typing import Any

import numpy as np

from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants.lang.expr import Expr
from quadrants.lang.matrix import Matrix
from quadrants.types.primitive_types import integer_types


def make_fill_value(value: Any, dtype: DataTypeCxx) -> Expr:
    """Construct integer constants with the destination type before the default type can truncate them."""
    if isinstance(value, Matrix):
        value = value.to_list()

    def convert(component: Any) -> Any:
        if isinstance(component, (list, tuple)):
            return tuple(convert(item) for item in component)
        if isinstance(component, (int, np.integer)) and not isinstance(component, bool) and dtype in integer_types:
            return Expr(component, dtype=dtype)
        return component

    return Expr(convert(value))
