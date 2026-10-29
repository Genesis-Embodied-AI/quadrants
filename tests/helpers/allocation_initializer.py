"""Helpers shared by allocation initializer tests."""

from typing import Any

import quadrants as qd
from quadrants._lib.core.quadrants_python import DataTypeCxx


def _create_array(
    container_type: str,
    backend: qd.Backend | None,
    element_class: str,
    shape: tuple[int, ...],
    *,
    dtype: DataTypeCxx = qd.f32,
    via_dtype: bool = False,
    **kwargs: Any,
) -> qd.Field | qd.Ndarray | qd.Tensor:
    if container_type == "field":
        scalar_factory, vector_factory, matrix_factory = qd.field, qd.Vector.field, qd.Matrix.field
    elif container_type == "ndarray":
        scalar_factory, vector_factory, matrix_factory = qd.ndarray, qd.Vector.ndarray, qd.Matrix.ndarray
    elif container_type == "tensor":
        scalar_factory, vector_factory, matrix_factory = qd.tensor, qd.Vector.tensor, qd.Matrix.tensor
        kwargs["backend"] = backend
    else:
        raise ValueError(f"Unknown container type: {container_type}")

    if element_class == "scalar":
        return scalar_factory(dtype=dtype, shape=shape, **kwargs)
    if element_class == "vector":
        if via_dtype:
            return scalar_factory(dtype=qd.types.vector(n=3, dtype=dtype), shape=shape, **kwargs)
        return vector_factory(n=3, dtype=dtype, shape=shape, **kwargs)
    if element_class == "matrix":
        if via_dtype:
            return scalar_factory(dtype=qd.types.matrix(n=2, m=3, dtype=dtype), shape=shape, **kwargs)
        return matrix_factory(n=2, m=3, dtype=dtype, shape=shape, **kwargs)
    raise ValueError(f"Unknown element class: {element_class}")


def _get_underlying_field(array: qd.Field | qd.Tensor) -> qd.Field:
    # Return the underlying field so tests can inspect its allocation tree, unwrapping a Tensor if needed.
    return array._unwrap() if isinstance(array, qd.Tensor) else array
