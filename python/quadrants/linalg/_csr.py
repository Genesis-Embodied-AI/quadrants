# type: ignore
"""Borrowed CSR storage and sparse-dense matrix multiplication into a caller-owned array."""

import math
import numbers
from typing import Any

from quadrants._lib import core as _qd_core
from quadrants.lang._ndarray import ScalarNdarray
from quadrants.lang._quadrants_callable import QuadrantsCallable
from quadrants.lang.exception import (
    QuadrantsRuntimeError,
    QuadrantsSyntaxError,
    QuadrantsTypeError,
)
from quadrants.lang.func import Func
from quadrants.lang.impl import (
    current_cfg,
    get_runtime,
    inside_kernel,
    static,
    static_assert,
)
from quadrants.lang.kernel_impl import func, kernel
from quadrants.linalg.sparse_matrix import SparseMatrix
from quadrants.types.ndarray_type import NDArray
from quadrants.types.primitive_types import f32, i32

__all__ = ["matmul"]

_MAX_ELEMENTS = (2**31 - 1) // 4


def _check_array(array: Any, dtype: Any, rank: int, name: str, context: str) -> None:
    if not isinstance(array, ScalarNdarray) or array.dtype != dtype or len(array.shape) != rank:
        raise TypeError(f"{context}: {name} must be a scalar {dtype} Quadrants ndarray of rank {rank}")
    if any(size <= 0 for size in array.shape) or math.prod(array.shape) > _MAX_ELEMENTS:
        raise ValueError(f"{context}: {name} must have positive dimensions and at most 2**31 - 1 bytes")


def _from_csr(
    cls: type[SparseMatrix],
    row_ptr: Any,
    col_idx: Any,
    values: Any,
    *,
    shape: Any,
) -> SparseMatrix:
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("SparseMatrix.from_csr: shape must be a tuple of two positive integers")
    for name, dimension in zip(("rows", "columns"), shape):
        if isinstance(dimension, bool) or not isinstance(dimension, numbers.Integral):
            raise TypeError(f"SparseMatrix.from_csr: {name} must be an integer")
        if not 0 < dimension <= _MAX_ELEMENTS:
            raise ValueError(f"SparseMatrix.from_csr: {name} must be positive and fit signed i32 byte addressing")
    rows, columns = int(shape[0]), int(shape[1])
    _check_array(row_ptr, i32, 1, "row_ptr", "SparseMatrix.from_csr")
    _check_array(col_idx, i32, 1, "col_idx", "SparseMatrix.from_csr")
    _check_array(values, f32, 1, "values", "SparseMatrix.from_csr")
    if row_ptr.shape[0] != rows + 1:
        raise ValueError("SparseMatrix.from_csr: row_ptr must contain rows + 1 entries")
    if col_idx.shape != values.shape:
        raise ValueError("SparseMatrix.from_csr: col_idx and values must have equal capacity")

    instance = object.__new__(cls)
    object.__setattr__(instance, "dtype", f32)
    object.__setattr__(instance, "n", rows)
    object.__setattr__(instance, "m", columns)
    object.__setattr__(instance, "matrix", None)
    object.__setattr__(instance, "row_ptr", row_ptr)
    object.__setattr__(instance, "col_idx", col_idx)
    object.__setattr__(instance, "values", values)
    object.__setattr__(instance, "rows", rows)
    object.__setattr__(instance, "columns", columns)
    object.__setattr__(instance, "_is_csr", True)
    return instance


def _require_forward() -> None:
    context = get_runtime()._current_global_context
    assert context is not None and context.current_kernel is not None
    if context.current_kernel.autodiff_mode != _qd_core.AutodiffMode.NONE:
        raise QuadrantsSyntaxError("matmul with CSR SparseMatrix supports forward execution only")


@func(requires_top_level=True)
def _write_csr(
    row_ptr: NDArray[i32, 1],
    col_idx: NDArray[i32, 1],
    values: NDArray[f32, 1],
    b: NDArray[f32, 2],
    out: NDArray[f32, 2],
    rows: i32,
    columns: i32,
) -> None:
    _require_forward()
    # Nested functions can receive arrays whose ranks differ from their annotations.
    static_assert(len(row_ptr.shape) == 1, "matmul: row_ptr must have rank 1")
    static_assert(len(col_idx.shape) == 1, "matmul: col_idx must have rank 1")
    static_assert(len(values.shape) == 1, "matmul: values must have rank 1")
    static_assert(len(b.shape) == 2, "matmul: b must have rank 2")
    static_assert(len(out.shape) == 2, "matmul: out must have rank 2")

    width = out.shape[1]
    # Division keeps validation within i32 even for incompatible runtime dimensions.
    valid = (
        rows > 0
        and columns > 0
        and width > 0
        and width <= _MAX_ELEMENTS
        and rows <= _MAX_ELEMENTS // max(width, 1)
        and columns <= _MAX_ELEMENTS // max(width, 1)
        and row_ptr.shape[0] - 1 == rows
        and row_ptr.shape[0] <= _MAX_ELEMENTS
        and col_idx.shape[0] > 0
        and col_idx.shape[0] <= _MAX_ELEMENTS
        and col_idx.shape[0] == values.shape[0]
        and b.shape[0] == columns
        and b.shape[1] == width
        and out.shape[0] == rows
    )
    extent = 0
    if valid:
        extent = rows * width
    # Keep the range loop top level: an invalid shape launches no multiplication,
    # including in release mode where assertions are disabled.
    for element in range(extent):
        row = element // width
        col = element % width
        acc = f32(0.0)
        for entry in range(row_ptr[row], row_ptr[row + 1]):
            acc += values[entry] * b[col_idx[entry], col]
        out[row, col] = acc

    # A GPU assertion can end its task; publish and consume the safe bound first.
    if static(current_cfg().debug):
        assert valid, "matmul: incompatible array shapes or i32 byte limits"


@kernel
def _launch_csr_matmul(a: SparseMatrix, b: NDArray[f32, 2], out: NDArray[f32, 2]) -> None:
    _write_csr(a.row_ptr, a.col_idx, a.values, b, out, a.rows, a.columns)


def _host_matmul(a: Any, b: Any, *, out: Any) -> Any:
    if not isinstance(a, SparseMatrix):
        raise TypeError("matmul: first argument must be a SparseMatrix created with SparseMatrix.from_csr(...)")
    if a.matrix is not None:
        raise QuadrantsRuntimeError("matmul requires a CSR SparseMatrix created with SparseMatrix.from_csr(...)")
    _check_array(b, f32, 2, "b", "matmul")
    _check_array(out, f32, 2, "out", "matmul")
    if b.shape[0] != a.columns:
        raise ValueError("matmul: b must have shape (columns, K)")
    if out.shape != (a.rows, b.shape[1]):
        raise ValueError("matmul: out must have shape (rows, K)")
    for source in (a.row_ptr, a.col_idx, a.values, b):
        if out is source or out.arr is source.arr:
            raise ValueError("matmul: out must not alias an input array")
    _launch_csr_matmul(a, b, out)
    return out


_host_matmul.__name__ = "matmul"
_host_matmul.__qualname__ = "matmul"


def _matmul_kernel_body(a: SparseMatrix, b: NDArray[f32, 2], out: NDArray[f32, 2]) -> None:
    """Multiply a CSR ``SparseMatrix`` ``a`` with dense ``f32`` matrix ``b`` into ``out``.

    Callable from Python host scope or at the top level of a ``@qd.kernel`` as
    ``qd.matmul(A, X, out=Y)``.
    """
    _write_csr(a.row_ptr, a.col_idx, a.values, b, out, a.rows, a.columns)


_matmul_kernel_body.__name__ = "matmul"
_matmul_kernel_body.__qualname__ = "matmul"


class _MatmulFunc(Func):
    def __call__(self, *py_args: Any, **kwargs: Any) -> Any:
        if not inside_kernel():
            return _host_matmul(*py_args, **kwargs)
        if "out" not in kwargs:
            raise QuadrantsSyntaxError("matmul requires keyword-only argument `out=`.")
        if len(py_args) < 5 and not any(key.startswith("__qd_a__qd_") for key in kwargs):
            raise QuadrantsTypeError(
                "matmul expects a CSR SparseMatrix (from SparseMatrix.from_csr(...)) as its first argument."
            )
        return super().__call__(*py_args, **kwargs)


def _build_matmul_callable() -> Any:
    fn = _MatmulFunc(_matmul_kernel_body)
    callable_obj = QuadrantsCallable(_matmul_kernel_body, fn)
    callable_obj._is_quadrants_function = True
    callable_obj._is_real_function = False
    callable_obj._qd_requires_top_level = True
    return callable_obj


matmul = _build_matmul_callable()
