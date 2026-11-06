# type: ignore
"""Borrowed CSR storage and multiplication into a caller-owned dense array."""

import ast
import dataclasses
import inspect
import math
import numbers

from quadrants._lib import core as _qd_core
from quadrants.lang._ndarray import ScalarNdarray
from quadrants.lang.exception import QuadrantsNameError, QuadrantsSyntaxError
from quadrants.lang.impl import current_cfg, get_runtime, static, static_assert
from quadrants.lang.kernel_impl import data_oriented, func, kernel
from quadrants.types.ndarray_type import NDArray
from quadrants.types.primitive_types import f32, i32

__all__ = ["CSRMatrix"]

_MAX_ELEMENTS = (2**31 - 1) // 4


def _check_array(array, dtype, rank, name):
    if not isinstance(array, ScalarNdarray) or array.dtype != dtype or len(array.shape) != rank:
        raise TypeError(f"CSRMatrix: {name} must be a scalar {dtype} Quadrants ndarray of rank {rank}")
    if any(size <= 0 for size in array.shape) or math.prod(array.shape) > _MAX_ELEMENTS:
        raise ValueError(f"CSRMatrix: {name} must have positive dimensions and at most 2**31 - 1 bytes")


@dataclasses.dataclass(frozen=True)
class CSRMatrix:
    """A sparse matrix borrowing i32 CSR indices and f32 coefficients.

    Create with ``SparseMatrix.from_csr(..., shape=(rows, columns))``. Array bindings
    are fixed; their contents can change between completed operations. Multiplication
    uses ``(matrix @ rhs).write_to(out)`` with rank-2 f32 Quadrants ndarrays.
    """

    row_ptr: NDArray[i32, 1]
    col_idx: NDArray[i32, 1]
    values: NDArray[f32, 1]
    rows: i32
    columns: i32

    def __post_init__(self):
        for name in ("rows", "columns"):
            dimension = getattr(self, name)
            if isinstance(dimension, bool) or not isinstance(dimension, numbers.Integral):
                raise TypeError(f"CSRMatrix: {name} must be an integer")
            if not 0 < dimension <= _MAX_ELEMENTS:
                raise ValueError(f"CSRMatrix: {name} must be positive and fit signed i32 byte addressing")
            object.__setattr__(self, name, int(dimension))
        _check_array(self.row_ptr, i32, 1, "row_ptr")
        _check_array(self.col_idx, i32, 1, "col_idx")
        _check_array(self.values, f32, 1, "values")
        if self.row_ptr.shape[0] != self.rows + 1:
            raise ValueError("CSRMatrix: row_ptr must contain rows + 1 entries")
        if self.col_idx.shape != self.values.shape:
            raise ValueError("CSRMatrix: col_idx and values must have equal capacity")

    @property
    def shape(self):
        return self.rows, self.columns

    def __matmul__(self, rhs):
        _check_array(rhs, f32, 2, "rhs")
        if rhs.shape[0] != self.columns:
            raise ValueError("CSRMatrix: rhs must have shape (columns, K)")
        return _HostCSRProduct(self, rhs)


@dataclasses.dataclass(frozen=True)
class _HostCSRProduct:
    matrix: CSRMatrix
    rhs: ScalarNdarray

    def write_to(self, out):
        _check_array(out, f32, 2, "out")
        if out.shape != (self.matrix.rows, self.rhs.shape[1]):
            raise ValueError("CSRMatrix: out must have shape (rows, K)")
        for source in (self.matrix.row_ptr, self.matrix.col_idx, self.matrix.values, self.rhs):
            if out is source or out.arr is source.arr:
                raise ValueError("CSRMatrix: out must not alias an input array")
        _launch_csr_product(self.matrix, self.rhs, out)


def _require_forward():
    context = get_runtime()._current_global_context
    if context.current_kernel.autodiff_mode != _qd_core.AutodiffMode.NONE:
        raise QuadrantsSyntaxError("CSRMatrix multiplication supports forward execution only")


@func(requires_top_level=True)
def _write_csr(
    row_ptr: NDArray[i32, 1],
    col_idx: NDArray[i32, 1],
    values: NDArray[f32, 1],
    rhs: NDArray[f32, 2],
    out: NDArray[f32, 2],
    rows: i32,
    columns: i32,
):
    _require_forward()
    # Nested functions can receive arrays whose ranks differ from their annotations.
    static_assert(len(row_ptr.shape) == 1, "CSRMatrix: row_ptr must have rank 1")
    static_assert(len(col_idx.shape) == 1, "CSRMatrix: col_idx must have rank 1")
    static_assert(len(values.shape) == 1, "CSRMatrix: values must have rank 1")
    static_assert(len(rhs.shape) == 2, "CSRMatrix: rhs must have rank 2")
    static_assert(len(out.shape) == 2, "CSRMatrix: out must have rank 2")

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
        and rhs.shape[0] == columns
        and rhs.shape[1] == width
        and out.shape[0] == rows
    )
    extent = 0
    if valid:
        extent = rows * width
    # Keep the range loop top level: an invalid shape launches no multiplication,
    # including in release mode where assertions are disabled.
    for element in range(extent):
        row = element // width
        channel = element % width
        accumulator = f32(0.0)
        for edge in range(row_ptr[row], row_ptr[row + 1]):
            accumulator += values[edge] * rhs[col_idx[edge], channel]
        out[row, channel] = accumulator

    # A GPU assertion can end its task; publish and consume the safe bound first.
    if static(current_cfg().debug):
        assert valid, "CSRMatrix: incompatible array shapes or i32 byte limits"


@data_oriented
class _KernelCSRProduct:
    def __init__(self, row_ptr, col_idx, values, rhs, rows, columns):
        self.row_ptr = row_ptr
        self.col_idx = col_idx
        self.values = values
        self.rhs = rhs
        self.rows = rows
        self.columns = columns

    @func(requires_top_level=True)
    def write_to(self, out: NDArray[f32, 2]):
        _write_csr(self.row_ptr, self.col_idx, self.values, self.rhs, out, self.rows, self.columns)


@func
def _make_product(matrix: CSRMatrix, rhs: NDArray[f32, 2]):
    return _KernelCSRProduct(matrix.row_ptr, matrix.col_idx, matrix.values, rhs, matrix.rows, matrix.columns)


@kernel
def _launch_csr_product(matrix: CSRMatrix, rhs: NDArray[f32, 2], out: NDArray[f32, 2]):
    _make_product(matrix, rhs).write_to(out)


def _static_operand(ctx, node):
    if isinstance(node, ast.Name):
        return ctx.get_var_by_name(node.id)[1]
    if isinstance(node, ast.Attribute):
        owner = _static_operand(ctx, node.value)
        return inspect.getattr_static(owner, node.attr, None)
    return None


def _lower_matmul(ctx, node, build_stmt):
    """Route CSR @ through the ordinary typed-call expansion and pruning path."""
    try:
        operand = _static_operand(ctx, node.left)
    except QuadrantsNameError:
        return False
    if operand is not CSRMatrix and not isinstance(operand, CSRMatrix):
        return False

    name = "_qd_csr_make_product"
    while ctx.is_var_declared(name) or name in ctx.template_vars or name in ctx.global_vars:
        name += "_"
    # Each compilation context owns its globals copy; no user namespace is changed.
    ctx.global_vars[name] = _make_product
    helper = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node.left)
    call = ast.copy_location(ast.Call(func=helper, args=[node.left, node.right], keywords=[]), node)
    ast.fix_missing_locations(call)
    try:
        node.ptr = build_stmt(ctx, call)
    finally:
        del ctx.global_vars[name]
    return True
