# type: ignore
"""Experimental scalar float32 CSR multiplication with a dense feature matrix."""

from quadrants.lang.impl import static_assert
from quadrants.lang.kernel_impl import func
from quadrants.types.ndarray_type import NDArray
from quadrants.types.primitive_types import f32, i32


@func(requires_top_level=True)
def csr_spmm(
    row_ptr: NDArray[i32, 1],
    col_idx: NDArray[i32, 1],
    values: NDArray[f32, 1],
    rhs: NDArray[f32, 2],
    out: NDArray[f32, 2],
):
    """Overwrite ``out`` with ``CSR(row_ptr, col_idx, values) @ rhs``.

    Call at the top level of a kernel with contiguous scalar ``qd.ndarray`` buffers. ``row_ptr`` describes
    incoming edges for each output row; duplicate and unsorted columns remain distinct stored edges. Empty rows
    write zero, and storage after ``row_ptr[-1]`` is ignored. No temporary storage or synchronization is added.

    Callers validate compatible positive shapes, valid CSR contents, nonoverlapping output and input buffers,
    and a byte count of at most ``2**31 - 1`` for every buffer before allocation. Inputs must remain stable during
    execution. See the CSR SpMM user guide and checked preparation example for the complete contract.
    """
    # Nested function annotations do not currently retain rank metadata. Check all ranks before indexing shapes.
    static_assert(len(row_ptr.shape) == 1, "csr_spmm: row_ptr must have rank 1")
    static_assert(len(col_idx.shape) == 1, "csr_spmm: col_idx must have rank 1")
    static_assert(len(values.shape) == 1, "csr_spmm: values must have rank 1")
    static_assert(len(rhs.shape) == 2, "csr_spmm: rhs must have rank 2")
    static_assert(len(out.shape) == 2, "csr_spmm: out must have rank 2")

    for element in range(out.shape[0] * out.shape[1]):
        row = element // out.shape[1]
        channel = element % out.shape[1]
        accumulator = f32(0.0)
        for edge in range(row_ptr[row], row_ptr[row + 1]):
            accumulator += values[edge] * rhs[col_idx[edge], channel]
        out[row, channel] = accumulator
