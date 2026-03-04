"""Tests for physical storage buffer pointer correctness on Metal.

Regression tests for:
1. A Metal shader compiler bug where stores through physical GPU pointers
   are silently dropped when the byte offset involves a runtime stride
   multiply and the stored value is loop-invariant.
2. Atomic operations on ndarrays accessed via physical storage buffers.

See doc/metal_physical_ptr_miscompile.md in perso_hugh for full details.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

archs_with_physical_storage_buffer = [qd.metal, qd.vulkan]


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32])
@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_2d_const_store_static_ndrange(dtype):
    """Constant store to 2D ndarray with static ndrange bounds."""
    np_dtype = np.float32 if dtype == qd.f32 else np.int32
    rows, cols = 4, 7

    @qd.kernel
    def fill(arr: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(4, 7):
            arr[i, j] = 99

    arr = np.zeros((rows, cols), dtype=np_dtype)
    fill(arr)
    qd.sync()
    np.testing.assert_array_equal(arr, np.full((rows, cols), 99, dtype=np_dtype))


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_2d_row_store_static_ndrange():
    """Row-dependent (but column-independent) store to 2D ndarray."""
    rows, cols = 4, 7

    @qd.kernel
    def fill(arr: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(4, 7):
            arr[i, j] = i

    arr = np.zeros((rows, cols), dtype=np.int32)
    fill(arr)
    qd.sync()
    expected = np.array([[i] * cols for i in range(rows)], dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_2d_idx_store_static_ndrange():
    """Index-dependent store to 2D ndarray (this case already passed)."""
    rows, cols = 4, 7

    @qd.kernel
    def fill(arr: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(4, 7):
            arr[i, j] = i * 100 + j

    arr = np.zeros((rows, cols), dtype=np.int32)
    fill(arr)
    qd.sync()
    expected = np.array([[i * 100 + j for j in range(cols)] for i in range(rows)], dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_2d_const_store_various_shapes():
    """Constant store across multiple 2D shapes."""
    shapes = [(2, 3), (3, 5), (8, 8), (1, 16), (16, 1)]
    for rows, cols in shapes:

        @qd.kernel
        def fill(arr: qd.types.NDArray) -> None:
            for i, j in qd.ndrange(rows, cols):
                arr[i, j] = 42

        arr = np.zeros((rows, cols), dtype=np.int32)
        fill(arr)
        qd.sync()
        np.testing.assert_array_equal(
            arr,
            np.full((rows, cols), 42, dtype=np.int32),
            err_msg=f"Failed for shape ({rows}, {cols})",
        )


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_2d_copy_const_static_ndrange():
    """Two-argument kernel: write constant to dst via static ndrange."""
    rows, cols = 4, 7

    @qd.kernel
    def copy_const(src: qd.types.NDArray, dst: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(4, 7):
            dst[i, j] = 77

    src = np.ones((rows, cols), dtype=np.int32)
    dst = np.zeros((rows, cols), dtype=np.int32)
    copy_const(src, dst)
    qd.sync()
    np.testing.assert_array_equal(dst, np.full((rows, cols), 77, dtype=np.int32))


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_2d_copy_src_static_ndrange():
    """Copy from src to dst using static ndrange (src values are row-uniform)."""
    rows, cols = 4, 7

    @qd.kernel
    def copy(src: qd.types.NDArray, dst: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(4, 7):
            dst[i, j] = src[i, j]

    src = np.full((rows, cols), 55, dtype=np.int32)
    dst = np.zeros((rows, cols), dtype=np.int32)
    copy(src, dst)
    qd.sync()
    np.testing.assert_array_equal(dst, src)


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_3d_const_store_static_ndrange():
    """Constant store to a 3D ndarray with static ndrange."""
    d0, d1, d2 = 3, 4, 5

    @qd.kernel
    def fill(arr: qd.types.NDArray) -> None:
        for i, j, k in qd.ndrange(3, 4, 5):
            arr[i, j, k] = 123

    arr = np.zeros((d0, d1, d2), dtype=np.int32)
    fill(arr)
    qd.sync()
    np.testing.assert_array_equal(arr, np.full((d0, d1, d2), 123, dtype=np.int32))


# --- Atomic operations on ndarrays via physical storage buffers ---


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_atomic_add_1d():
    """atomic_add on a 1D ndarray element from multiple threads."""
    n = 1024

    @qd.kernel
    def reduce_sum(arr: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            qd.atomic_add(out[0], arr[i])

    arr = np.ones(n, dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    reduce_sum(arr, out)
    qd.sync()
    assert out[0] == n


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_atomic_add_2d():
    """atomic_add accumulating into each row of a 2D ndarray."""
    rows, cols = 4, 128

    @qd.kernel
    def row_sums(src: qd.types.NDArray, dst: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(rows, cols):
            qd.atomic_add(dst[i, 0], src[i, j])

    src = np.ones((rows, cols), dtype=np.int32)
    dst = np.zeros((rows, cols), dtype=np.int32)
    row_sums(src, dst)
    qd.sync()
    for i in range(rows):
        assert dst[i, 0] == cols, f"row {i}: expected {cols}, got {dst[i, 0]}"


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_atomic_add_f32():
    """atomic_add with float values on a 1D ndarray."""
    n = 512

    @qd.kernel
    def reduce_sum(arr: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            qd.atomic_add(out[0], arr[i])

    arr = np.full(n, 0.25, dtype=np.float32)
    out = np.zeros(1, dtype=np.float32)
    reduce_sum(arr, out)
    qd.sync()
    np.testing.assert_allclose(out[0], n * 0.25, rtol=1e-5)


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_atomic_sub():
    """atomic_sub on an ndarray element."""
    n = 256

    @qd.kernel
    def reduce_sub(arr: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            qd.atomic_sub(out[0], arr[i])

    arr = np.ones(n, dtype=np.int32)
    out = np.zeros(1, dtype=np.int32)
    reduce_sub(arr, out)
    qd.sync()
    assert out[0] == -n


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_atomic_min_max():
    """atomic_min and atomic_max on ndarray elements."""
    n = 256

    @qd.kernel
    def find_min(arr: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            qd.atomic_min(out[0], arr[i])

    @qd.kernel
    def find_max(arr: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            qd.atomic_max(out[0], arr[i])

    arr = np.arange(n, dtype=np.int32)

    out_min = np.full(1, 999999, dtype=np.int32)
    find_min(arr, out_min)
    qd.sync()
    assert out_min[0] == 0

    out_max = np.full(1, -999999, dtype=np.int32)
    find_max(arr, out_max)
    qd.sync()
    assert out_max[0] == n - 1


@test_utils.test(arch=archs_with_physical_storage_buffer)
def test_ndarray_atomic_histogram():
    """Build a histogram via atomic_add — multiple threads write to same ndarray."""
    n = 1024
    num_bins = 8

    @qd.kernel
    def histogram(data: qd.types.NDArray, bins: qd.types.NDArray) -> None:
        for i in range(n):
            b = data[i] % num_bins
            qd.atomic_add(bins[b], 1)

    data = np.arange(n, dtype=np.int32)
    bins = np.zeros(num_bins, dtype=np.int32)
    histogram(data, bins)
    qd.sync()
    expected = np.array([n // num_bins] * num_bins, dtype=np.int32)
    np.testing.assert_array_equal(bins, expected)
