"""Tests for physical storage buffer pointer correctness on Metal.

Regression tests for a Metal shader compiler bug where stores through
physical GPU pointers (reinterpret_cast<device T*>(ulong + offset)) are
silently dropped when the byte offset involves a runtime stride multiply
and the stored value is row-uniform.

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
