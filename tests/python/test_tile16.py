import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.simt.tile16 import _TILE, _make_tile16x16

from tests import test_utils

_QD_DTYPES = [qd.f32, qd.f64]
_NP_DTYPES = {qd.f32: np.float32, qd.f64: np.float64}


def _skip_if_f64_unsupported(dtype):
    if dtype != qd.f64:
        return
    arch = qd.lang.impl.current_cfg().arch
    if arch == qd.metal:
        pytest.skip("Metal does not support f64")
    if arch == qd.vulkan:
        pytest.skip("Vulkan does not reliably support f64")


@test_utils.test(arch=qd.gpu)
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_zeros(qd_dtype):
    _skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))
    dst.from_numpy(np.ones((_TILE, _TILE), dtype=np_dtype))

    @qd.kernel
    def k1(dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    k1(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((_TILE, _TILE), dtype=np_dtype))


@test_utils.test(arch=qd.gpu)
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_eye(qd_dtype):
    _skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel
    def k1(dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile.eye()
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    k1(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(_TILE, dtype=np_dtype))


@test_utils.test(arch=qd.gpu)
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_eye_inplace(qd_dtype):
    _skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (_TILE, _TILE))
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, 0, _TILE, _TILE)
            t._eye_()
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 100.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(_TILE, dtype=np_dtype))


@test_utils.test(arch=qd.gpu)
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@pytest.mark.parametrize(
    "src_row, src_col, dst_row_dx, dst_col_dx, ncols, nrows",
    [
        (0, 0, 0, 0, _TILE, _TILE),
        (3, 7, 0, 0, _TILE, _TILE),
        (0, 0, 5, 10, _TILE, _TILE),
        (16, 32, 4, 8, _TILE, _TILE),
        (0, 0, 0, 0, 5, _TILE),
        (0, 0, 0, 0, _TILE, 10),
        (8, 4, 2, 6, 7, 12),
        (60, 60, 0, 0, 16, 16),
    ],
)
def test_tile16_load_store(qd_dtype, src_row, src_col, dst_row_dx, dst_col_dx, ncols, nrows):
    _skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 92
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (GRID, GRID))
    dst = qd.ndarray(qd_dtype, (GRID, GRID))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, src_row, src_col, ncols, nrows)
            t._store(dst_arr, src_row + dst_row_dx, src_col + dst_col_dx, ncols, nrows)

    data = np.arange(GRID * GRID, dtype=np_dtype).reshape(GRID, GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((GRID, GRID), -1.0, dtype=np_dtype))
    k1(src, dst)

    result = dst.to_numpy()
    dr, dc = src_row + dst_row_dx, src_col + dst_col_dx
    expected = np.full((GRID, GRID), -1.0, dtype=np_dtype)
    expected[dr : dr + nrows, dc : dc + ncols] = data[src_row : src_row + nrows, src_col : src_col + ncols]
    np.testing.assert_allclose(result, expected)


@test_utils.test(arch=qd.gpu)
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@pytest.mark.parametrize(
    "batch, src_row, src_col, ncols, nrows",
    [
        (0, 0, 0, _TILE, _TILE),
        (2, 0, 0, _TILE, _TILE),
        (5, 0, 0, _TILE, _TILE),
        (1, 8, 4, _TILE, _TILE),
        (3, 0, 0, 7, 10),
        (0, 16, 16, 12, 8),
    ],
)
def test_tile16_load3d_store3d(qd_dtype, batch, src_row, src_col, ncols, nrows):
    _skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 92
    NBATCH = 6
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (NBATCH, GRID, GRID))
    dst = qd.ndarray(qd_dtype, (NBATCH, GRID, GRID))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 3], dst_arr: qd.types.NDArray[qd_dtype, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load3d(src_arr, batch, src_row, src_col, ncols, nrows)
            t._store3d(dst_arr, batch, src_row, src_col, ncols, nrows)

    data = np.arange(NBATCH * GRID * GRID, dtype=np_dtype).reshape(NBATCH, GRID, GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((NBATCH, GRID, GRID), -1.0, dtype=np_dtype))
    k1(src, dst)

    result = dst.to_numpy()
    expected = np.full((NBATCH, GRID, GRID), -1.0, dtype=np_dtype)
    expected[batch, src_row : src_row + nrows, src_col : src_col + ncols] = data[
        batch, src_row : src_row + nrows, src_col : src_col + ncols
    ]
    np.testing.assert_allclose(result, expected)
