import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.simt.tile16 import _TILE, _make_tile16x16

from tests import test_utils


@test_utils.test(arch=qd.gpu)
def test_tile16_zeros():
    Tile = _make_tile16x16(qd.f32)
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst.from_numpy(np.ones((_TILE, _TILE), dtype=np.float32))

    @qd.kernel
    def k1(dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    k1(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((_TILE, _TILE), dtype=np.float32))


@test_utils.test(arch=qd.gpu)
def test_tile16_eye():
    Tile = _make_tile16x16(qd.f32)
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile.eye()
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    k1(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(_TILE, dtype=np.float32))


@test_utils.test(arch=qd.gpu)
def test_tile16_eye_inplace():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 2], dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, 0, _TILE, _TILE)
            t.eye_()
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 100.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(_TILE, dtype=np.float32))


_GRID = 92


@test_utils.test(arch=qd.gpu)
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
def test_tile16_load_store(src_row, src_col, dst_row_dx, dst_col_dx, ncols, nrows):
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_GRID, _GRID))
    dst = qd.ndarray(qd.f32, (_GRID, _GRID))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 2], dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, src_row, src_col, ncols, nrows)
            t._store(dst_arr, src_row + dst_row_dx, src_col + dst_col_dx, ncols, nrows)

    data = np.arange(_GRID * _GRID, dtype=np.float32).reshape(_GRID, _GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_GRID, _GRID), -1.0, dtype=np.float32))
    k1(src, dst)

    result = dst.to_numpy()
    dr, dc = src_row + dst_row_dx, src_col + dst_col_dx
    expected = np.full((_GRID, _GRID), -1.0, dtype=np.float32)
    expected[dr : dr + nrows, dc : dc + ncols] = data[src_row : src_row + nrows, src_col : src_col + ncols]
    np.testing.assert_allclose(result, expected)


_NBATCH = 6


@test_utils.test(arch=qd.gpu)
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
def test_tile16_load3d_store3d(batch, src_row, src_col, ncols, nrows):
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_NBATCH, _GRID, _GRID))
    dst = qd.ndarray(qd.f32, (_NBATCH, _GRID, _GRID))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 3], dst_arr: qd.types.NDArray[qd.f32, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load3d(src_arr, batch, src_row, src_col, ncols, nrows)
            t._store3d(dst_arr, batch, src_row, src_col, ncols, nrows)

    data = np.arange(_NBATCH * _GRID * _GRID, dtype=np.float32).reshape(_NBATCH, _GRID, _GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_NBATCH, _GRID, _GRID), -1.0, dtype=np.float32))
    k1(src, dst)

    result = dst.to_numpy()
    expected = np.full((_NBATCH, _GRID, _GRID), -1.0, dtype=np.float32)
    expected[batch, src_row : src_row + nrows, src_col : src_col + ncols] = data[batch, src_row : src_row + nrows, src_col : src_col + ncols]
    np.testing.assert_allclose(result, expected)
