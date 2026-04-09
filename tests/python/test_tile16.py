import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.simt.tile16 import _make_tile16x16

from tests import test_utils

_TILE = 16


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


@test_utils.test(arch=qd.gpu)
def test_tile16_load_store_roundtrip():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 2], dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, 0, _TILE, _TILE)
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=qd.gpu)
def test_tile16_load_store_partial_cols():
    """Load/store only 5 of 16 columns."""
    Tile = _make_tile16x16(qd.f32)
    NCOLS = 5
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 2], dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, 0, NCOLS, _TILE)
            t._store(dst_arr, 0, 0, NCOLS, _TILE)

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_TILE, _TILE), -1.0, dtype=np.float32))
    k1(src, dst)

    result = dst.to_numpy()
    expected = np.full((_TILE, _TILE), -1.0, dtype=np.float32)
    expected[:, :NCOLS] = data[:, :NCOLS]
    np.testing.assert_allclose(result, expected)


@test_utils.test(arch=qd.gpu)
def test_tile16_load_store_partial_rows():
    """Load/store only 10 of 16 rows."""
    Tile = _make_tile16x16(qd.f32)
    NROWS = 10
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 2], dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, 0, _TILE, NROWS)
            t._store(dst_arr, 0, 0, _TILE, NROWS)

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_TILE, _TILE), -1.0, dtype=np.float32))
    k1(src, dst)

    result = dst.to_numpy()
    expected = np.full((_TILE, _TILE), -1.0, dtype=np.float32)
    expected[:NROWS, :] = data[:NROWS, :]
    np.testing.assert_allclose(result, expected)


@test_utils.test(arch=qd.gpu)
def test_tile16_load3d_store3d():
    Tile = _make_tile16x16(qd.f32)
    NBATCH = 4
    src = qd.ndarray(qd.f32, (NBATCH, _TILE, _TILE))
    dst = qd.ndarray(qd.f32, (NBATCH, _TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 3], dst_arr: qd.types.NDArray[qd.f32, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            for b in range(NBATCH):
                t = Tile()
                t._load3d(src_arr, b, 0, 0, _TILE, _TILE)
                t._store3d(dst_arr, b, 0, 0, _TILE, _TILE)

    data = np.arange(NBATCH * _TILE * _TILE, dtype=np.float32).reshape(NBATCH, _TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)
