import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.simt._tile16 import _TILE, _make_tile16x16

from tests import test_utils

_QD_DTYPES = [qd.f32, qd.f64]
_NP_DTYPES = {qd.f32: np.float32, qd.f64: np.float64}


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_zeros(qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
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


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_eye(qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
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


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_eye_inplace(qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
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


@pytest.mark.parametrize(
    "src_row, src_col, row_offset, col_offset, ncols, nrows",
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
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load_store(qd_dtype, src_row, src_col, row_offset, col_offset, ncols, nrows):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 92
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (GRID, GRID))
    dst = qd.ndarray(qd_dtype, (GRID, GRID))

    src_col_end = src_col + ncols
    src_row_end = src_row + nrows
    dst_row = src_row + row_offset
    dst_col = src_col + col_offset
    dst_col_end = dst_col + ncols
    dst_row_end = dst_row + nrows

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, src_row, src_col, src_col_end, src_row_end)
            t._store(dst_arr, dst_row, dst_col, dst_col_end, dst_row_end)

    data = np.arange(GRID * GRID, dtype=np_dtype).reshape(GRID, GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((GRID, GRID), -1.0, dtype=np_dtype))
    k1(src, dst)

    result = dst.to_numpy()
    expected = np.full((GRID, GRID), -1.0, dtype=np_dtype)
    expected[dst_row : dst_row + nrows, dst_col : dst_col + ncols] = data[
        src_row : src_row + nrows, src_col : src_col + ncols
    ]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load_clamp_to_array_rows(qd_dtype):
    """Load from an array shorter than 16 rows. Rows beyond arr height should be zero."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (NROWS, _TILE))
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, 0, _TILE, _TILE)
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    data = np.arange(NROWS * _TILE, dtype=np_dtype).reshape(NROWS, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:NROWS, :], data)
    np.testing.assert_allclose(result[NROWS:, :], 0.0)


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_store_clamp_to_array_rows(qd_dtype):
    """Store to an array shorter than 16 rows. Must not write out of bounds."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (_TILE, _TILE))
    dst = qd.ndarray(qd_dtype, (NROWS, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, 0, _TILE, _TILE)
            t._store(dst_arr, 0, 0, _TILE, _TILE)

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result, data[:NROWS, :])


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load3d_clamp_to_array_rows(qd_dtype):
    """Load from a 3D array shorter than 16 rows. Extra tile rows should be zero."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (1, NROWS, _TILE))
    dst = qd.ndarray(qd_dtype, (1, _TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 3], dst_arr: qd.types.NDArray[qd_dtype, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load3d(src_arr, 0, 0, 0, _TILE, _TILE)
            t._store3d(dst_arr, 0, 0, 0, _TILE, _TILE)

    data = np.arange(NROWS * _TILE, dtype=np_dtype).reshape(1, NROWS, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[0, :NROWS, :], data[0])
    np.testing.assert_allclose(result[0, NROWS:, :], 0.0)


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_store3d_clamp_to_array_rows(qd_dtype):
    """Store to a 3D array shorter than 16 rows. Must not write out of bounds."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (1, _TILE, _TILE))
    dst = qd.ndarray(qd_dtype, (1, NROWS, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 3], dst_arr: qd.types.NDArray[qd_dtype, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load3d(src_arr, 0, 0, 0, _TILE, _TILE)
            t._store3d(dst_arr, 0, 0, 0, _TILE, _TILE)

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(1, _TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[0], data[0, :NROWS, :])


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
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load3d_store3d(qd_dtype, batch, src_row, src_col, ncols, nrows):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 92
    NBATCH = 6
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (NBATCH, GRID, GRID))
    dst = qd.ndarray(qd_dtype, (NBATCH, GRID, GRID))

    col_end = src_col + ncols
    row_end = src_row + nrows

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 3], dst_arr: qd.types.NDArray[qd_dtype, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load3d(src_arr, batch, src_row, src_col, col_end, row_end)
            t._store3d(dst_arr, batch, src_row, src_col, col_end, row_end)

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
