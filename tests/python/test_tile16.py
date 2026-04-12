import numpy as np
import pytest
import scipy.linalg

import quadrants as qd
from quadrants.lang.simt._tile16 import _TILE, _make_tile16x16

from tests import test_utils

_QD_DTYPES = [qd.f32, qd.f64]
_NP_DTYPES = {qd.f32: np.float32, qd.f64: np.float64}


@pytest.mark.parametrize("use_zeros_alias", [False, True])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_zeros(qd_dtype, use_zeros_alias):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))
    dst.from_numpy(np.ones((_TILE, _TILE), dtype=np_dtype))

    @qd.kernel
    def k1(dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            if qd.static(use_zeros_alias):
                t = Tile.zeros()
                t._store(dst_arr, 0, _TILE, 0, _TILE)
            else:
                t = Tile()
                t._store(dst_arr, 0, _TILE, 0, _TILE)

    k1(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((_TILE, _TILE), dtype=np_dtype))


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_eye(qd_dtype, inplace):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (_TILE, _TILE))
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            if qd.static(inplace):
                t = Tile()
                t._load(src_arr, 0, _TILE, 0, _TILE)
                t._eye_()
                t._store(dst_arr, 0, _TILE, 0, _TILE)
            else:
                t = Tile.eye()
                t._store(dst_arr, 0, _TILE, 0, _TILE)

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
            t._load(src_arr, src_row, src_row_end, src_col, src_col_end)
            t._store(dst_arr, dst_row, dst_row_end, dst_col, dst_col_end)

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


@pytest.mark.parametrize("clamp_side", ["load", "store"])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_clamp_to_array_rows(qd_dtype, clamp_side):
    """Row clamping: load from short src (extra tile rows zero) or store to short dst (no OOB)."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src_rows = NROWS if clamp_side == "load" else _TILE
    dst_rows = _TILE if clamp_side == "load" else NROWS
    src = qd.ndarray(qd_dtype, (src_rows, _TILE))
    dst = qd.ndarray(qd_dtype, (dst_rows, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, _TILE, 0, _TILE)
            t._store(dst_arr, 0, _TILE, 0, _TILE)

    data = np.arange(src_rows * _TILE, dtype=np_dtype).reshape(src_rows, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    result = dst.to_numpy()
    if clamp_side == "load":
        np.testing.assert_allclose(result[:NROWS, :], data)
        np.testing.assert_allclose(result[NROWS:, :], 0.0)
    else:
        np.testing.assert_allclose(result, data[:NROWS, :])


@pytest.mark.parametrize("clamp_side", ["load", "store"])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_3d_clamp_to_array_rows(qd_dtype, clamp_side):
    """3D row clamping: load from short src (extra tile rows zero) or store to short dst (no OOB)."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src_rows = NROWS if clamp_side == "load" else _TILE
    dst_rows = _TILE if clamp_side == "load" else NROWS
    src = qd.ndarray(qd_dtype, (1, src_rows, _TILE))
    dst = qd.ndarray(qd_dtype, (1, dst_rows, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 3], dst_arr: qd.types.NDArray[qd_dtype, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load3d(src_arr, 0, 0, _TILE, 0, _TILE)
            t._store3d(dst_arr, 0, 0, _TILE, 0, _TILE)

    data = np.arange(src_rows * _TILE, dtype=np_dtype).reshape(1, src_rows, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    result = dst.to_numpy()
    if clamp_side == "load":
        np.testing.assert_allclose(result[0, :NROWS, :], data[0])
        np.testing.assert_allclose(result[0, NROWS:, :], 0.0)
    else:
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
            t._load3d(src_arr, batch, src_row, row_end, src_col, col_end)
            t._store3d(dst_arr, batch, src_row, row_end, src_col, col_end)

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


def test_tile16_size_constant():
    Tile = _make_tile16x16(qd.f32)
    assert Tile.SIZE == 16


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load_clamp_to_array_cols(qd_dtype):
    """Load from an array narrower than 16 columns. Columns beyond arr width should be zero."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (_TILE, NCOLS))
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, _TILE, 0, _TILE)
            t._store(dst_arr, 0, _TILE, 0, _TILE)

    data = np.arange(_TILE * NCOLS, dtype=np_dtype).reshape(_TILE, NCOLS) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data)
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_store_partial_cols_untouched(qd_dtype):
    """Load full 16 columns, store only NCOLS. Remaining dst columns must be untouched."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = qd.ndarray(qd_dtype, (_TILE, _TILE))
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, _TILE, 0, _TILE)
            t._store(dst_arr, 0, _TILE, 0, NCOLS)

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_TILE, _TILE), -1.0, dtype=np_dtype))
    k1(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], -1.0)


def test_tile16_make_caching():
    """_make_tile16x16 must return the same object for the same dtype."""
    a = _make_tile16x16(qd.f32)
    b = _make_tile16x16(qd.f32)
    assert a is b
    c = _make_tile16x16(qd.f64)
    assert a is not c
    d = _make_tile16x16(qd.f64)
    assert c is d


def _make_spd(seed: int = 42):
    """Return a well-conditioned 16x16 symmetric positive-definite matrix."""
    rng = np.random.RandomState(seed)
    B = rng.randn(_TILE, _TILE).astype(np.float64)
    return (B @ B.T + _TILE * np.eye(_TILE)).astype(np.float32)


@test_utils.test(arch=qd.gpu)
def test_tile16_ger_sub():
    Tile = _make_tile16x16(qd.f32)
    mat = qd.ndarray(qd.f32, (_TILE, _TILE))
    vec_a = qd.ndarray(qd.f32, (_TILE,))
    vec_b = qd.ndarray(qd.f32, (_TILE,))
    out = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(
        mat_arr: qd.types.NDArray[qd.f32, 2],
        a_arr: qd.types.NDArray[qd.f32, 1],
        b_arr: qd.types.NDArray[qd.f32, 1],
        out_arr: qd.types.NDArray[qd.f32, 2],
    ):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(mat_arr, 0, _TILE, 0, _TILE)
            tid = qd.simt.subgroup.invocation_id()
            a_val = a_arr[tid]
            b_val = b_arr[tid]
            t._ger_sub(a_val, b_val)
            t._store(out_arr, 0, _TILE, 0, _TILE)

    M = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np.float32) + 1.0
    b = np.arange(_TILE, dtype=np.float32) + 2.0
    mat.from_numpy(M)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    k1(mat, vec_a, vec_b, out)

    expected = M - np.outer(a, b)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-5)


@test_utils.test(arch=qd.gpu)
def test_tile16_cholesky():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(src_arr: qd.types.NDArray[qd.f32, 2], dst_arr: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, 0, _TILE, 0, _TILE)
            t.cholesky_(qd.f32(1e-6))
            t._store(dst_arr, 0, _TILE, 0, _TILE)

    A = _make_spd()
    src.from_numpy(A)
    k1(src, dst)

    L_gpu = np.tril(dst.to_numpy())
    L_ref = scipy.linalg.cholesky(A.astype(np.float64), lower=True).astype(np.float32)
    np.testing.assert_allclose(L_gpu, L_ref, atol=1e-5)
