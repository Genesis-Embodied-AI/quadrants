import numpy as np
import pytest
import scipy.linalg

import quadrants as qd
from quadrants.lang.exception import QuadrantsSyntaxError
from quadrants.lang.simt._tile16 import (
    _TILE,
    _make_tile16x16,
    _TileSliceProxy,
    _VecSliceProxy,
)

from tests import test_utils

_QD_DTYPES = [qd.f32, qd.f64]
_NP_DTYPES = {qd.f32: np.float32, qd.f64: np.float64}
_ATOLS = {qd.f32: 1e-5, qd.f64: 1e-10}
_EPS_VALS = {qd.f32: 1e-6, qd.f64: 1e-14}


def _ann(tensor_type, dtype, ndim):
    """Return the right kernel annotation for the given tensor_type."""
    if tensor_type == qd.ndarray:
        return qd.types.NDArray[dtype, ndim]
    return qd.Template


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("use_zeros_alias", [False, True])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_zeros(tensor_type, qd_dtype, use_zeros_alias):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    dst = tensor_type(qd_dtype, (_TILE, _TILE))
    dst.from_numpy(np.ones((_TILE, _TILE), dtype=np_dtype))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(dst_arr: Ann):
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


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_eye(tensor_type, qd_dtype, inplace):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    src = tensor_type(qd_dtype, (_TILE, _TILE))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            if qd.static(inplace):
                t = Tile()
                t._load(src_arr, 0, _TILE, 0, _TILE)
                t.eye_()
                t._store(dst_arr, 0, _TILE, 0, _TILE)
            else:
                t = Tile.eye()
                t._store(dst_arr, 0, _TILE, 0, _TILE)

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 100.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(_TILE, dtype=np_dtype))


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
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
def test_tile16_load_store(tensor_type, qd_dtype, src_row, src_col, row_offset, col_offset, ncols, nrows):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 92
    Tile = _make_tile16x16(qd_dtype)
    src = tensor_type(qd_dtype, (GRID, GRID))
    dst = tensor_type(qd_dtype, (GRID, GRID))

    src_col_end = src_col + ncols
    src_row_end = src_row + nrows
    dst_row = src_row + row_offset
    dst_col = src_col + col_offset
    dst_col_end = dst_col + ncols
    dst_row_end = dst_row + nrows

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
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


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("clamp_side", ["load", "store"])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_clamp_to_array_rows(tensor_type, qd_dtype, clamp_side):
    """Row clamping: load from short src (extra tile rows zero) or store to short dst (no OOB)."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src_rows = NROWS if clamp_side == "load" else _TILE
    dst_rows = _TILE if clamp_side == "load" else NROWS
    src = tensor_type(qd_dtype, (src_rows, _TILE))
    dst = tensor_type(qd_dtype, (dst_rows, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
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


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("clamp_side", ["load", "store"])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_3d_clamp_to_array_rows(tensor_type, qd_dtype, clamp_side):
    """3D row clamping: load from short src (extra tile rows zero) or store to short dst (no OOB)."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NROWS = 10
    Tile = _make_tile16x16(qd_dtype)
    src_rows = NROWS if clamp_side == "load" else _TILE
    dst_rows = _TILE if clamp_side == "load" else NROWS
    src = tensor_type(qd_dtype, (1, src_rows, _TILE))
    dst = tensor_type(qd_dtype, (1, dst_rows, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 3)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
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


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
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
def test_tile16_load3d_store3d(tensor_type, qd_dtype, batch, src_row, src_col, ncols, nrows):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 92
    NBATCH = 6
    Tile = _make_tile16x16(qd_dtype)
    src = tensor_type(qd_dtype, (NBATCH, GRID, GRID))
    dst = tensor_type(qd_dtype, (NBATCH, GRID, GRID))

    col_end = src_col + ncols
    row_end = src_row + nrows

    Ann = _ann(tensor_type, qd_dtype, 3)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
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


@test_utils.test(arch=qd.gpu)
def test_tile16_size_constant_in_kernel():
    """Tile.SIZE must be accessible inside a kernel without purity violations."""
    Tile = _make_tile16x16(qd.f32)
    out = qd.ndarray(qd.i32, (1,))

    @qd.kernel
    def k1(result: qd.types.NDArray[qd.i32, 1]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            result[0] = Tile.SIZE

    k1(out)
    assert out.to_numpy()[0] == 16


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load_clamp_to_array_cols(tensor_type, qd_dtype):
    """Load from an array narrower than 16 columns. Columns beyond arr width should be zero."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = tensor_type(qd_dtype, (_TILE, NCOLS))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
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


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_store_partial_cols_untouched(tensor_type, qd_dtype):
    """Load full 16 columns, store only NCOLS. Remaining dst columns must be untouched."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 10
    Tile = _make_tile16x16(qd_dtype)
    src = tensor_type(qd_dtype, (_TILE, _TILE))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
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


def _make_spd(np_dtype=np.float32, seed: int = 42):
    """Return a well-conditioned 16x16 symmetric positive-definite matrix."""
    rng = np.random.RandomState(seed)
    B = rng.randn(_TILE, _TILE).astype(np.float64)
    return (B @ B.T + _TILE * np.eye(_TILE)).astype(np_dtype)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_ger_sub(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    mat = tensor_type(qd_dtype, (_TILE, _TILE))
    vec_a = tensor_type(qd_dtype, (_TILE,))
    vec_b = tensor_type(qd_dtype, (_TILE,))
    out = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann2 = _ann(tensor_type, qd_dtype, 2)
    Ann1 = _ann(tensor_type, qd_dtype, 1)

    @qd.kernel
    def k1(
        mat_arr: Ann2,
        a_arr: Ann1,
        b_arr: Ann1,
        out_arr: Ann2,
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

    M = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np_dtype) + 1.0
    b = np.arange(_TILE, dtype=np_dtype) + 2.0
    mat.from_numpy(M)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    k1(mat, vec_a, vec_b, out)

    expected = M - np.outer(a, b)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=_ATOLS[qd_dtype])


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("dst_delta", [0, 3, 16])
@pytest.mark.parametrize("src_offset", [0, 5, 32])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_cholesky(tensor_type, qd_dtype, src_offset, dst_delta):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 64
    Tile = _make_tile16x16(qd_dtype)
    src = tensor_type(qd_dtype, (GRID, GRID))
    dst = tensor_type(qd_dtype, (GRID, GRID))

    dst_offset = src_offset + dst_delta
    src_row_end = src_offset + _TILE
    dst_row_end = dst_offset + _TILE

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t._load(src_arr, src_offset, src_row_end, src_offset, src_row_end)
            if qd.static(qd_dtype == qd.f64):
                t.cholesky_(qd.f64(1e-12))
            else:
                t.cholesky_(qd.f32(1e-6))
            t._store(dst_arr, dst_offset, dst_row_end, dst_offset, dst_row_end)

    A = _make_spd(np_dtype)
    src_np = np.zeros((GRID, GRID), dtype=np_dtype)
    src_np[src_offset : src_offset + _TILE, src_offset : src_offset + _TILE] = A
    src.from_numpy(src_np)
    dst.from_numpy(np.full((GRID, GRID), -1.0, dtype=np_dtype))
    k1(src, dst)

    result = dst.to_numpy()
    L_gpu = np.tril(result[dst_offset : dst_offset + _TILE, dst_offset : dst_offset + _TILE])
    L_ref = scipy.linalg.cholesky(A.astype(np.float64), lower=True).astype(np_dtype)
    atol = _ATOLS[qd_dtype]
    np.testing.assert_allclose(L_gpu, L_ref, atol=atol)
    untouched = np.full((GRID, GRID), -1.0, dtype=np_dtype)
    untouched[dst_offset : dst_offset + _TILE, dst_offset : dst_offset + _TILE] = result[
        dst_offset : dst_offset + _TILE, dst_offset : dst_offset + _TILE
    ]
    np.testing.assert_allclose(result, untouched)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_trsm(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    a_arr = tensor_type(qd_dtype, (_TILE, _TILE))
    b_arr = tensor_type(qd_dtype, (_TILE, _TILE))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(
        a_in: Ann,
        b_in: Ann,
        out: Ann,
    ):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            L = Tile()
            L._load(a_in, 0, _TILE, 0, _TILE)
            B = Tile()
            B._load(b_in, 0, _TILE, 0, _TILE)
            L.solve_triangular_(B)
            B._store(out, 0, _TILE, 0, _TILE)

    A = _make_spd(np_dtype)
    L_ref = scipy.linalg.cholesky(A.astype(np.float64), lower=True).astype(np_dtype)
    B = np.random.RandomState(123).randn(_TILE, _TILE).astype(np_dtype)

    a_arr.from_numpy(L_ref)
    b_arr.from_numpy(B)
    k1(a_arr, b_arr, dst)

    X_ref = scipy.linalg.solve_triangular(L_ref.astype(np.float64), B.astype(np.float64).T, lower=True).T.astype(
        np_dtype
    )
    trsm_atol = {qd.f32: 1e-4, qd.f64: 1e-10}
    np.testing.assert_allclose(dst.to_numpy(), X_ref, atol=trsm_atol[qd_dtype])


@test_utils.test(arch=qd.gpu)
def test_tile16_solve_triangular_upper_raises():
    Tile = _make_tile16x16(qd.f32)
    with pytest.raises(TypeError, match="only lower=True"):
        Tile().solve_triangular_(Tile(), lower=False)


# =============================================================================
# Slice-syntax tests (PR 3: arr[r0:r1, c0:c1])
# =============================================================================


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_slice_load_store_roundtrip(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    src = tensor_type(qd_dtype, (_TILE, _TILE))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = src_arr[0:_TILE, 0:_TILE]
            dst_arr[0:_TILE, 0:_TILE] = t

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_slice_partial_cols(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    NCOLS = 7
    src = tensor_type(qd_dtype, (_TILE, _TILE))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = src_arr[0:_TILE, 0:NCOLS]
            dst_arr[0:_TILE, 0:NCOLS] = t

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_TILE, _TILE), -1.0, dtype=np_dtype))
    k1(src, dst)

    result = dst.to_numpy()
    expected = np.full((_TILE, _TILE), -1.0, dtype=np_dtype)
    expected[:, :NCOLS] = data[:, :NCOLS]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_slice_3d_batch(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    NBATCH = 3
    src = tensor_type(qd_dtype, (NBATCH, _TILE, _TILE))
    dst = tensor_type(qd_dtype, (NBATCH, _TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 3)

    @qd.kernel
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            for b in range(NBATCH):
                t = Tile()
                t[:] = src_arr[b, 0:_TILE, 0:_TILE]
                dst_arr[b, 0:_TILE, 0:_TILE] = t

    data = np.arange(NBATCH * _TILE * _TILE, dtype=np_dtype).reshape(NBATCH, _TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_slice_ger_sub_via_outer(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    mat = tensor_type(qd_dtype, (_TILE, _TILE))
    vec_a = tensor_type(qd_dtype, (_TILE,))
    vec_b = tensor_type(qd_dtype, (_TILE,))
    out = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann2 = _ann(tensor_type, qd_dtype, 2)
    Ann1 = _ann(tensor_type, qd_dtype, 1)

    @qd.kernel
    def k1(
        mat_arr: Ann2,
        a_arr: Ann1,
        b_arr: Ann1,
        out_arr: Ann2,
    ):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = mat_arr[0:_TILE, 0:_TILE]
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            a_val = a_arr[tid]
            b_val = b_arr[tid]
            t -= qd.outer(a_val, b_val)
            out_arr[0:_TILE, 0:_TILE] = t

    M = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np_dtype) + 1.0
    b = np.arange(_TILE, dtype=np_dtype) + 2.0
    mat.from_numpy(M)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    k1(mat, vec_a, vec_b, out)

    expected = M - np.outer(a, b)
    atol = _ATOLS[qd_dtype]
    np.testing.assert_allclose(out.to_numpy(), expected, atol=atol)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_ger_sub_2d(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    mat = tensor_type(qd_dtype, (_TILE, _TILE))
    vecs = tensor_type(qd_dtype, (_TILE, 2))
    out = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def k1(
        mat_arr: Ann,
        vecs_arr: Ann,
        out_arr: Ann,
    ):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = mat_arr[0:_TILE, 0:_TILE]
            t -= qd.outer(vecs_arr[0:_TILE, 0], vecs_arr[0:_TILE, 1])
            out_arr[0:_TILE, 0:_TILE] = t

    M = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np_dtype) + 1.0
    b = np.arange(_TILE, dtype=np_dtype) + 2.0
    mat.from_numpy(M)
    vecs.from_numpy(np.column_stack([a, b]))
    k1(mat, vecs, out)

    expected = M - np.outer(a, b)
    atol = _ATOLS[qd_dtype]
    np.testing.assert_allclose(out.to_numpy(), expected, atol=atol)


@test_utils.test(arch=qd.gpu)
def test_tile16_outer_symmetric_same_variable():
    """t -= qd.outer(v, v) with the same variable for both args."""
    Tile = _make_tile16x16(qd.f32)
    mat = qd.ndarray(qd.f32, (_TILE, _TILE))
    vecs = qd.ndarray(qd.f32, (_TILE, 1))
    out = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(
        mat_arr: qd.types.NDArray[qd.f32, 2],
        vecs_arr: qd.types.NDArray[qd.f32, 2],
        out_arr: qd.types.NDArray[qd.f32, 2],
    ):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = mat_arr[0:_TILE, 0:_TILE]
            v = vecs_arr[0:_TILE, 0]
            t -= qd.outer(v, v)
            out_arr[0:_TILE, 0:_TILE] = t

    M = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np.float32) + 1.0
    mat.from_numpy(M)
    vecs.from_numpy(a.reshape(-1, 1))
    k1(mat, vecs, out)

    expected = M - np.outer(a, a)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=_ATOLS[qd.f32])


@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_ger_sub_3d():
    """Column vector load from a 3D array: v = arr[batch, r0:r1, col]."""
    Tile = _make_tile16x16(qd.f32)
    NBATCH = 2
    mat = qd.ndarray(qd.f32, (_TILE, _TILE))
    vecs = qd.ndarray(qd.f32, (NBATCH, _TILE, 2))
    out = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(
        mat_arr: qd.types.NDArray[qd.f32, 2],
        vecs_arr: qd.types.NDArray[qd.f32, 3],
        out_arr: qd.types.NDArray[qd.f32, 2],
    ):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = mat_arr[0:_TILE, 0:_TILE]
            a = vecs_arr[1, 0:_TILE, 0]
            b = vecs_arr[1, 0:_TILE, 1]
            t -= qd.outer(a, b)
            out_arr[0:_TILE, 0:_TILE] = t

    M = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np.float32) + 1.0
    b = np.arange(_TILE, dtype=np.float32) + 2.0
    vecs_np = np.zeros((NBATCH, _TILE, 2), dtype=np.float32)
    vecs_np[1, :, 0] = a
    vecs_np[1, :, 1] = b
    mat.from_numpy(M)
    vecs.from_numpy(vecs_np)
    k1(mat, vecs, out)

    expected = M - np.outer(a, b)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=_ATOLS[qd.f32])


# =============================================================================
# Error-raising tests
# =============================================================================


def test_outer_composition_raises():
    """qd.outer(a, b) + qd.outer(c, d) must raise TypeError."""
    p1 = qd.outer(1, 2)
    p2 = qd.outer(3, 4)
    with pytest.raises(TypeError, match="does not support composition"):
        _ = p1 + p2
    with pytest.raises(TypeError, match="does not support composition"):
        _ = p2 + p1


def test_tile_slice_proxy_misuse_errors():
    """Accidentally using a tile slice proxy as a value gives a clear error."""
    tile_proxy = _TileSliceProxy(None, 0, 16, 0, 16)
    vec_proxy = _VecSliceProxy(None, 0, 16, 0)

    with pytest.raises(TypeError, match="only valid in tile operations"):
        _ = tile_proxy + 1
    with pytest.raises(TypeError, match="only valid in tile operations"):
        _ = 1 + tile_proxy
    with pytest.raises(TypeError, match="only valid in tile operations"):
        _ = tile_proxy - 1
    with pytest.raises(TypeError, match="only valid in tile operations"):
        _ = tile_proxy * 2
    with pytest.raises(TypeError, match="only valid in tile operations"):
        _ = tile_proxy[0]

    with pytest.raises(TypeError, match="only valid in tile operations"):
        _ = vec_proxy + 1
    with pytest.raises(TypeError, match="only valid in tile operations"):
        _ = vec_proxy[0]

    assert "not a value" in repr(tile_proxy)
    assert "not a value" in repr(vec_proxy)


@test_utils.test(arch=qd.gpu)
def test_tile16_load_negative_row_raises():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[-1:_TILE, 0:_TILE]
            d[0:_TILE, 0:_TILE] = t

    with pytest.raises(QuadrantsSyntaxError, match="Negative indices"):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_load_negative_col_raises():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[0:_TILE, -1:_TILE]
            d[0:_TILE, 0:_TILE] = t

    with pytest.raises(QuadrantsSyntaxError, match="Negative indices"):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_load_missing_start_raises():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[:_TILE, 0:_TILE]
            d[0:_TILE, 0:_TILE] = t

    with pytest.raises(QuadrantsSyntaxError, match="start and stop indices are required"):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_load_missing_stop_raises():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[0:, 0:_TILE]
            d[0:_TILE, 0:_TILE] = t

    with pytest.raises(QuadrantsSyntaxError, match="start and stop indices are required"):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_store_missing_stop_raises():
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[0:_TILE, 0:_TILE]
            d[0:, 0:_TILE] = t

    with pytest.raises(QuadrantsSyntaxError, match="start and stop indices are required"):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_slice_wrong_index_order_raises():
    """arr[r:r2, col, batch] must be rejected (batch must come first)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (3, _TILE, 2))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 3]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            v = s[0:_TILE, 0, 1]
            t -= qd.outer(v, v)

    with pytest.raises(Exception):
        k1(src)


@test_utils.test(arch=qd.gpu)
def test_tile16_slice_extra_indices_raises():
    """arr[a, b, r:r2, c:c2] must be rejected (too many non-slice indices)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[0, 0, 0:_TILE, 0:_TILE]
            d[0:_TILE, 0:_TILE] = t

    with pytest.raises(Exception):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_outer_product_intermediate_variable():
    """qd.outer(a, b) assigned to a variable before -= must work."""
    Tile = _make_tile16x16(qd.f32)
    mat = qd.ndarray(qd.f32, (_TILE, _TILE))
    out = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(
        mat_arr: qd.types.NDArray[qd.f32, 2],
        out_arr: qd.types.NDArray[qd.f32, 2],
    ):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = mat_arr[0:_TILE, 0:_TILE]
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            a_val = qd.f32(tid + 1)
            b_val = qd.f32(tid + 2)
            op = qd.outer(a_val, b_val)
            t -= op
            out_arr[0:_TILE, 0:_TILE] = t

    M = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np.float32) + 1.0
    b = np.arange(_TILE, dtype=np.float32) + 2.0
    mat.from_numpy(M)
    k1(mat, out)

    expected = M - np.outer(a, b)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=_ATOLS[qd.f32])


@test_utils.test(arch=qd.gpu)
def test_tile16_load_without_slice_rebinds():
    """Omitting [:] on the LHS rebinds the variable to a proxy, not a tile."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t = s[0:_TILE, 0:_TILE]
            d[0:_TILE, 0:_TILE] = t

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    with pytest.raises(Exception):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_augassign_add_outer_raises():
    """t += qd.outer(a, b) must raise TypeError (only -= is supported)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[0:_TILE, 0:_TILE]
            tid = qd.f32(qd.simt.subgroup.invocation_id())
            t += qd.outer(tid, tid)

    with pytest.raises(TypeError, match="unsupported augmented assignment op"):
        k1(src)


@test_utils.test(arch=qd.gpu)
def test_tile16_augassign_non_outer_raises():
    """t -= <scalar> must raise TypeError (only outer products allowed)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            t[:] = s[0:_TILE, 0:_TILE]
            t -= qd.f32(1.0)

    with pytest.raises(TypeError, match="unsupported augmented assignment"):
        k1(src)


@test_utils.test(arch=qd.gpu)
def test_tile16_vec_slice_missing_stop_raises():
    """arr[0:, col] must be rejected (vec slice missing stop)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, 2))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            v = s[0:, 0]
            t -= qd.outer(v, v)
            d[0:_TILE, 0:_TILE] = t

    with pytest.raises(QuadrantsSyntaxError, match="both start and stop"):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_vec_slice_missing_start_raises():
    """arr[:16, col] must be rejected (vec slice missing start)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, 2))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=_TILE)
        for _ in range(_TILE):
            t = Tile()
            v = s[:_TILE, 0]
            t -= qd.outer(v, v)
            d[0:_TILE, 0:_TILE] = t

    with pytest.raises(QuadrantsSyntaxError, match="both start and stop"):
        k1(src, dst)
