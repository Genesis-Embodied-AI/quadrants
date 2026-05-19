import subprocess
import sys
from pathlib import Path

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

    @qd.kernel(fastcache=True)
    def k1(dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            if qd.static(use_zeros_alias):
                t = Tile.zeros()
                t._store(dst_arr, 0, tile_size, 0, tile_size)
            else:
                t = Tile()
                t._store(dst_arr, 0, tile_size, 0, tile_size)

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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            if qd.static(inplace):
                t = Tile()
                t._load(src_arr, 0, tile_size, 0, tile_size)
                t.eye_()
                t._store(dst_arr, 0, tile_size, 0, tile_size)
            else:
                t = Tile.eye()
                t._store(dst_arr, 0, tile_size, 0, tile_size)

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
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
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

    @qd.kernel(fastcache=True)
    def k1(
        src_arr: Ann,
        dst_arr: Ann,
        src_row: qd.i32,
        src_row_end: qd.i32,
        src_col: qd.i32,
        src_col_end: qd.i32,
        dst_row: qd.i32,
        dst_row_end: qd.i32,
        dst_col: qd.i32,
        dst_col_end: qd.i32,
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t._load(src_arr, src_row, src_row_end, src_col, src_col_end)
            t._store(dst_arr, dst_row, dst_row_end, dst_col, dst_col_end)

    data = np.arange(GRID * GRID, dtype=np_dtype).reshape(GRID, GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((GRID, GRID), -1.0, dtype=np_dtype))
    k1(src, dst, src_row, src_row_end, src_col, src_col_end, dst_row, dst_row_end, dst_col, dst_col_end)

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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t._load(src_arr, 0, tile_size, 0, tile_size)
            t._store(dst_arr, 0, tile_size, 0, tile_size)

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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t._load3d(src_arr, 0, 0, tile_size, 0, tile_size)
            t._store3d(dst_arr, 0, 0, tile_size, 0, tile_size)

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
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
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

    @qd.kernel(fastcache=True)
    def k1(
        src_arr: Ann,
        dst_arr: Ann,
        batch: qd.i32,
        src_row: qd.i32,
        row_end: qd.i32,
        src_col: qd.i32,
        col_end: qd.i32,
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t._load3d(src_arr, batch, src_row, row_end, src_col, col_end)
            t._store3d(dst_arr, batch, src_row, row_end, src_col, col_end)

    data = np.arange(NBATCH * GRID * GRID, dtype=np_dtype).reshape(NBATCH, GRID, GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((NBATCH, GRID, GRID), -1.0, dtype=np_dtype))
    k1(src, dst, batch, src_row, row_end, src_col, col_end)

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

    @qd.kernel(fastcache=True)
    def k1(result: qd.types.NDArray[qd.i32, 1]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t._load(src_arr, 0, tile_size, 0, tile_size)
            t._store(dst_arr, 0, tile_size, 0, tile_size)

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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann, NCOLS: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t._load(src_arr, 0, tile_size, 0, tile_size)
            t._store(dst_arr, 0, tile_size, 0, NCOLS)

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_TILE, _TILE), -1.0, dtype=np_dtype))
    k1(src, dst, NCOLS)
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


@pytest.mark.slow
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

    @qd.kernel(fastcache=True)
    def k1(
        mat_arr: Ann2,
        a_arr: Ann1,
        b_arr: Ann1,
        out_arr: Ann2,
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t._load(mat_arr, 0, tile_size, 0, tile_size)
            tid = qd.simt.subgroup.invocation_id()
            a_val = a_arr[tid]
            b_val = b_arr[tid]
            t._ger_sub(a_val, b_val)
            t._store(out_arr, 0, tile_size, 0, tile_size)

    M = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np_dtype) + 1.0
    b = np.arange(_TILE, dtype=np_dtype) + 2.0
    mat.from_numpy(M)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    k1(mat, vec_a, vec_b, out)

    expected = M - np.outer(a, b)
    atol = 1e-10 if qd_dtype == qd.f64 else 1e-5
    np.testing.assert_allclose(out.to_numpy(), expected, atol=atol)


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

    @qd.kernel(fastcache=True)
    def k1(
        src_arr: Ann,
        dst_arr: Ann,
        src_offset: qd.i32,
        src_row_end: qd.i32,
        dst_offset: qd.i32,
        dst_row_end: qd.i32,
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
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
    k1(src, dst, src_offset, src_row_end, dst_offset, dst_row_end)

    result = dst.to_numpy()
    L_gpu = np.tril(result[dst_offset : dst_offset + _TILE, dst_offset : dst_offset + _TILE])
    L_ref = scipy.linalg.cholesky(A.astype(np.float64), lower=True).astype(np_dtype)
    atol = 1e-10 if qd_dtype == qd.f64 else 1e-5
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

    @qd.kernel(fastcache=True)
    def k1(
        a_in: Ann,
        b_in: Ann,
        out: Ann,
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            L = Tile()
            L._load(a_in, 0, tile_size, 0, tile_size)
            B = Tile()
            B._load(b_in, 0, tile_size, 0, tile_size)
            L.solve_triangular_(B)
            B._store(out, 0, tile_size, 0, tile_size)

    A = _make_spd(np_dtype)
    L_ref = scipy.linalg.cholesky(A.astype(np.float64), lower=True).astype(np_dtype)
    B = np.random.RandomState(123).randn(_TILE, _TILE).astype(np_dtype)

    a_arr.from_numpy(L_ref)
    b_arr.from_numpy(B)
    k1(a_arr, b_arr, dst)

    X_ref = scipy.linalg.solve_triangular(L_ref.astype(np.float64), B.astype(np.float64).T, lower=True).T.astype(
        np_dtype
    )
    atol = 1e-10 if qd_dtype == qd.f64 else 1e-4
    np.testing.assert_allclose(dst.to_numpy(), X_ref, atol=atol)


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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = src_arr[0:tile_size, 0:tile_size]
            dst_arr[0:tile_size, 0:tile_size] = t

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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann, NCOLS: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = src_arr[0:tile_size, 0:NCOLS]
            dst_arr[0:tile_size, 0:NCOLS] = t

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((_TILE, _TILE), -1.0, dtype=np_dtype))
    k1(src, dst, NCOLS)

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

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann, NBATCH: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            for b in range(NBATCH):
                t = Tile()
                t[:] = src_arr[b, 0:tile_size, 0:tile_size]
                dst_arr[b, 0:tile_size, 0:tile_size] = t

    data = np.arange(NBATCH * _TILE * _TILE, dtype=np_dtype).reshape(NBATCH, _TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst, NBATCH)
    np.testing.assert_allclose(dst.to_numpy(), data)


@pytest.mark.slow
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

    @qd.kernel(fastcache=True)
    def k1(
        mat_arr: Ann2,
        a_arr: Ann1,
        b_arr: Ann1,
        out_arr: Ann2,
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            a_val = a_arr[tid]
            b_val = b_arr[tid]
            t -= qd.outer(a_val, b_val)
            out_arr[0:tile_size, 0:tile_size] = t

    M = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np_dtype) + 1.0
    b = np.arange(_TILE, dtype=np_dtype) + 2.0
    mat.from_numpy(M)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    k1(mat, vec_a, vec_b, out)

    expected = M - np.outer(a, b)
    atol = 1e-10 if qd_dtype == qd.f64 else 1e-5
    np.testing.assert_allclose(out.to_numpy(), expected, atol=atol)


@pytest.mark.slow
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_ger_sub_2d(qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    Tile = _make_tile16x16(qd_dtype)
    mat = qd.ndarray(qd_dtype, (_TILE, _TILE))
    vecs = qd.ndarray(qd_dtype, (_TILE, 2))
    out = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(
        mat_arr: qd.types.NDArray[qd_dtype, 2],
        vecs_arr: qd.types.NDArray[qd_dtype, 2],
        out_arr: qd.types.NDArray[qd_dtype, 2],
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            t -= qd.outer(vecs_arr[0:tile_size, 0], vecs_arr[0:tile_size, 1])
            out_arr[0:tile_size, 0:tile_size] = t

    M = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np_dtype) + 1.0
    b = np.arange(_TILE, dtype=np_dtype) + 2.0
    mat.from_numpy(M)
    vecs.from_numpy(np.column_stack([a, b]))
    k1(mat, vecs, out)

    expected = M - np.outer(a, b)
    atol = 1e-10 if qd_dtype == qd.f64 else 1e-5
    np.testing.assert_allclose(out.to_numpy(), expected, atol=atol)


@pytest.mark.slow
@test_utils.test(arch=qd.gpu)
def test_tile16_outer_symmetric_same_variable():
    """t -= qd.outer(v, v) with the same variable for both args."""
    Tile = _make_tile16x16(qd.f32)
    mat = qd.ndarray(qd.f32, (_TILE, _TILE))
    vecs = qd.ndarray(qd.f32, (_TILE, 1))
    out = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(
        mat_arr: qd.types.NDArray[qd.f32, 2],
        vecs_arr: qd.types.NDArray[qd.f32, 2],
        out_arr: qd.types.NDArray[qd.f32, 2],
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            v = vecs_arr[0:tile_size, 0]
            t -= qd.outer(v, v)
            out_arr[0:tile_size, 0:tile_size] = t

    M = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np.float32) + 1.0
    mat.from_numpy(M)
    vecs.from_numpy(a.reshape(-1, 1))
    k1(mat, vecs, out)

    expected = M - np.outer(a, a)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-5)


@pytest.mark.slow
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_ger_sub_3d():
    """Column vector load from a 3D array: v = arr[batch, r0:r1, col]."""
    Tile = _make_tile16x16(qd.f32)
    NBATCH = 2
    mat = qd.ndarray(qd.f32, (_TILE, _TILE))
    vecs = qd.ndarray(qd.f32, (NBATCH, _TILE, 2))
    out = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(
        mat_arr: qd.types.NDArray[qd.f32, 2],
        vecs_arr: qd.types.NDArray[qd.f32, 3],
        out_arr: qd.types.NDArray[qd.f32, 2],
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            a = vecs_arr[1, 0:tile_size, 0]
            b = vecs_arr[1, 0:tile_size, 1]
            t -= qd.outer(a, b)
            out_arr[0:tile_size, 0:tile_size] = t

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
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-5)


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


@pytest.mark.parametrize(
    "bad_slice,match",
    [
        ("neg_row", "Negative indices"),
        ("neg_col", "Negative indices"),
        ("no_start", "start and stop indices are required"),
        ("no_stop", "start and stop indices are required"),
    ],
)
@test_utils.test(arch=qd.gpu)
def test_tile16_load_slice_errors(bad_slice, match):
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            if qd.static(bad_slice == "neg_row"):
                t[:] = s[-1:tile_size, 0:tile_size]
            elif qd.static(bad_slice == "neg_col"):
                t[:] = s[0:tile_size, -1:tile_size]
            elif qd.static(bad_slice == "no_start"):
                t[:] = s[:tile_size, 0:tile_size]
            elif qd.static(bad_slice == "no_stop"):
                t[:] = s[0:, 0:tile_size]
            d[0:tile_size, 0:tile_size] = t

    with pytest.raises(QuadrantsSyntaxError, match=match):
        k1(src, dst)


@pytest.mark.parametrize(
    "bad_slice,match",
    [
        ("neg_row", "Negative indices"),
        ("neg_col", "Negative indices"),
        ("no_start", "start and stop indices are required"),
        ("no_stop", "start and stop indices are required"),
    ],
)
@test_utils.test(arch=qd.gpu)
def test_tile16_store_slice_errors(bad_slice, match):
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = s[0:tile_size, 0:tile_size]
            if qd.static(bad_slice == "neg_row"):
                d[-1:tile_size, 0:tile_size] = t
            elif qd.static(bad_slice == "neg_col"):
                d[0:tile_size, -1:tile_size] = t
            elif qd.static(bad_slice == "no_start"):
                d[:tile_size, 0:tile_size] = t
            elif qd.static(bad_slice == "no_stop"):
                d[0:, 0:tile_size] = t

    with pytest.raises(QuadrantsSyntaxError, match=match):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_slice_wrong_index_order_raises():
    """arr[r:r2, col, batch] must be rejected (batch must come first)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (3, _TILE, 2))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 3]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            v = s[0:tile_size, 0, 1]
            t -= qd.outer(v, v)

    with pytest.raises(Exception):
        k1(src)


@test_utils.test(arch=qd.gpu)
def test_tile16_slice_extra_indices_raises():
    """arr[a, b, r:r2, c:c2] must be rejected (too many non-slice indices)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = s[0, 0, 0:tile_size, 0:tile_size]
            d[0:tile_size, 0:tile_size] = t

    with pytest.raises(Exception):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_outer_product_intermediate_variable():
    """qd.outer(a, b) assigned to a variable before -= must work."""
    Tile = _make_tile16x16(qd.f32)
    mat = qd.ndarray(qd.f32, (_TILE, _TILE))
    out = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(
        mat_arr: qd.types.NDArray[qd.f32, 2],
        out_arr: qd.types.NDArray[qd.f32, 2],
    ):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            a_val = qd.f32(tid + 1)
            b_val = qd.f32(tid + 2)
            op = qd.outer(a_val, b_val)
            t -= op
            out_arr[0:tile_size, 0:tile_size] = t

    M = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE)
    a = np.arange(_TILE, dtype=np.float32) + 1.0
    b = np.arange(_TILE, dtype=np.float32) + 2.0
    mat.from_numpy(M)
    k1(mat, out)

    expected = M - np.outer(a, b)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-5)


@test_utils.test(arch=qd.gpu)
def test_tile16_load_without_slice_rebinds():
    """Omitting [:] on the LHS rebinds the variable to a proxy, not a tile."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t = s[0:tile_size, 0:tile_size]
            d[0:tile_size, 0:tile_size] = t

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    with pytest.raises(Exception):
        k1(src, dst)


@test_utils.test(arch=qd.gpu)
def test_tile16_augassign_add_outer_raises():
    """t += qd.outer(a, b) must raise TypeError (only -= is supported)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = s[0:tile_size, 0:tile_size]
            tid = qd.f32(qd.simt.subgroup.invocation_id())
            t += qd.outer(tid, tid)

    with pytest.raises(TypeError, match="unsupported augmented assignment op"):
        k1(src)


@test_utils.test(arch=qd.gpu)
def test_tile16_augassign_non_outer_raises():
    """t -= <scalar> must raise TypeError (only outer products allowed)."""
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            t[:] = s[0:tile_size, 0:tile_size]
            t -= qd.f32(1.0)

    with pytest.raises(TypeError, match="unsupported augmented assignment"):
        k1(src)


@pytest.mark.parametrize("bad_slice", ["no_stop", "no_start"])
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_slice_errors(bad_slice):
    Tile = _make_tile16x16(qd.f32)
    src = qd.ndarray(qd.f32, (_TILE, 2))
    dst = qd.ndarray(qd.f32, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(s: qd.types.NDArray[qd.f32, 2], d: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = Tile()
            if qd.static(bad_slice == "no_stop"):
                v = s[0:, 0]
            elif qd.static(bad_slice == "no_start"):
                v = s[:tile_size, 0]
            t -= qd.outer(v, v)
            d[0:tile_size, 0:tile_size] = t

    with pytest.raises(QuadrantsSyntaxError, match="both start and stop"):
        k1(src, dst)


# =============================================================================
# PR 4: public-API tests (qd.simt.Tile16x16 proxy, tensor_type, SharedArray)
# =============================================================================

_M = 40


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load_store_partial(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 12
    src = tensor_type(qd_dtype, (_TILE, _TILE))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann, NCOLS: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:tile_size, 0:NCOLS]
            dst_arr[0:tile_size, 0:tile_size] = t

    data = np.arange(_TILE * _TILE, dtype=np_dtype).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst, NCOLS)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@pytest.mark.parametrize("nrows", [10, 8, 1])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_load_store_partial_rows(qd_dtype, nrows):
    """Load/store with fewer than 16 rows -- threads beyond nrows should be skipped."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    GRID = 48
    src = qd.ndarray(qd_dtype, (GRID, GRID))
    dst = qd.ndarray(qd_dtype, (GRID, GRID))

    @qd.kernel(fastcache=True)
    def k1(src_arr: qd.types.NDArray[qd_dtype, 2], dst_arr: qd.types.NDArray[qd_dtype, 2], nrows: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:nrows, 0:tile_size]
            dst_arr[0:nrows, 0:tile_size] = t

    data = np.arange(GRID * GRID, dtype=np_dtype).reshape(GRID, GRID) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((GRID, GRID), -1.0, dtype=np_dtype))
    k1(src, dst, nrows)

    result = dst.to_numpy()
    expected = np.full((GRID, GRID), -1.0, dtype=np_dtype)
    expected[:nrows, :_TILE] = data[:nrows, :_TILE]
    np.testing.assert_allclose(result, expected)


@pytest.mark.slow
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_syr_sub(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    mat = tensor_type(qd_dtype, (_TILE, _TILE))
    vec = tensor_type(qd_dtype, (_TILE,))
    out = tensor_type(qd_dtype, (_TILE, _TILE))

    Ann2 = _ann(tensor_type, qd_dtype, 2)
    Ann1 = _ann(tensor_type, qd_dtype, 1)

    @qd.kernel(fastcache=True)
    def k1(mat_arr: Ann2, vec_arr: Ann1, out_arr: Ann2):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for tid in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            t -= qd.outer(vec_arr[tid], vec_arr[tid])
            out_arr[0:tile_size, 0:tile_size] = t

    rng = np.random.RandomState(123)
    R = rng.randn(_TILE, _TILE).astype(np_dtype)
    v = rng.randn(_TILE).astype(np_dtype)
    mat.from_numpy(R)
    vec.from_numpy(v)
    k1(mat, vec, out)
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(v, v), atol=1e-5)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_potrf(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    atol = _ATOLS[qd_dtype]
    src = tensor_type(qd_dtype, (_TILE, _TILE))
    dst = tensor_type(qd_dtype, (_TILE, _TILE))
    eps_field = qd.field(dtype=qd_dtype, shape=())

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann, eps_f: qd.Template):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:tile_size, 0:tile_size]
            t.cholesky_(eps_f[None])
            dst_arr[0:tile_size, 0:tile_size] = t

    A = _make_spd(np_dtype)
    src.from_numpy(A)
    eps_field[None] = _EPS_VALS[qd_dtype]
    k1(src, dst, eps_field)
    L_expected = np.linalg.cholesky(A.astype(np.float64)).astype(np_dtype)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=atol)


@pytest.mark.slow
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_tile16_potrf_then_trsm(tensor_type, qd_dtype):
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    atol = _ATOLS[qd_dtype]
    a_field = tensor_type(qd_dtype, (_TILE, _TILE))
    b_field = tensor_type(qd_dtype, (_TILE, _TILE))
    x_field = tensor_type(qd_dtype, (_TILE, _TILE))
    eps_field = qd.field(dtype=qd_dtype, shape=())

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel(fastcache=True)
    def k1(a_arr: Ann, b_arr: Ann, x_arr: Ann, eps_f: qd.Template):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            L = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            L[:] = a_arr[0:tile_size, 0:tile_size]
            L.cholesky_(eps_f[None])
            B = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            B[:] = b_arr[0:tile_size, 0:tile_size]
            L.solve_triangular_(B)
            x_arr[0:tile_size, 0:tile_size] = B

    A = _make_spd(np_dtype, seed=55)
    rng = np.random.RandomState(66)
    Bnp = rng.randn(_TILE, _TILE).astype(np_dtype)
    a_field.from_numpy(A)
    b_field.from_numpy(Bnp)
    eps_field[None] = _EPS_VALS[qd_dtype]
    k1(a_field, b_field, x_field, eps_field)
    X = x_field.to_numpy()
    L_ref = np.linalg.cholesky(A.astype(np.float64))
    X_ref = scipy.linalg.solve_triangular(L_ref, Bnp.T.astype(np.float64), lower=True).T.astype(np_dtype)
    np.testing.assert_allclose(X, X_ref, atol=max(atol, 1e-3))


# -- SharedArray tests --


@pytest.mark.slow
@test_utils.test(arch=qd.gpu)
def test_tile16_shared_array_roundtrip():
    """Load from field -> tile -> SharedArray -> tile -> field, verify data survives."""
    src = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))
    dst = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(src_f: qd.Template, dst_f: qd.Template):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            sh = qd.simt.block.SharedArray((qd.simt.Tile16x16.SIZE, qd.simt.Tile16x16.SIZE), qd.f32)
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src_f[0:tile_size, 0:tile_size]
            sh[0:tile_size, 0:tile_size] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t2[:] = sh[0:tile_size, 0:tile_size]
            dst_f[0:tile_size, 0:tile_size] = t2

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


@pytest.mark.slow
@pytest.mark.parametrize("partial_store,partial_load", [(True, True), (True, False), (False, True)])
@test_utils.test(arch=qd.gpu)
def test_tile16_shared_array_partial_cols(partial_store, partial_load):
    """Partial-column load/store through SharedArray."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))
    dst = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(src_f: qd.Template, dst_f: qd.Template, NCOLS: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            sh = qd.simt.block.SharedArray((qd.simt.Tile16x16.SIZE, qd.simt.Tile16x16.SIZE), qd.f32)
            tid = qd.simt.subgroup.invocation_id()
            for c in range(tile_size):
                sh[tid, c] = qd.f32(-1.0)
            qd.simt.block.sync()

            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            if qd.static(partial_store):
                t[:] = src_f[0:tile_size, 0:NCOLS]
                sh[0:tile_size, 0:NCOLS] = t
            else:
                t[:] = src_f[0:tile_size, 0:tile_size]
                sh[0:tile_size, 0:tile_size] = t
            qd.simt.block.sync()

            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            if qd.static(partial_load):
                t2[:] = sh[0:tile_size, 0:NCOLS]
            else:
                t2[:] = sh[0:tile_size, 0:tile_size]
            dst_f[0:tile_size, 0:tile_size] = t2

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst, NCOLS)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    if partial_load:
        np.testing.assert_allclose(result[:, NCOLS:], 0.0)
    else:
        np.testing.assert_allclose(result[:, NCOLS:], -1.0)


@pytest.mark.slow
@test_utils.test(arch=qd.gpu)
def test_tile16_shared_array_cholesky():
    """Cholesky via tiles, L stored in SharedArray, verify reconstruction."""
    src = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))
    dst = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))
    eps_field = qd.field(dtype=qd.f32, shape=())

    @qd.kernel(fastcache=True)
    def k1(src_f: qd.Template, dst_f: qd.Template, eps_f: qd.Template):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            sh = qd.simt.block.SharedArray((qd.simt.Tile16x16.SIZE, qd.simt.Tile16x16.SIZE), qd.f32)
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src_f[0:tile_size, 0:tile_size]
            t.cholesky_(eps_f[None])
            sh[0:tile_size, 0:tile_size] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t2[:] = sh[0:tile_size, 0:tile_size]
            dst_f[0:tile_size, 0:tile_size] = t2

    A = _make_spd()
    src.from_numpy(A)
    eps_field[None] = 1e-10
    k1(src, dst, eps_field)
    L_expected = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=1e-4)


@test_utils.test(arch=qd.gpu)
def test_tile16_shared_array_clamp_store():
    """Store tile to SharedArray narrower than 16 cols. Must auto-clamp, no OOB."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))
    dst = qd.field(dtype=qd.f32, shape=(_TILE, NCOLS))

    @qd.kernel(fastcache=True)
    def k1(src_f: qd.Template, dst_f: qd.Template, NCOLS: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            sh = qd.simt.block.SharedArray((qd.simt.Tile16x16.SIZE, 10), qd.f32)
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src_f[0:tile_size, 0:tile_size]
            sh[0:tile_size, 0:tile_size] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t2[:] = sh[0:tile_size, 0:NCOLS]
            dst_f[0:tile_size, 0:NCOLS] = t2

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)
    k1(src, dst, NCOLS)
    result = dst.to_numpy()
    np.testing.assert_allclose(result, data[:, :NCOLS])


@test_utils.test(arch=qd.gpu)
def test_tile16_shared_array_clamp_load():
    """Load tile from SharedArray narrower than 16 cols. Must auto-clamp, extra regs zero."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(_TILE, NCOLS))
    dst = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(src_f: qd.Template, dst_f: qd.Template, NCOLS: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            sh = qd.simt.block.SharedArray((qd.simt.Tile16x16.SIZE, 10), qd.f32)
            t_load = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t_load[:] = src_f[0:tile_size, 0:NCOLS]
            sh[0:tile_size, 0:NCOLS] = t_load
            qd.simt.block.sync()
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = sh[0:tile_size, 0:tile_size]
            dst_f[0:tile_size, 0:tile_size] = t

    data = np.arange(_TILE * NCOLS, dtype=np.float32).reshape(_TILE, NCOLS) + 1.0
    src.from_numpy(data)
    k1(src, dst, NCOLS)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data)
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


# -- Vec proxy tests with tensor_type --


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_syr_sub_2d(tensor_type):
    """Symmetric rank-1 subtract via vec proxy from a 2D array, non-zero row offset."""
    mat = tensor_type(qd.f32, (_TILE, _TILE))
    vecs = tensor_type(qd.f32, (_M, _M))
    out = tensor_type(qd.f32, (_TILE, _TILE))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 2)

    K0 = 16
    COL = 5

    @qd.kernel(fastcache=True)
    def k1(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile, K0: qd.i32, COL: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            v = vecs_arr[K0 : K0 + qd.simt.Tile16x16.SIZE, COL]
            t -= qd.outer(v, v)
            out_arr[0:tile_size, 0:tile_size] = t

    rng = np.random.RandomState(100)
    R = rng.randn(_TILE, _TILE).astype(np.float32)
    V = rng.randn(_M, _M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    k1(mat, vecs, out, K0, COL)
    col = V[K0 : K0 + 16, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_syr_sub_3d(tensor_type):
    """Symmetric rank-1 subtract via vec proxy from a 3D array (batch dimension)."""
    N_BATCH = 2
    mat = tensor_type(qd.f32, (_TILE, _TILE))
    vecs = tensor_type(qd.f32, (N_BATCH, _M, _M))
    out = tensor_type(qd.f32, (_TILE, _TILE))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 3)

    K0 = 16
    COL = 3

    @qd.kernel(fastcache=True)
    def k1(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile, K0: qd.i32, COL: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            v = vecs_arr[1, K0 : K0 + qd.simt.Tile16x16.SIZE, COL]
            t -= qd.outer(v, v)
            out_arr[0:tile_size, 0:tile_size] = t

    rng = np.random.RandomState(200)
    R = rng.randn(_TILE, _TILE).astype(np.float32)
    V = rng.randn(N_BATCH, _M, _M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    k1(mat, vecs, out, K0, COL)
    col = V[1, K0 : K0 + 16, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@pytest.mark.slow
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_shared_array():
    """Symmetric rank-1 subtract via vec proxy from SharedArray at non-zero offset."""
    mat = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))
    vecs = qd.field(dtype=qd.f32, shape=(_M, _M))
    out = qd.field(dtype=qd.f32, shape=(_TILE, _TILE))

    K0 = 16
    COL = 2

    @qd.kernel(fastcache=True)
    def k1(mat_f: qd.Template, vecs_f: qd.Template, out_f: qd.Template, K0: qd.i32, COL: qd.i32, m_size: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            sh = qd.simt.block.SharedArray((40, 40), qd.f32)
            tid = qd.simt.subgroup.invocation_id()
            for row in range(m_size):
                if row % tile_size == tid:
                    for c in range(m_size):
                        sh[row, c] = vecs_f[row, c]
            qd.simt.block.sync()
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_f[0:tile_size, 0:tile_size]
            v = sh[K0 : K0 + qd.simt.Tile16x16.SIZE, COL]
            t -= qd.outer(v, v)
            out_f[0:tile_size, 0:tile_size] = t

    rng = np.random.RandomState(400)
    R = rng.randn(_TILE, _TILE).astype(np.float32)
    V = rng.randn(_M, _M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    k1(mat, vecs, out, K0, COL, _M)
    col = V[K0 : K0 + 16, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_partial_rows(tensor_type):
    """Vec proxy with partial last block: only _M-K0=8 of 16 threads contribute."""
    mat = tensor_type(qd.f32, (_TILE, _TILE))
    vecs = tensor_type(qd.f32, (_M, _M))
    out = tensor_type(qd.f32, (_TILE, _TILE))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 2)

    K0 = 32
    COL = 3

    @qd.kernel(fastcache=True)
    def k1(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile, K0: qd.i32, COL: qd.i32, m_size: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            v = vecs_arr[K0:m_size, COL]
            t -= qd.outer(v, v)
            out_arr[0:tile_size, 0:tile_size] = t

    rng = np.random.RandomState(500)
    R = rng.randn(_TILE, _TILE).astype(np.float32)
    V = rng.randn(_M, _M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    k1(mat, vecs, out, K0, COL, _M)
    col_padded = np.zeros(_TILE, dtype=np.float32)
    col_padded[: _M - K0] = V[K0:_M, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col_padded, col_padded), atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test(arch=qd.gpu)
def test_tile16_vec_proxy_multi_column_accumulate(tensor_type):
    """Accumulate rank-1 updates over columns at a non-zero row offset, like Cholesky lookback."""
    mat = tensor_type(qd.f32, (_TILE, _TILE))
    vecs = tensor_type(qd.f32, (_M, _M))
    out = tensor_type(qd.f32, (_TILE, _TILE))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 2)

    K0 = 16
    NCOLS = 4

    @qd.kernel(fastcache=True)
    def k1(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile, K0: qd.i32, NCOLS: qd.i32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:tile_size, 0:tile_size]
            for c in range(NCOLS):
                v = vecs_arr[K0 : K0 + qd.simt.Tile16x16.SIZE, c]
                t -= qd.outer(v, v)
            out_arr[0:tile_size, 0:tile_size] = t

    rng = np.random.RandomState(600)
    R = rng.randn(_TILE, _TILE).astype(np.float32)
    V = rng.randn(_M, _M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    k1(mat, vecs, out, K0, NCOLS)
    expected = R.copy()
    for c in range(NCOLS):
        col = V[K0 : K0 + 16, c]
        expected -= np.outer(col, col)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-4)


@test_utils.test(arch=qd.gpu, exclude=[qd.vulkan, qd.metal])
def test_tile16_f64_roundtrip_into_f32_array():
    """Load f32 data through an f64 tile and store back -- must be lossless."""
    src = qd.ndarray(shape=(_TILE, _TILE), dtype=qd.f32)
    dst_f32 = qd.ndarray(shape=(_TILE, _TILE), dtype=qd.f32)
    dst_f64 = qd.ndarray(shape=(_TILE, _TILE), dtype=qd.f32)

    Ann = qd.types.NDArray[qd.f32, 2]

    @qd.kernel(fastcache=True)
    def roundtrip_f32(s: Ann, d: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = s[0:tile_size, 0:tile_size]
            d[0:tile_size, 0:tile_size] = t

    @qd.kernel(fastcache=True)
    def roundtrip_f64(s: Ann, d: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f64)
            t[:] = s[0:tile_size, 0:tile_size]
            d[0:tile_size, 0:tile_size] = t

    data = np.arange(_TILE * _TILE, dtype=np.float32).reshape(_TILE, _TILE) + 1.0
    src.from_numpy(data)

    roundtrip_f32(src, dst_f32)
    roundtrip_f64(src, dst_f64)

    np.testing.assert_array_equal(dst_f32.to_numpy(), data)
    np.testing.assert_array_equal(dst_f64.to_numpy(), data)


@test_utils.test(arch=qd.cpu)
def test_tile16_raises_on_cpu():
    """Using Tile16x16 on a CPU backend must raise QuadrantsSyntaxError, not crash."""

    @qd.kernel(fastcache=True)
    def k1(dst: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            dst[0:tile_size, 0:tile_size] = t

    dst = qd.ndarray(qd.f32, (_TILE, _TILE))
    with pytest.raises(QuadrantsSyntaxError, match="requires a GPU backend"):
        k1(dst)


# -- Proxy tests (qd.simt.Tile16x16 as a proxy object) --


@test_utils.test(arch=qd.gpu)
def test_proxy_size_constant():
    assert qd.simt.Tile16x16.SIZE == 16


@test_utils.test(arch=qd.gpu)
def test_simt_invalid_attr_raises():
    with pytest.raises(AttributeError):
        _ = qd.simt.NoSuchThing


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test(arch=qd.gpu)
def test_proxy_default_dtype(tensor_type):
    """Omitting dtype= uses the compile config's default_fp (f32 by default)."""
    dst = tensor_type(qd.f32, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel(fastcache=True)
    def k1(dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.zeros()
            dst_arr[0:tile_size, 0:tile_size] = t

    k1(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((_TILE, _TILE), dtype=np.float32))


@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
@test_utils.test(arch=qd.gpu)
def test_proxy_eye_explicit_dtype(qd_dtype):
    """eye(dtype=...) via the proxy must produce an identity tile."""
    test_utils.skip_if_f64_unsupported(qd_dtype)
    np_dtype = _NP_DTYPES[qd_dtype]
    dst = qd.ndarray(qd_dtype, (_TILE, _TILE))

    @qd.kernel(fastcache=True)
    def k1(d: qd.types.NDArray[qd_dtype, 2]):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.eye(dtype=qd_dtype)
            d[0:tile_size, 0:tile_size] = t

    k1(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(_TILE, dtype=np_dtype))


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test(arch=qd.gpu)
def test_proxy_in_func(tensor_type):
    """Proxy works when called from a @qd.func, not just @qd.kernel."""
    src = tensor_type(qd.f32, (_TILE, _TILE))
    dst = tensor_type(qd.f32, (_TILE, _TILE))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.func
    def cholesky_via_proxy(s: Ann, d: Ann):
        t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        t[:] = s[0:_TILE, 0:_TILE]
        t.cholesky_(qd.f32(1e-6))
        d[0:_TILE, 0:_TILE] = t

    @qd.kernel(fastcache=True)
    def k1(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            cholesky_via_proxy(src_arr, dst_arr)

    H = _make_spd()
    src.from_numpy(H)
    dst.from_numpy(np.zeros_like(H))
    k1(src, dst)

    L_qd = np.tril(dst.to_numpy())
    L_ref = np.linalg.cholesky(H)
    np.testing.assert_allclose(L_qd, L_ref, atol=1e-4)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test(arch=qd.gpu, exclude=[qd.vulkan, qd.metal])
def test_proxy_default_dtype_survives_reinit(tensor_type):
    """Proxy with default dtype must follow default_fp across init/reset cycles.

    This is the actual regression scenario: init with f64, compile a kernel, reset, reinit with f32, compile the
    same kernel pattern -- the second kernel must use f32 tiles, not stale f64.
    """
    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

    def _make(dtype):
        if tensor_type == qd.ndarray:
            return qd.ndarray(dtype, (_TILE, _TILE))
        return qd.field(dtype, shape=(_TILE, _TILE))

    Ann64 = _ann(tensor_type, qd.f64, 2)
    Ann32 = _ann(tensor_type, qd.f32, 2)

    @qd.kernel(fastcache=True)
    def write_eye_f64(dst: Ann64):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.eye()
            dst[0:tile_size, 0:tile_size] = t

    @qd.kernel(fastcache=True)
    def write_eye_f32(dst: Ann32):
        qd.loop_config(block_dim=qd.simt.Tile16x16.SIZE)
        tile_size = qd.simt.Tile16x16.SIZE
        for _ in range(tile_size):
            t = qd.simt.Tile16x16.eye()
            dst[0:tile_size, 0:tile_size] = t

    impl.get_runtime().set_default_fp(qd.f64)
    dst64 = _make(qd.f64)
    write_eye_f64(dst64)
    np.testing.assert_allclose(dst64.to_numpy(), np.eye(_TILE))

    qd.reset()
    qd.init(arch=qd.gpu, default_fp=qd.f32)

    dst32 = _make(qd.f32)
    write_eye_f32(dst32)
    result32 = dst32.to_numpy()

    np.testing.assert_allclose(result32, np.eye(_TILE, dtype=np.float32))
    assert result32.dtype == np.float32


@pytest.mark.slow
@test_utils.test(arch=[qd.cuda])
def test_tile16_cholesky_blocked_demo():
    """Smoke-test that misc/demos/cholesky_blocked.py runs to completion."""
    demo = Path(__file__).resolve().parents[2] / "misc" / "demos" / "cholesky_blocked.py"
    result = subprocess.run([sys.executable, str(demo)], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        pytest.fail(f"cholesky_blocked.py exited with code {result.returncode}\nstderr:\n{result.stderr}")
