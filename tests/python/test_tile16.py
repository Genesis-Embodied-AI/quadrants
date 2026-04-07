import time

import numpy as np
import pytest
import scipy.linalg

import quadrants as qd
from quadrants.lang.simt.tile16 import Tile16x16, make_tile16x16, outer

from tests import test_utils

N = 16

Tile16x16_f64 = make_tile16x16(qd.f64)


def _make_spd(seed: int = 42, dtype: type = np.float32):
    rng = np.random.RandomState(seed)
    B = rng.randn(N, N).astype(np.float64)
    return (B @ B.T + N * np.eye(N)).astype(dtype)


def _ann(tensor_type, dtype, ndim):
    """Return the right kernel annotation for the given tensor_type."""
    if tensor_type == qd.ndarray:
        return qd.types.NDArray[dtype, ndim]
    return qd.Template


# =============================================================================
# Tile16x16 class API tests (field + ndarray)
# =============================================================================


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_zeros(tensor_type):
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((N, N), dtype=np.float32))


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_eye(tensor_type):
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.eye()
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(N, dtype=np.float32))


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_eye_inplace(tensor_type):
    """Load non-zero data into tile, call eye_(), verify identity overwrites it."""
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = src_arr[0:N, 0:N]
            t.eye_()
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 100.0
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(N, dtype=np.float32))


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_load_store(tensor_type):
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N)
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_load_store_partial(tensor_type):
    NCOLS = 12
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = src_arr[0:N, 0:NCOLS]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_store_partial_cols(tensor_type):
    """Load full 16 columns, store only NCOLS < 16. Remaining dst columns must be untouched."""
    NCOLS = 10
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:NCOLS] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((N, N), -1.0, dtype=np.float32))
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], -1.0)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_load_clamp_to_array_shape(tensor_type):
    """Load from an array narrower than 16 columns. Columns beyond arr width should be zero."""
    NCOLS = 10
    src = tensor_type(qd.f32, (N, NCOLS))
    dst = tensor_type(qd.f32, (N, N))

    Ann_src = _ann(tensor_type, qd.f32, 2)
    Ann_dst = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann_src, dst_arr: Ann_dst):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * NCOLS, dtype=np.float32).reshape(N, NCOLS) + 1.0
    src.from_numpy(data)
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data)
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_store_clamp_to_array_shape(tensor_type):
    """Store to an array narrower than 16 columns. Must not write out of bounds."""
    NCOLS = 10
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, NCOLS))

    Ann_src = _ann(tensor_type, qd.f32, 2)
    Ann_dst = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann_src, dst_arr: Ann_dst):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result, data[:, :NCOLS])


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_syr_sub(tensor_type):
    mat = tensor_type(qd.f32, (N, N))
    vec = tensor_type(qd.f32, (N,))
    out = tensor_type(qd.f32, (N, N))

    Ann2 = _ann(tensor_type, qd.f32, 2)
    Ann1 = _ann(tensor_type, qd.f32, 1)

    @qd.kernel
    def run(mat_arr: Ann2, vec_arr: Ann1, out_arr: Ann2):
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16x16(
                mat_arr[tid, 0],
                mat_arr[tid, 1],
                mat_arr[tid, 2],
                mat_arr[tid, 3],
                mat_arr[tid, 4],
                mat_arr[tid, 5],
                mat_arr[tid, 6],
                mat_arr[tid, 7],
                mat_arr[tid, 8],
                mat_arr[tid, 9],
                mat_arr[tid, 10],
                mat_arr[tid, 11],
                mat_arr[tid, 12],
                mat_arr[tid, 13],
                mat_arr[tid, 14],
                mat_arr[tid, 15],
            )
            t -= outer(vec_arr[tid], vec_arr[tid])
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(123)
    R = rng.randn(N, N).astype(np.float32)
    v = rng.randn(N).astype(np.float32)
    mat.from_numpy(R)
    vec.from_numpy(v)
    run(mat, vec, out)
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(v, v), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_ger_sub(tensor_type):
    mat = tensor_type(qd.f32, (N, N))
    vec_a = tensor_type(qd.f32, (N,))
    vec_b = tensor_type(qd.f32, (N,))
    out = tensor_type(qd.f32, (N, N))

    Ann2 = _ann(tensor_type, qd.f32, 2)
    Ann1 = _ann(tensor_type, qd.f32, 1)

    @qd.kernel
    def run(
        mat_arr: Ann2,
        va_arr: Ann1,
        vb_arr: Ann1,
        out_arr: Ann2,
    ):
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16x16(
                mat_arr[tid, 0],
                mat_arr[tid, 1],
                mat_arr[tid, 2],
                mat_arr[tid, 3],
                mat_arr[tid, 4],
                mat_arr[tid, 5],
                mat_arr[tid, 6],
                mat_arr[tid, 7],
                mat_arr[tid, 8],
                mat_arr[tid, 9],
                mat_arr[tid, 10],
                mat_arr[tid, 11],
                mat_arr[tid, 12],
                mat_arr[tid, 13],
                mat_arr[tid, 14],
                mat_arr[tid, 15],
            )
            t -= outer(va_arr[tid], vb_arr[tid])
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(456)
    R = rng.randn(N, N).astype(np.float32)
    a = rng.randn(N).astype(np.float32)
    b = rng.randn(N).astype(np.float32)
    mat.from_numpy(R)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    run(mat, vec_a, vec_b, out)
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(a, b), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_potrf(tensor_type):
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))
    eps_field = qd.field(dtype=qd.f32, shape=())

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = src_arr[0:N, 0:N]
            t.cholesky_(eps_field[None])
            dst_arr[0:N, 0:N] = t

    A = _make_spd()
    src.from_numpy(A)
    eps_field[None] = 1e-10
    run(src, dst)
    L_expected = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=1e-4)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_trsm(tensor_type):
    l_field = tensor_type(qd.f32, (N, N))
    b_field = tensor_type(qd.f32, (N, N))
    x_field = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(l_arr: Ann, b_arr: Ann, x_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = Tile16x16.zeros()
            L[:] = l_arr[0:N, 0:N]
            B = Tile16x16.zeros()
            B[:] = b_arr[0:N, 0:N]
            L.solve_triangular_(B)
            x_arr[0:N, 0:N] = B

    A = _make_spd(seed=99)
    Lnp = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    rng = np.random.RandomState(77)
    Bnp = rng.randn(N, N).astype(np.float32)
    l_field.from_numpy(Lnp)
    b_field.from_numpy(Bnp)
    run(l_field, b_field, x_field)
    X = x_field.to_numpy()
    np.testing.assert_allclose(X @ Lnp.T, Bnp, atol=1e-3)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_potrf_then_trsm(tensor_type):
    a_field = tensor_type(qd.f32, (N, N))
    b_field = tensor_type(qd.f32, (N, N))
    x_field = tensor_type(qd.f32, (N, N))
    eps_field = qd.field(dtype=qd.f32, shape=())

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(a_arr: Ann, b_arr: Ann, x_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = Tile16x16.zeros()
            L[:] = a_arr[0:N, 0:N]
            L.cholesky_(eps_field[None])
            B = Tile16x16.zeros()
            B[:] = b_arr[0:N, 0:N]
            L.solve_triangular_(B)
            x_arr[0:N, 0:N] = B

    A = _make_spd(seed=55)
    rng = np.random.RandomState(66)
    Bnp = rng.randn(N, N).astype(np.float32)
    a_field.from_numpy(A)
    b_field.from_numpy(Bnp)
    eps_field[None] = 1e-10
    run(a_field, b_field, x_field)
    X = x_field.to_numpy()
    L_ref = np.linalg.cholesky(A.astype(np.float64))
    X_ref = scipy.linalg.solve_triangular(L_ref, Bnp.T.astype(np.float64), lower=True).T.astype(np.float32)
    np.testing.assert_allclose(X, X_ref, atol=1e-3)


# =============================================================================
# f64 precision tests — verify make_tile16x16(qd.f64) preserves double precision
# =============================================================================


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_f64_load_store(tensor_type):
    src = tensor_type(qd.f64, (N, N))
    dst = tensor_type(qd.f64, (N, N))

    Ann = _ann(tensor_type, qd.f64, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16_f64.zeros()
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float64).reshape(N, N)
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_f64_potrf(tensor_type):
    src = tensor_type(qd.f64, (N, N))
    dst = tensor_type(qd.f64, (N, N))
    eps_field = qd.field(dtype=qd.f64, shape=())

    Ann = _ann(tensor_type, qd.f64, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16_f64.zeros()
            t[:] = src_arr[0:N, 0:N]
            t.cholesky_(eps_field[None])
            dst_arr[0:N, 0:N] = t

    A = _make_spd(dtype=np.float64)
    src.from_numpy(A)
    eps_field[None] = 1e-30
    run(src, dst)
    L_expected = np.linalg.cholesky(A)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=1e-12)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_f64_potrf_then_trsm(tensor_type):
    a_field = tensor_type(qd.f64, (N, N))
    b_field = tensor_type(qd.f64, (N, N))
    x_field = tensor_type(qd.f64, (N, N))
    eps_field = qd.field(dtype=qd.f64, shape=())

    Ann = _ann(tensor_type, qd.f64, 2)

    @qd.kernel
    def run(a_arr: Ann, b_arr: Ann, x_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = Tile16x16_f64.zeros()
            L[:] = a_arr[0:N, 0:N]
            L.cholesky_(eps_field[None])
            B = Tile16x16_f64.zeros()
            B[:] = b_arr[0:N, 0:N]
            L.solve_triangular_(B)
            x_arr[0:N, 0:N] = B

    A = _make_spd(seed=55, dtype=np.float64)
    rng = np.random.RandomState(66)
    Bnp = rng.randn(N, N).astype(np.float64)
    a_field.from_numpy(A)
    b_field.from_numpy(Bnp)
    eps_field[None] = 1e-30
    run(a_field, b_field, x_field)
    X = x_field.to_numpy()
    L_ref = np.linalg.cholesky(A)
    X_ref = scipy.linalg.solve_triangular(L_ref, Bnp.T, lower=True).T
    np.testing.assert_allclose(X, X_ref, atol=1e-12)


# =============================================================================
# 3D load/store tests — verify load3d/store3d with field and ndarray
# =============================================================================


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_load3d_store3d(tensor_type):
    N_BATCH = 2
    src = tensor_type(qd.f32, (N_BATCH, N, N))
    dst = tensor_type(qd.f32, (N_BATCH, N, N))

    Ann = _ann(tensor_type, qd.f32, 3)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            for i_b in range(N_BATCH):
                t = Tile16x16.zeros()
                t[:] = src_arr[i_b, 0:N, 0:N]
                dst_arr[i_b, 0:N, 0:N] = t

    data = np.arange(N_BATCH * N * N, dtype=np.float32).reshape(N_BATCH, N, N)
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


# =============================================================================
# SharedArray load/store tests — verify tile <-> shared memory transfers
# =============================================================================


@test_utils.test(arch=qd.cuda)
def test_tile16_shared_array_roundtrip():
    """Load from field -> tile -> SharedArray -> tile -> field, verify data survives."""
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, N), qd.f32)
            t = Tile16x16.zeros()
            t[:] = src[0:N, 0:N]
            sh[0:N, 0:N] = t
            qd.simt.block.sync()
            t2 = Tile16x16.zeros()
            t2[:] = sh[0:N, 0:N]
            dst[0:N, 0:N] = t2

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run()
    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=qd.cuda)
def test_tile16_shared_array_partial_cols():
    """Store/load partial columns (< 16) via SharedArray slice syntax."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, N), qd.f32)
            t = Tile16x16.zeros()
            t[:] = src[0:N, 0:NCOLS]
            sh[0:N, 0:NCOLS] = t
            qd.simt.block.sync()
            t2 = Tile16x16.zeros()
            t2[:] = sh[0:N, 0:NCOLS]
            dst[0:N, 0:N] = t2

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run()
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=qd.cuda)
def test_tile16_shared_array_cholesky():
    """Cholesky via tiles, L stored in SharedArray, verify reconstruction."""
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))
    eps_field = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, N), qd.f32)
            t = Tile16x16.zeros()
            t[:] = src[0:N, 0:N]
            t.cholesky_(eps_field[None])
            sh[0:N, 0:N] = t
            qd.simt.block.sync()
            t2 = Tile16x16.zeros()
            t2[:] = sh[0:N, 0:N]
            dst[0:N, 0:N] = t2

    A = _make_spd()
    src.from_numpy(A)
    eps_field[None] = 1e-10
    run()
    L_expected = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=1e-4)


@test_utils.test(arch=qd.cuda)
def test_tile16_shared_array_store_partial_cols():
    """Store only NCOLS < 16 from tile to SharedArray; remaining SharedArray columns untouched."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, N), qd.f32)
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            for c in range(N):
                sh[tid, c] = qd.f32(-1.0)
            qd.simt.block.sync()
            t = Tile16x16.zeros()
            t[:] = src[0:N, 0:N]
            sh[0:N, 0:NCOLS] = t
            qd.simt.block.sync()
            t2 = Tile16x16.zeros()
            t2[:] = sh[0:N, 0:N]
            dst[0:N, 0:N] = t2

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run()
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], -1.0)


@test_utils.test(arch=qd.cuda)
def test_tile16_shared_array_load_partial_cols():
    """Load only NCOLS < 16 from SharedArray to tile; remaining tile registers should be zero."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, N), qd.f32)
            t_load = Tile16x16.zeros()
            t_load[:] = src[0:N, 0:N]
            sh[0:N, 0:N] = t_load
            qd.simt.block.sync()
            t = Tile16x16.zeros()
            t[:] = sh[0:N, 0:NCOLS]
            dst[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run()
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=qd.cuda)
def test_tile16_shared_array_clamp_store():
    """Store tile to SharedArray narrower than 16 cols. Must auto-clamp, no OOB."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, NCOLS))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, NCOLS), qd.f32)
            t = Tile16x16.zeros()
            t[:] = src[0:N, 0:N]
            sh[0:N, 0:N] = t
            qd.simt.block.sync()
            t2 = Tile16x16.zeros()
            t2[:] = sh[0:N, 0:NCOLS]
            dst[0:N, 0:NCOLS] = t2

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run()
    result = dst.to_numpy()
    np.testing.assert_allclose(result, data[:, :NCOLS])


@test_utils.test(arch=qd.cuda)
def test_tile16_shared_array_clamp_load():
    """Load tile from SharedArray narrower than 16 cols. Must auto-clamp, extra regs zero."""
    NCOLS = 10
    src = qd.field(dtype=qd.f32, shape=(N, NCOLS))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, NCOLS), qd.f32)
            t_load = Tile16x16.zeros()
            t_load[:] = src[0:N, 0:NCOLS]
            sh[0:N, 0:NCOLS] = t_load
            qd.simt.block.sync()
            t = Tile16x16.zeros()
            t[:] = sh[0:N, 0:N]
            dst[0:N, 0:N] = t

    data = np.arange(N * NCOLS, dtype=np.float32).reshape(N, NCOLS) + 1.0
    src.from_numpy(data)
    run()
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data)
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


# =============================================================================
# VecSliceProxy tests — verify column-vector loads via arr[r0:r_end, col]
# =============================================================================

M = 40  # rows in vec source arrays; not a multiple of 16 to test partial blocks


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_syr_sub_2d(tensor_type):
    """Symmetric rank-1 subtract via vec proxy from a 2D array, non-zero row offset."""
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (M, M))
    out = tensor_type(qd.f32, (N, N))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 2)

    K0 = 16
    COL = 5

    @qd.kernel
    def run(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[K0:K0 + Tile16x16.SIZE, COL]
            t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(100)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(M, M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    col = V[K0:K0 + 16, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_syr_sub_3d(tensor_type):
    """Symmetric rank-1 subtract via vec proxy from a 3D array (batch dimension)."""
    N_BATCH = 2
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (N_BATCH, M, M))
    out = tensor_type(qd.f32, (N, N))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 3)

    K0 = 16
    COL = 3

    @qd.kernel
    def run(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[1, K0:K0 + Tile16x16.SIZE, COL]
            t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(200)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(N_BATCH, M, M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    col = V[1, K0:K0 + 16, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_ger_sub_2d(tensor_type):
    """General rank-1 subtract via two vec proxies at different row offsets."""
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (M, M))
    out = tensor_type(qd.f32, (N, N))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 2)

    K0_A = 0
    K0_B = 16
    COL = 7

    @qd.kernel
    def run(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            a = vecs_arr[K0_A:K0_A + Tile16x16.SIZE, COL]
            b = vecs_arr[K0_B:K0_B + Tile16x16.SIZE, COL]
            t -= outer(a, b)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(300)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(M, M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    va = V[K0_A:K0_A + 16, COL]
    vb = V[K0_B:K0_B + 16, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(va, vb), atol=1e-5)


@test_utils.test(arch=qd.cuda)
def test_tile16_vec_proxy_shared_array():
    """Symmetric rank-1 subtract via vec proxy from SharedArray at non-zero offset."""
    mat = qd.field(dtype=qd.f32, shape=(N, N))
    vecs = qd.field(dtype=qd.f32, shape=(M, M))
    out = qd.field(dtype=qd.f32, shape=(N, N))

    K0 = 16
    COL = 2

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((M, M), qd.f32)
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            for row in range(M):
                if row % N == tid:
                    for c in range(M):
                        sh[row, c] = vecs[row, c]
            qd.simt.block.sync()
            t = Tile16x16.zeros()
            t[:] = mat[0:N, 0:N]
            v = sh[K0:K0 + Tile16x16.SIZE, COL]
            t -= outer(v, v)
            out[0:N, 0:N] = t

    rng = np.random.RandomState(400)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(M, M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run()
    col = V[K0:K0 + 16, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_partial_rows(tensor_type):
    """Vec proxy with partial last block: only M-K0=8 of 16 threads contribute."""
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (M, M))
    out = tensor_type(qd.f32, (N, N))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 2)

    K0 = 32
    COL = 3

    @qd.kernel
    def run(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[K0:M, COL]
            t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(500)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(M, M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    col_padded = np.zeros(N, dtype=np.float32)
    col_padded[:M - K0] = V[K0:M, COL]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col_padded, col_padded), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_multi_column_accumulate(tensor_type):
    """Accumulate rank-1 updates over columns at a non-zero row offset, like Cholesky lookback."""
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (M, M))
    out = tensor_type(qd.f32, (N, N))

    Ann_tile = _ann(tensor_type, qd.f32, 2)
    Ann_vecs = _ann(tensor_type, qd.f32, 2)

    K0 = 16
    NCOLS = 4

    @qd.kernel
    def run(mat_arr: Ann_tile, vecs_arr: Ann_vecs, out_arr: Ann_tile):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            for c in range(NCOLS):
                v = vecs_arr[K0:K0 + Tile16x16.SIZE, c]
                t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(600)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(M, M).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    expected = R.copy()
    for c in range(NCOLS):
        col = V[K0:K0 + 16, c]
        expected -= np.outer(col, col)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-4)


# =============================================================================
# Dtype mismatch regression tests — f64 tile on f32 arrays
#
# Reproduces the g1_fall 50% FPS regression: when Genesis's solver.py captures
# Tile16x16 = make_tile16x16(qd.f64) at import time (because a CPU test with
# precision=64 imported it first), all subsequent GPU kernels use f64 tile
# registers on f32 data — doubling register pressure and halving throughput.
# =============================================================================


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_tile16_f64_roundtrip_into_f32_array():
    """Load f32 data through an f64 tile and store back — must be lossless."""

    Tile_f64 = make_tile16x16(qd.f64)
    Tile_f32 = make_tile16x16(qd.f32)

    src = qd.ndarray(shape=(N, N), dtype=qd.f32)
    dst_f32 = qd.ndarray(shape=(N, N), dtype=qd.f32)
    dst_f64 = qd.ndarray(shape=(N, N), dtype=qd.f32)

    Ann = qd.types.NDArray[qd.f32, 2]

    @qd.kernel
    def roundtrip_f32(s: Ann, d: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile_f32()
            t[:] = s[0:N, 0:N]
            d[0:N, 0:N] = t

    @qd.kernel
    def roundtrip_f64(s: Ann, d: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile_f64()
            t[:] = s[0:N, 0:N]
            d[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)

    roundtrip_f32(src, dst_f32)
    roundtrip_f64(src, dst_f64)

    np.testing.assert_array_equal(dst_f32.to_numpy(), data)
    np.testing.assert_array_equal(dst_f64.to_numpy(), data)

_ENVS = 4096
_NDOFS = 48
_N_BLOCKS = (_NDOFS + N - 1) // N


def _make_cholesky_kernel(tile_cls):
    """Build a tiled Cholesky kernel using the given Tile16x16 class."""

    @qd.kernel
    def cholesky_tiled(H: qd.types.NDArray[qd.f32, 3], eps_arr: qd.types.NDArray[qd.f32, 1]):
        n_dofs = H.shape[1]
        _B = H.shape[0]
        EPS = eps_arr[0]

        qd.loop_config(block_dim=tile_cls.SIZE)
        for i in range(_B * tile_cls.SIZE):
            tid = i % tile_cls.SIZE
            i_b = i // tile_cls.SIZE
            if i_b >= _B:
                continue

            for kb in range(_N_BLOCKS):
                k0 = kb * tile_cls.SIZE

                L_kk = tile_cls()
                if k0 + tid < n_dofs:
                    L_kk[:] = H[i_b, k0 : k0 + tile_cls.SIZE, k0:n_dofs]
                else:
                    L_kk.eye_()

                for jb in range(kb):
                    j0 = jb * tile_cls.SIZE
                    for t in range(tile_cls.SIZE):
                        v = H[i_b, k0:n_dofs, j0 + t]
                        L_kk -= outer(v, v)

                L_kk.cholesky_(EPS)

                for ib in range(kb + 1, _N_BLOCKS):
                    i0 = ib * tile_cls.SIZE

                    L_ik = tile_cls()
                    if i0 + tid < n_dofs:
                        L_ik[:] = H[i_b, i0 : i0 + tile_cls.SIZE, k0:n_dofs]

                    for jb in range(kb):
                        j0 = jb * tile_cls.SIZE
                        for t in range(tile_cls.SIZE):
                            v_own = H[i_b, i0:n_dofs, j0 + t]
                            v_diag = H[i_b, k0:n_dofs, j0 + t]
                            L_ik -= outer(v_own, v_diag)

                    L_kk.solve_triangular_(L_ik)

                    if i0 + tid < n_dofs:
                        H[i_b, i0 : i0 + tile_cls.SIZE, k0:n_dofs] = L_ik

                if k0 + tid < n_dofs:
                    H[i_b, k0 : k0 + tile_cls.SIZE, k0:n_dofs] = L_kk

    return cholesky_tiled


@test_utils.test(arch=qd.cuda)
def test_tile16_f64_on_f32_arrays_perf_regression():
    """f64 Tile16x16 on f32 arrays must not be silently slower than f32 tile.

    This reproduces the g1_fall regression where solver.py captured
    Tile16x16 = make_tile16x16(qd.f64) at module import time because
    a CPU test with precision=64 imported it first.  The f64 tile uses
    double-width registers for all Cholesky/trsm math, causing ~2x
    slowdown on f32 data.
    """
    Tile_f32 = make_tile16x16(qd.f32)
    Tile_f64 = make_tile16x16(qd.f64)

    kernel_f32 = _make_cholesky_kernel(Tile_f32)
    kernel_f64 = _make_cholesky_kernel(Tile_f64)

    rng = np.random.RandomState(42)
    B_np = rng.randn(_ENVS, _NDOFS, _NDOFS).astype(np.float32)
    H_np = np.einsum("bij,bkj->bik", B_np, B_np) + _NDOFS * np.eye(_NDOFS, dtype=np.float32)

    H_f32 = qd.ndarray(qd.f32, (_ENVS, _NDOFS, _NDOFS))
    H_f64_on_f32 = qd.ndarray(qd.f32, (_ENVS, _NDOFS, _NDOFS))
    eps_arr = qd.ndarray(qd.f32, (1,))
    eps_arr.from_numpy(np.array([1e-6], dtype=np.float32))

    # Warmup + correctness
    H_f32.from_numpy(H_np.copy())
    kernel_f32(H_f32, eps_arr)
    result_f32 = H_f32.to_numpy()

    H_f64_on_f32.from_numpy(H_np.copy())
    kernel_f64(H_f64_on_f32, eps_arr)
    result_f64 = H_f64_on_f32.to_numpy()

    # Both must produce the same Cholesky factor (lower triangle)
    for b in [0, 1, _ENVS - 1]:
        L32 = np.tril(result_f32[b])
        L64 = np.tril(result_f64[b])
        np.testing.assert_allclose(L32, L64, atol=1e-3,
                                   err_msg=f"Cholesky mismatch at batch {b}")

    # Benchmark
    REPS = 20
    qd.sync()
    H_f32.from_numpy(H_np.copy())
    kernel_f32(H_f32, eps_arr)  # compile warmup
    qd.sync()

    t0 = time.perf_counter()
    for _ in range(REPS):
        H_f32.from_numpy(H_np.copy())
        kernel_f32(H_f32, eps_arr)
    qd.sync()
    ms_f32 = (time.perf_counter() - t0) / REPS * 1000

    H_f64_on_f32.from_numpy(H_np.copy())
    kernel_f64(H_f64_on_f32, eps_arr)  # compile warmup
    qd.sync()

    t0 = time.perf_counter()
    for _ in range(REPS):
        H_f64_on_f32.from_numpy(H_np.copy())
        kernel_f64(H_f64_on_f32, eps_arr)
    qd.sync()
    ms_f64 = (time.perf_counter() - t0) / REPS * 1000

    ratio = ms_f64 / ms_f32
    print(f"\n[tile16 dtype mismatch regression test]")
    print(f"  f32 tile on f32 data: {ms_f32:.2f} ms/call")
    print(f"  f64 tile on f32 data: {ms_f64:.2f} ms/call")
    print(f"  ratio: {ratio:.2f}x")

    assert ratio > 1.3, (
        f"Expected f64 tile on f32 arrays to be >1.3x slower than f32 tile, "
        f"got {ratio:.2f}x. If f64 is as fast as f32, the dtype mismatch "
        f"regression may no longer apply on this hardware."
    )


# =============================================================================
# qd.simt.Tile16x16 proxy API tests
# =============================================================================


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_proxy_zeros(tensor_type):
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((N, N), dtype=np.float32))


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_proxy_eye(tensor_type):
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.eye(dtype=qd.f32)
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(N, dtype=np.float32))


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_proxy_default_dtype(tensor_type):
    """Omitting dtype= uses the compile config's default_fp (f32 by default)."""
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros()
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((N, N), dtype=np.float32))


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_proxy_call_is_zeros(tensor_type):
    """Calling the proxy directly (no method) produces a zero tile."""
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16(dtype=qd.f32)
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((N, N), dtype=np.float32))


@test_utils.test(arch=qd.cuda)
def test_proxy_size_constant():
    assert qd.simt.Tile16x16.SIZE == 16


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_proxy_cholesky(tensor_type):
    """Full Cholesky factorisation via proxy API."""
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src_arr[0:N, 0:N]
            t.cholesky_(qd.f32(1e-6))
            dst_arr[0:N, 0:N] = t

    H = _make_spd()
    src.from_numpy(H)
    dst.from_numpy(np.zeros_like(H))
    run(src, dst)

    L_qd = np.tril(dst.to_numpy())
    L_ref = np.linalg.cholesky(H)
    np.testing.assert_allclose(L_qd, L_ref, atol=1e-4)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_proxy_f64(tensor_type):
    """Proxy with explicit dtype=qd.f64 produces f64-precision tiles."""
    src = tensor_type(qd.f64, (N, N))
    dst = tensor_type(qd.f64, (N, N))

    Ann = _ann(tensor_type, qd.f64, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f64)
            t[:] = src_arr[0:N, 0:N]
            t.cholesky_(qd.f64(1e-14))
            dst_arr[0:N, 0:N] = t

    H = _make_spd(dtype=np.float64)
    src.from_numpy(H)
    dst.from_numpy(np.zeros_like(H))
    run(src, dst)

    L_qd = np.tril(dst.to_numpy())
    L_ref = np.linalg.cholesky(H)
    np.testing.assert_allclose(L_qd, L_ref, atol=1e-10)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_proxy_in_func(tensor_type):
    """Proxy works when called from a @qd.func, not just @qd.kernel."""
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.func
    def cholesky_via_proxy(s: Ann, d: Ann):
        t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        t[:] = s[0:N, 0:N]
        t.cholesky_(qd.f32(1e-6))
        d[0:N, 0:N] = t

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            cholesky_via_proxy(src_arr, dst_arr)

    H = _make_spd()
    src.from_numpy(H)
    dst.from_numpy(np.zeros_like(H))
    run(src, dst)

    L_qd = np.tril(dst.to_numpy())
    L_ref = np.linalg.cholesky(H)
    np.testing.assert_allclose(L_qd, L_ref, atol=1e-4)
