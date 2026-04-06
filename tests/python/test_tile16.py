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


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_syr_sub_2d(tensor_type):
    """Symmetric rank-1 subtract via vec proxy from a 2D array."""
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (N, N))
    out = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(mat_arr: Ann, vecs_arr: Ann, out_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[0:N, 0]
            t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(100)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(N, N).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    col0 = V[:, 0]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col0, col0), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_syr_sub_3d(tensor_type):
    """Symmetric rank-1 subtract via vec proxy from a 3D array (batch dimension)."""
    N_BATCH = 2
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (N_BATCH, N, N))
    out = tensor_type(qd.f32, (N, N))

    Ann2 = _ann(tensor_type, qd.f32, 2)
    Ann3 = _ann(tensor_type, qd.f32, 3)

    @qd.kernel
    def run(mat_arr: Ann2, vecs_arr: Ann3, out_arr: Ann2):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[1, 0:N, 3]
            t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(200)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(N_BATCH, N, N).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    col = V[1, :, 3]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_ger_sub_2d(tensor_type):
    """General rank-1 subtract via two different vec proxies from a 2D array."""
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (N, N))
    out = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(mat_arr: Ann, vecs_arr: Ann, out_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            a = vecs_arr[0:N, 0]
            b = vecs_arr[0:N, 1]
            t -= outer(a, b)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(300)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(N, N).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(V[:, 0], V[:, 1]), atol=1e-5)


@test_utils.test(arch=qd.cuda)
def test_tile16_vec_proxy_shared_array(tensor_type=qd.field):
    """Symmetric rank-1 subtract via vec proxy from SharedArray."""
    mat = qd.field(dtype=qd.f32, shape=(N, N))
    vecs = qd.field(dtype=qd.f32, shape=(N, N))
    out = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for _ in range(N):
            sh = qd.simt.block.SharedArray((N, N), qd.f32)
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            for c in range(N):
                sh[tid, c] = vecs[tid, c]
            qd.simt.block.sync()
            t = Tile16x16.zeros()
            t[:] = mat[0:N, 0:N]
            v = sh[0:N, 2]
            t -= outer(v, v)
            out[0:N, 0:N] = t

    rng = np.random.RandomState(400)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(N, N).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run()
    col = V[:, 2]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col, col), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_partial_rows(tensor_type):
    """Vec proxy with row_stop < row_start + 16: out-of-range threads contribute 0."""
    N_VALID = 10
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (N, N))
    out = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)

    @qd.kernel
    def run(mat_arr: Ann, vecs_arr: Ann, out_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[0:N_VALID, 0]
            t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(500)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(N, N).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    col_padded = np.zeros(N, dtype=np.float32)
    col_padded[:N_VALID] = V[:N_VALID, 0]
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(col_padded, col_padded), atol=1e-5)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_vec_proxy_multi_column_accumulate(tensor_type):
    """Accumulate multiple rank-1 updates from consecutive columns, like Cholesky lookback."""
    mat = tensor_type(qd.f32, (N, N))
    vecs = tensor_type(qd.f32, (N, N))
    out = tensor_type(qd.f32, (N, N))

    Ann = _ann(tensor_type, qd.f32, 2)
    NCOLS = 4

    @qd.kernel
    def run(mat_arr: Ann, vecs_arr: Ann, out_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16x16.zeros()
            t[:] = mat_arr[0:N, 0:N]
            for c in range(NCOLS):
                v = vecs_arr[0:N, c]
                t -= outer(v, v)
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(600)
    R = rng.randn(N, N).astype(np.float32)
    V = rng.randn(N, N).astype(np.float32)
    mat.from_numpy(R)
    vecs.from_numpy(V)
    run(mat, vecs, out)
    expected = R.copy()
    for c in range(NCOLS):
        expected -= np.outer(V[:, c], V[:, c])
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-4)
