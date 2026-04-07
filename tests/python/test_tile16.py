import time

import numpy as np
import pytest
import scipy.linalg

import quadrants as qd
from quadrants.lang.simt.tile16 import outer

from tests import test_utils

N = 16

_QD_DTYPES = [qd.f32, qd.f64]
_NP_DTYPES = {qd.f32: np.float32, qd.f64: np.float64}
_ATOLS = {qd.f32: 1e-4, qd.f64: 1e-10}
_EPS_VALS = {qd.f32: 1e-6, qd.f64: 1e-14}


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
# Tile16x16 API tests (field + ndarray, f32 + f64)
# =============================================================================


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_zeros(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    dst = tensor_type(qd_dtype, (N, N))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.zeros((N, N), dtype=np_dtype))


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_eye(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    dst = tensor_type(qd_dtype, (N, N))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.eye(dtype=qd_dtype)
            dst_arr[0:N, 0:N] = t

    run(dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(N, dtype=np_dtype))


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_eye_inplace(tensor_type, qd_dtype):
    """Load non-zero data into tile, call eye_(), verify identity overwrites it."""
    np_dtype = _NP_DTYPES[qd_dtype]
    src = tensor_type(qd_dtype, (N, N))
    dst = tensor_type(qd_dtype, (N, N))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:N, 0:N]
            t.eye_()
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np_dtype).reshape(N, N) + 100.0
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), np.eye(N, dtype=np_dtype))


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_load_store(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    src = tensor_type(qd_dtype, (N, N))
    dst = tensor_type(qd_dtype, (N, N))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np_dtype).reshape(N, N)
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_load_store_partial(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 12
    src = tensor_type(qd_dtype, (N, N))
    dst = tensor_type(qd_dtype, (N, N))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:N, 0:NCOLS]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np_dtype).reshape(N, N) + 1.0
    src.from_numpy(data)
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_store_partial_cols(tensor_type, qd_dtype):
    """Load full 16 columns, store only NCOLS < 16. Remaining dst columns must be untouched."""
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 10
    src = tensor_type(qd_dtype, (N, N))
    dst = tensor_type(qd_dtype, (N, N))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:NCOLS] = t

    data = np.arange(N * N, dtype=np_dtype).reshape(N, N) + 1.0
    src.from_numpy(data)
    dst.from_numpy(np.full((N, N), -1.0, dtype=np_dtype))
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], -1.0)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_load_clamp_to_array_shape(tensor_type, qd_dtype):
    """Load from an array narrower than 16 columns. Columns beyond arr width should be zero."""
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 10
    src = tensor_type(qd_dtype, (N, NCOLS))
    dst = tensor_type(qd_dtype, (N, N))

    Ann_src = _ann(tensor_type, qd_dtype, 2)
    Ann_dst = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(src_arr: Ann_src, dst_arr: Ann_dst):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * NCOLS, dtype=np_dtype).reshape(N, NCOLS) + 1.0
    src.from_numpy(data)
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data)
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_store_clamp_to_array_shape(tensor_type, qd_dtype):
    """Store to an array narrower than 16 columns. Must not write out of bounds."""
    np_dtype = _NP_DTYPES[qd_dtype]
    NCOLS = 10
    src = tensor_type(qd_dtype, (N, N))
    dst = tensor_type(qd_dtype, (N, NCOLS))

    Ann_src = _ann(tensor_type, qd_dtype, 2)
    Ann_dst = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(src_arr: Ann_src, dst_arr: Ann_dst):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:N, 0:N]
            dst_arr[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np_dtype).reshape(N, N) + 1.0
    src.from_numpy(data)
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result, data[:, :NCOLS])


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_syr_sub(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    mat = tensor_type(qd_dtype, (N, N))
    vec = tensor_type(qd_dtype, (N,))
    out = tensor_type(qd_dtype, (N, N))

    Ann2 = _ann(tensor_type, qd_dtype, 2)
    Ann1 = _ann(tensor_type, qd_dtype, 1)

    @qd.kernel
    def run(mat_arr: Ann2, vec_arr: Ann1, out_arr: Ann2):
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = qd.simt.Tile16x16(
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
                dtype=qd_dtype,
            )
            t -= outer(vec_arr[tid], vec_arr[tid])
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(123)
    R = rng.randn(N, N).astype(np_dtype)
    v = rng.randn(N).astype(np_dtype)
    mat.from_numpy(R)
    vec.from_numpy(v)
    run(mat, vec, out)
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(v, v), atol=1e-5)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_ger_sub(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    mat = tensor_type(qd_dtype, (N, N))
    vec_a = tensor_type(qd_dtype, (N,))
    vec_b = tensor_type(qd_dtype, (N,))
    out = tensor_type(qd_dtype, (N, N))

    Ann2 = _ann(tensor_type, qd_dtype, 2)
    Ann1 = _ann(tensor_type, qd_dtype, 1)

    @qd.kernel
    def run(
        mat_arr: Ann2,
        va_arr: Ann1,
        vb_arr: Ann1,
        out_arr: Ann2,
    ):
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = qd.simt.Tile16x16(
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
                dtype=qd_dtype,
            )
            t -= outer(va_arr[tid], vb_arr[tid])
            out_arr[0:N, 0:N] = t

    rng = np.random.RandomState(456)
    R = rng.randn(N, N).astype(np_dtype)
    a = rng.randn(N).astype(np_dtype)
    b = rng.randn(N).astype(np_dtype)
    mat.from_numpy(R)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    run(mat, vec_a, vec_b, out)
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(a, b), atol=1e-5)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_potrf(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    atol = _ATOLS[qd_dtype]
    src = tensor_type(qd_dtype, (N, N))
    dst = tensor_type(qd_dtype, (N, N))
    eps_field = qd.field(dtype=qd_dtype, shape=())

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            t[:] = src_arr[0:N, 0:N]
            t.cholesky_(eps_field[None])
            dst_arr[0:N, 0:N] = t

    A = _make_spd(dtype=np_dtype)
    src.from_numpy(A)
    eps_field[None] = _EPS_VALS[qd_dtype]
    run(src, dst)
    L_expected = np.linalg.cholesky(A.astype(np.float64)).astype(np_dtype)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=atol)


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_trsm(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    atol = _ATOLS[qd_dtype]
    l_field = tensor_type(qd_dtype, (N, N))
    b_field = tensor_type(qd_dtype, (N, N))
    x_field = tensor_type(qd_dtype, (N, N))

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(l_arr: Ann, b_arr: Ann, x_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            L[:] = l_arr[0:N, 0:N]
            B = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            B[:] = b_arr[0:N, 0:N]
            L.solve_triangular_(B)
            x_arr[0:N, 0:N] = B

    A = _make_spd(seed=99, dtype=np_dtype)
    Lnp = np.linalg.cholesky(A.astype(np.float64)).astype(np_dtype)
    rng = np.random.RandomState(77)
    Bnp = rng.randn(N, N).astype(np_dtype)
    l_field.from_numpy(Lnp)
    b_field.from_numpy(Bnp)
    run(l_field, b_field, x_field)
    X = x_field.to_numpy()
    np.testing.assert_allclose(X @ Lnp.T, Bnp, atol=max(atol, 1e-3))


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_potrf_then_trsm(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    atol = _ATOLS[qd_dtype]
    a_field = tensor_type(qd_dtype, (N, N))
    b_field = tensor_type(qd_dtype, (N, N))
    x_field = tensor_type(qd_dtype, (N, N))
    eps_field = qd.field(dtype=qd_dtype, shape=())

    Ann = _ann(tensor_type, qd_dtype, 2)

    @qd.kernel
    def run(a_arr: Ann, b_arr: Ann, x_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            L[:] = a_arr[0:N, 0:N]
            L.cholesky_(eps_field[None])
            B = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
            B[:] = b_arr[0:N, 0:N]
            L.solve_triangular_(B)
            x_arr[0:N, 0:N] = B

    A = _make_spd(seed=55, dtype=np_dtype)
    rng = np.random.RandomState(66)
    Bnp = rng.randn(N, N).astype(np_dtype)
    a_field.from_numpy(A)
    b_field.from_numpy(Bnp)
    eps_field[None] = _EPS_VALS[qd_dtype]
    run(a_field, b_field, x_field)
    X = x_field.to_numpy()
    L_ref = np.linalg.cholesky(A.astype(np.float64))
    X_ref = scipy.linalg.solve_triangular(L_ref, Bnp.T.astype(np.float64), lower=True).T.astype(np_dtype)
    np.testing.assert_allclose(X, X_ref, atol=max(atol, 1e-3))


# =============================================================================
# 3D load/store tests
# =============================================================================


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@pytest.mark.parametrize("qd_dtype", _QD_DTYPES)
def test_tile16_load3d_store3d(tensor_type, qd_dtype):
    np_dtype = _NP_DTYPES[qd_dtype]
    N_BATCH = 2
    src = tensor_type(qd_dtype, (N_BATCH, N, N))
    dst = tensor_type(qd_dtype, (N_BATCH, N, N))

    Ann = _ann(tensor_type, qd_dtype, 3)

    @qd.kernel
    def run(src_arr: Ann, dst_arr: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            for i_b in range(N_BATCH):
                t = qd.simt.Tile16x16.zeros(dtype=qd_dtype)
                t[:] = src_arr[i_b, 0:N, 0:N]
                dst_arr[i_b, 0:N, 0:N] = t

    data = np.arange(N_BATCH * N * N, dtype=np_dtype).reshape(N_BATCH, N, N)
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)


# =============================================================================
# SharedArray load/store tests (CUDA only, f32 only — SharedArray dtype is explicit)
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src[0:N, 0:N]
            sh[0:N, 0:N] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src[0:N, 0:NCOLS]
            sh[0:N, 0:NCOLS] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src[0:N, 0:N]
            t.cholesky_(eps_field[None])
            sh[0:N, 0:N] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src[0:N, 0:N]
            sh[0:N, 0:NCOLS] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t_load = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t_load[:] = src[0:N, 0:N]
            sh[0:N, 0:N] = t_load
            qd.simt.block.sync()
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = src[0:N, 0:N]
            sh[0:N, 0:N] = t
            qd.simt.block.sync()
            t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t_load = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t_load[:] = src[0:N, 0:NCOLS]
            sh[0:N, 0:NCOLS] = t_load
            qd.simt.block.sync()
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[K0:K0 + qd.simt.Tile16x16.SIZE, COL]
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:N, 0:N]
            v = vecs_arr[1, K0:K0 + qd.simt.Tile16x16.SIZE, COL]
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:N, 0:N]
            a = vecs_arr[K0_A:K0_A + qd.simt.Tile16x16.SIZE, COL]
            b = vecs_arr[K0_B:K0_B + qd.simt.Tile16x16.SIZE, COL]
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat[0:N, 0:N]
            v = sh[K0:K0 + qd.simt.Tile16x16.SIZE, COL]
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
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
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = mat_arr[0:N, 0:N]
            for c in range(NCOLS):
                v = vecs_arr[K0:K0 + qd.simt.Tile16x16.SIZE, c]
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
# f64 tile on f32 array roundtrip — must be lossless
# =============================================================================


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_tile16_f64_roundtrip_into_f32_array():
    """Load f32 data through an f64 tile and store back — must be lossless."""
    src = qd.ndarray(shape=(N, N), dtype=qd.f32)
    dst_f32 = qd.ndarray(shape=(N, N), dtype=qd.f32)
    dst_f64 = qd.ndarray(shape=(N, N), dtype=qd.f32)

    Ann = qd.types.NDArray[qd.f32, 2]

    @qd.kernel
    def roundtrip_f32(s: Ann, d: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
            t[:] = s[0:N, 0:N]
            d[0:N, 0:N] = t

    @qd.kernel
    def roundtrip_f64(s: Ann, d: Ann):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.zeros(dtype=qd.f64)
            t[:] = s[0:N, 0:N]
            d[0:N, 0:N] = t

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)

    roundtrip_f32(src, dst_f32)
    roundtrip_f64(src, dst_f64)

    np.testing.assert_array_equal(dst_f32.to_numpy(), data)
    np.testing.assert_array_equal(dst_f64.to_numpy(), data)


# =============================================================================
# Dtype mismatch perf regression test (kept for hardware where f64 is slower)
# =============================================================================

_ENVS = 4096
_NDOFS = 48
_N_BLOCKS = (_NDOFS + N - 1) // N


def _make_cholesky_kernel(qd_dtype):
    """Build a tiled Cholesky kernel using qd.simt.Tile16x16 with the given dtype."""

    @qd.kernel
    def cholesky_tiled(H: qd.types.NDArray[qd.f32, 3], eps_arr: qd.types.NDArray[qd.f32, 1]):
        n_dofs = H.shape[1]
        _B = H.shape[0]
        EPS = eps_arr[0]

        qd.loop_config(block_dim=N)
        for i in range(_B * N):
            tid = i % N
            i_b = i // N
            if i_b >= _B:
                continue

            for kb in range(_N_BLOCKS):
                k0 = kb * N

                L_kk = qd.simt.Tile16x16(dtype=qd_dtype)
                if k0 + tid < n_dofs:
                    L_kk[:] = H[i_b, k0 : k0 + N, k0:n_dofs]
                else:
                    L_kk.eye_()

                for jb in range(kb):
                    j0 = jb * N
                    for t in range(N):
                        v = H[i_b, k0:n_dofs, j0 + t]
                        L_kk -= outer(v, v)

                L_kk.cholesky_(EPS)

                for ib in range(kb + 1, _N_BLOCKS):
                    i0 = ib * N

                    L_ik = qd.simt.Tile16x16(dtype=qd_dtype)
                    if i0 + tid < n_dofs:
                        L_ik[:] = H[i_b, i0 : i0 + N, k0:n_dofs]

                    for jb in range(kb):
                        j0 = jb * N
                        for t in range(N):
                            v_own = H[i_b, i0:n_dofs, j0 + t]
                            v_diag = H[i_b, k0:n_dofs, j0 + t]
                            L_ik -= outer(v_own, v_diag)

                    L_kk.solve_triangular_(L_ik)

                    if i0 + tid < n_dofs:
                        H[i_b, i0 : i0 + N, k0:n_dofs] = L_ik

                if k0 + tid < n_dofs:
                    H[i_b, k0 : k0 + N, k0:n_dofs] = L_kk

    return cholesky_tiled


@test_utils.test(arch=qd.cuda)
def test_tile16_f64_on_f32_arrays_perf_regression():
    """f64 Tile16x16 on f32 arrays must not be silently slower than f32 tile.

    This reproduces the g1_fall regression where the solver captured an f64
    tile at import time, causing all subsequent GPU kernels to use f64 tile
    registers on f32 data — doubling register pressure and halving throughput.
    """
    kernel_f32 = _make_cholesky_kernel(qd.f32)
    kernel_f64 = _make_cholesky_kernel(qd.f64)

    rng = np.random.RandomState(42)
    B_np = rng.randn(_ENVS, _NDOFS, _NDOFS).astype(np.float32)
    H_np = np.einsum("bij,bkj->bik", B_np, B_np) + _NDOFS * np.eye(_NDOFS, dtype=np.float32)

    H_f32 = qd.ndarray(qd.f32, (_ENVS, _NDOFS, _NDOFS))
    H_f64_on_f32 = qd.ndarray(qd.f32, (_ENVS, _NDOFS, _NDOFS))
    eps_arr = qd.ndarray(qd.f32, (1,))
    eps_arr.from_numpy(np.array([1e-6], dtype=np.float32))

    H_f32.from_numpy(H_np.copy())
    kernel_f32(H_f32, eps_arr)
    result_f32 = H_f32.to_numpy()

    H_f64_on_f32.from_numpy(H_np.copy())
    kernel_f64(H_f64_on_f32, eps_arr)
    result_f64 = H_f64_on_f32.to_numpy()

    for b in [0, 1, _ENVS - 1]:
        L32 = np.tril(result_f32[b])
        L64 = np.tril(result_f64[b])
        np.testing.assert_allclose(L32, L64, atol=1e-3,
                                   err_msg=f"Cholesky mismatch at batch {b}")

    REPS = 20
    qd.sync()
    H_f32.from_numpy(H_np.copy())
    kernel_f32(H_f32, eps_arr)
    qd.sync()

    t0 = time.perf_counter()
    for _ in range(REPS):
        H_f32.from_numpy(H_np.copy())
        kernel_f32(H_f32, eps_arr)
    qd.sync()
    ms_f32 = (time.perf_counter() - t0) / REPS * 1000

    H_f64_on_f32.from_numpy(H_np.copy())
    kernel_f64(H_f64_on_f32, eps_arr)
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
# Proxy API specific tests
# =============================================================================


@test_utils.test(arch=qd.cuda)
def test_proxy_size_constant():
    assert qd.simt.Tile16x16.SIZE == 16


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


@test_utils.test(arch=qd.cuda)
def test_proxy_default_dtype_survives_reinit():
    """Proxy with default dtype must follow default_fp across init/reset cycles.

    This is the actual regression scenario: init with f64, compile a kernel,
    reset, reinit with f32, compile the same kernel pattern — the second
    kernel must use f32 tiles, not stale f64.
    """
    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

    @qd.kernel
    def write_eye_f64(dst: qd.types.NDArray[qd.f64, 2]):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.eye()
            dst[0:N, 0:N] = t

    # Phase 1: switch to f64, compile and run
    impl.get_runtime().set_default_fp(qd.f64)
    dst64 = qd.ndarray(qd.f64, (N, N))
    write_eye_f64(dst64)
    np.testing.assert_allclose(dst64.to_numpy(), np.eye(N))

    # Phase 2: reset, reinit with f32
    qd.reset()
    qd.init(arch=qd.cuda, default_fp=qd.f32)

    @qd.kernel
    def write_eye_f32(dst: qd.types.NDArray[qd.f32, 2]):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = qd.simt.Tile16x16.eye()
            dst[0:N, 0:N] = t

    dst32 = qd.ndarray(qd.f32, (N, N))
    write_eye_f32(dst32)
    result32 = dst32.to_numpy()

    np.testing.assert_allclose(result32, np.eye(N, dtype=np.float32))
    assert result32.dtype == np.float32
