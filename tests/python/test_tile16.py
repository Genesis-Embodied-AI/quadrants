import numpy as np
import pytest
import scipy.linalg

import quadrants as qd
from quadrants.lang.simt.tile16 import Tile16, make_tile16

from tests import test_utils

N = 16

Tile16_f64 = make_tile16(qd.f64)


def _make_spd(seed=42, dtype=np.float32):
    rng = np.random.RandomState(seed)
    B = rng.randn(N, N).astype(np.float64)
    return (B @ B.T + N * np.eye(N)).astype(dtype)


# =============================================================================
# Tile16 class API tests (field + ndarray)
# =============================================================================


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_load_store(tensor_type):
    src = tensor_type(qd.f32, (N, N))
    dst = tensor_type(qd.f32, (N, N))

    @qd.kernel
    def run(src_arr: qd.template(), dst_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16()
            t.load(src_arr, 0, 0, N)
            t.store(dst_arr, 0, 0, N)

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

    @qd.kernel
    def run(src_arr: qd.template(), dst_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16()
            t.load(src_arr, 0, 0, NCOLS)
            t.store(dst_arr, 0, 0, N)

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run(src, dst)
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_syr_sub(tensor_type):
    mat = tensor_type(qd.f32, (N, N))
    vec = tensor_type(qd.f32, (N,))
    out = tensor_type(qd.f32, (N, N))

    @qd.kernel
    def run(mat_arr: qd.template(), vec_arr: qd.template(), out_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16(
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
            t -= qd.outer(vec_arr[tid], vec_arr[tid])
            t.store(out_arr, 0, 0, N)

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

    @qd.kernel
    def run(
        mat_arr: qd.template(),
        va_arr: qd.template(),
        vb_arr: qd.template(),
        out_arr: qd.template(),
    ):
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16(
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
            t -= qd.outer(va_arr[tid], vb_arr[tid])
            t.store(out_arr, 0, 0, N)

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

    @qd.kernel
    def run(src_arr: qd.template(), dst_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16()
            t.load(src_arr, 0, 0, N)
            t.cholesky_(eps_field[None])
            t.store(dst_arr, 0, 0, N)

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

    @qd.kernel
    def run(l_arr: qd.template(), b_arr: qd.template(), x_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = Tile16()
            L.load(l_arr, 0, 0, N)
            B = Tile16()
            B.load(b_arr, 0, 0, N)
            L.solve_triangular_(B)
            B.store(x_arr, 0, 0, N)

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

    @qd.kernel
    def run(a_arr: qd.template(), b_arr: qd.template(), x_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = Tile16()
            L.load(a_arr, 0, 0, N)
            L.cholesky_(eps_field[None])
            B = Tile16()
            B.load(b_arr, 0, 0, N)
            L.solve_triangular_(B)
            B.store(x_arr, 0, 0, N)

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
# f64 precision tests — verify make_tile16(qd.f64) preserves double precision
# =============================================================================


@test_utils.test(arch=qd.cuda)
@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
def test_tile16_f64_load_store(tensor_type):
    src = tensor_type(qd.f64, (N, N))
    dst = tensor_type(qd.f64, (N, N))

    @qd.kernel
    def run(src_arr: qd.template(), dst_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16_f64()
            t.load(src_arr, 0, 0, N)
            t.store(dst_arr, 0, 0, N)

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

    @qd.kernel
    def run(src_arr: qd.template(), dst_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            t = Tile16_f64()
            t.load(src_arr, 0, 0, N)
            t.cholesky_(eps_field[None])
            t.store(dst_arr, 0, 0, N)

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

    @qd.kernel
    def run(a_arr: qd.template(), b_arr: qd.template(), x_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            L = Tile16_f64()
            L.load(a_arr, 0, 0, N)
            L.cholesky_(eps_field[None])
            B = Tile16_f64()
            B.load(b_arr, 0, 0, N)
            L.solve_triangular_(B)
            B.store(x_arr, 0, 0, N)

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

    @qd.kernel
    def run(src_arr: qd.template(), dst_arr: qd.template()):
        qd.loop_config(block_dim=N)
        for _ in range(N):
            for i_b in range(N_BATCH):
                t = Tile16()
                t.load3d(src_arr, i_b, 0, 0, N)
                t.store3d(dst_arr, i_b, 0, 0, N)

    data = np.arange(N_BATCH * N * N, dtype=np.float32).reshape(N_BATCH, N, N)
    src.from_numpy(data)
    run(src, dst)
    np.testing.assert_allclose(dst.to_numpy(), data)
