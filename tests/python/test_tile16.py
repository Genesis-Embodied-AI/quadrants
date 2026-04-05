import numpy as np
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
# Tile16 class API tests
# =============================================================================


@test_utils.test(arch=qd.cuda)
def test_tile16_load_store():
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16()
            t.load(src, tid, 0, N)
            t.store(dst, tid, 0, N)

    data = np.arange(N * N, dtype=np.float32).reshape(N, N)
    src.from_numpy(data)
    run()
    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=qd.cuda)
def test_tile16_load_store_partial():
    NCOLS = 12
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16()
            t.load(src, tid, 0, NCOLS)
            t.store(dst, tid, 0, N)

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)
    run()
    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=qd.cuda)
def test_tile16_syr_sub():
    mat = qd.field(dtype=qd.f32, shape=(N, N))
    vec = qd.field(dtype=qd.f32, shape=(N,))
    out = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16(
                mat[tid, 0],
                mat[tid, 1],
                mat[tid, 2],
                mat[tid, 3],
                mat[tid, 4],
                mat[tid, 5],
                mat[tid, 6],
                mat[tid, 7],
                mat[tid, 8],
                mat[tid, 9],
                mat[tid, 10],
                mat[tid, 11],
                mat[tid, 12],
                mat[tid, 13],
                mat[tid, 14],
                mat[tid, 15],
            )
            t.syr_sub(vec[tid])
            t.store(out, tid, 0, N)

    rng = np.random.RandomState(123)
    R = rng.randn(N, N).astype(np.float32)
    v = rng.randn(N).astype(np.float32)
    mat.from_numpy(R)
    vec.from_numpy(v)
    run()
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(v, v), atol=1e-5)


@test_utils.test(arch=qd.cuda)
def test_tile16_ger_sub():
    mat = qd.field(dtype=qd.f32, shape=(N, N))
    vec_a = qd.field(dtype=qd.f32, shape=(N,))
    vec_b = qd.field(dtype=qd.f32, shape=(N,))
    out = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16(
                mat[tid, 0],
                mat[tid, 1],
                mat[tid, 2],
                mat[tid, 3],
                mat[tid, 4],
                mat[tid, 5],
                mat[tid, 6],
                mat[tid, 7],
                mat[tid, 8],
                mat[tid, 9],
                mat[tid, 10],
                mat[tid, 11],
                mat[tid, 12],
                mat[tid, 13],
                mat[tid, 14],
                mat[tid, 15],
            )
            t.ger_sub(vec_a[tid], vec_b[tid])
            t.store(out, tid, 0, N)

    rng = np.random.RandomState(456)
    R = rng.randn(N, N).astype(np.float32)
    a = rng.randn(N).astype(np.float32)
    b = rng.randn(N).astype(np.float32)
    mat.from_numpy(R)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)
    run()
    np.testing.assert_allclose(out.to_numpy(), R - np.outer(a, b), atol=1e-5)


@test_utils.test(arch=qd.cuda)
def test_tile16_potrf():
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))
    eps_field = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16(
                src[tid, 0],
                src[tid, 1],
                src[tid, 2],
                src[tid, 3],
                src[tid, 4],
                src[tid, 5],
                src[tid, 6],
                src[tid, 7],
                src[tid, 8],
                src[tid, 9],
                src[tid, 10],
                src[tid, 11],
                src[tid, 12],
                src[tid, 13],
                src[tid, 14],
                src[tid, 15],
            )
            t.potrf(tid, eps_field[None])
            t.store(dst, tid, 0, N)

    A = _make_spd()
    src.from_numpy(A)
    eps_field[None] = 1e-10
    run()
    L_expected = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=1e-4)


@test_utils.test(arch=qd.cuda)
def test_tile16_trsm():
    l_field = qd.field(dtype=qd.f32, shape=(N, N))
    b_field = qd.field(dtype=qd.f32, shape=(N, N))
    x_field = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            L = Tile16(
                l_field[tid, 0],
                l_field[tid, 1],
                l_field[tid, 2],
                l_field[tid, 3],
                l_field[tid, 4],
                l_field[tid, 5],
                l_field[tid, 6],
                l_field[tid, 7],
                l_field[tid, 8],
                l_field[tid, 9],
                l_field[tid, 10],
                l_field[tid, 11],
                l_field[tid, 12],
                l_field[tid, 13],
                l_field[tid, 14],
                l_field[tid, 15],
            )
            B = Tile16(
                b_field[tid, 0],
                b_field[tid, 1],
                b_field[tid, 2],
                b_field[tid, 3],
                b_field[tid, 4],
                b_field[tid, 5],
                b_field[tid, 6],
                b_field[tid, 7],
                b_field[tid, 8],
                b_field[tid, 9],
                b_field[tid, 10],
                b_field[tid, 11],
                b_field[tid, 12],
                b_field[tid, 13],
                b_field[tid, 14],
                b_field[tid, 15],
            )
            B.trsm(L)
            B.store(x_field, tid, 0, N)

    A = _make_spd(seed=99)
    Lnp = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    rng = np.random.RandomState(77)
    Bnp = rng.randn(N, N).astype(np.float32)
    l_field.from_numpy(Lnp)
    b_field.from_numpy(Bnp)
    run()
    X = x_field.to_numpy()
    np.testing.assert_allclose(X @ Lnp.T, Bnp, atol=1e-3)


@test_utils.test(arch=qd.cuda)
def test_tile16_potrf_then_trsm():
    a_field = qd.field(dtype=qd.f32, shape=(N, N))
    b_field = qd.field(dtype=qd.f32, shape=(N, N))
    x_field = qd.field(dtype=qd.f32, shape=(N, N))
    eps_field = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            L = Tile16(
                a_field[tid, 0],
                a_field[tid, 1],
                a_field[tid, 2],
                a_field[tid, 3],
                a_field[tid, 4],
                a_field[tid, 5],
                a_field[tid, 6],
                a_field[tid, 7],
                a_field[tid, 8],
                a_field[tid, 9],
                a_field[tid, 10],
                a_field[tid, 11],
                a_field[tid, 12],
                a_field[tid, 13],
                a_field[tid, 14],
                a_field[tid, 15],
            )
            L.potrf(tid, eps_field[None])
            B = Tile16(
                b_field[tid, 0],
                b_field[tid, 1],
                b_field[tid, 2],
                b_field[tid, 3],
                b_field[tid, 4],
                b_field[tid, 5],
                b_field[tid, 6],
                b_field[tid, 7],
                b_field[tid, 8],
                b_field[tid, 9],
                b_field[tid, 10],
                b_field[tid, 11],
                b_field[tid, 12],
                b_field[tid, 13],
                b_field[tid, 14],
                b_field[tid, 15],
            )
            B.trsm(L)
            B.store(x_field, tid, 0, N)

    A = _make_spd(seed=55)
    rng = np.random.RandomState(66)
    Bnp = rng.randn(N, N).astype(np.float32)
    a_field.from_numpy(A)
    b_field.from_numpy(Bnp)
    eps_field[None] = 1e-10
    run()
    X = x_field.to_numpy()
    L_ref = np.linalg.cholesky(A.astype(np.float64))
    X_ref = scipy.linalg.solve_triangular(L_ref, Bnp.T.astype(np.float64), lower=True).T.astype(np.float32)
    np.testing.assert_allclose(X, X_ref, atol=1e-3)


# =============================================================================
# f64 precision tests — verify make_tile16(qd.f64) preserves double precision
# =============================================================================


@test_utils.test(arch=qd.cuda)
def test_tile16_f64_load_store():
    src = qd.field(dtype=qd.f64, shape=(N, N))
    dst = qd.field(dtype=qd.f64, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16_f64()
            t.load(src, tid, 0, N)
            t.store(dst, tid, 0, N)

    data = np.arange(N * N, dtype=np.float64).reshape(N, N)
    src.from_numpy(data)
    run()
    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=qd.cuda)
def test_tile16_f64_potrf():
    src = qd.field(dtype=qd.f64, shape=(N, N))
    dst = qd.field(dtype=qd.f64, shape=(N, N))
    eps_field = qd.field(dtype=qd.f64, shape=())

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            t = Tile16_f64()
            t.load(src, tid, 0, N)
            t.potrf(tid, eps_field[None])
            t.store(dst, tid, 0, N)

    A = _make_spd(dtype=np.float64)
    src.from_numpy(A)
    eps_field[None] = 1e-30
    run()
    L_expected = np.linalg.cholesky(A)
    np.testing.assert_allclose(np.tril(dst.to_numpy()), L_expected, atol=1e-12)


@test_utils.test(arch=qd.cuda)
def test_tile16_f64_potrf_then_trsm():
    a_field = qd.field(dtype=qd.f64, shape=(N, N))
    b_field = qd.field(dtype=qd.f64, shape=(N, N))
    x_field = qd.field(dtype=qd.f64, shape=(N, N))
    eps_field = qd.field(dtype=qd.f64, shape=())

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            L = Tile16_f64()
            L.load(a_field, tid, 0, N)
            L.potrf(tid, eps_field[None])
            B = Tile16_f64()
            B.load(b_field, tid, 0, N)
            B.trsm(L)
            B.store(x_field, tid, 0, N)

    A = _make_spd(seed=55, dtype=np.float64)
    rng = np.random.RandomState(66)
    Bnp = rng.randn(N, N).astype(np.float64)
    a_field.from_numpy(A)
    b_field.from_numpy(Bnp)
    eps_field[None] = 1e-30
    run()
    X = x_field.to_numpy()
    L_ref = np.linalg.cholesky(A)
    X_ref = scipy.linalg.solve_triangular(L_ref, Bnp.T, lower=True).T
    np.testing.assert_allclose(X, X_ref, atol=1e-12)
