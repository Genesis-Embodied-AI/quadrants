import numpy as np
import pytest
import scipy.linalg

import quadrants as qd
from quadrants.lang.simt import tile16

from tests import test_utils

N = 16


def _make_spd(seed=42):
    rng = np.random.RandomState(seed)
    B = rng.randn(N, N).astype(np.float64)
    return (B @ B.T + N * np.eye(N)).astype(np.float32)


@test_utils.test(arch=qd.cuda)
def test_load_store_roundtrip():
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def roundtrip():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            z = qd.f32(0.0)
            r0 = z; r1 = z; r2 = z; r3 = z; r4 = z; r5 = z; r6 = z; r7 = z
            r8 = z; r9 = z; r10 = z; r11 = z; r12 = z; r13 = z; r14 = z; r15 = z
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 = \
                tile16.load(src, tid, 0, N,
                            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            tile16.store(dst, tid, 0, N,
                         r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)

    data = np.arange(N * N, dtype=np.float32).reshape(N, N)
    src.from_numpy(data)

    roundtrip()

    np.testing.assert_allclose(dst.to_numpy(), data)


@test_utils.test(arch=qd.cuda)
def test_load_store_partial():
    """Partial tile: only first 12 columns should be loaded; rest stay zero."""
    NCOLS = 12
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def partial():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            z = qd.f32(0.0)
            r0 = z; r1 = z; r2 = z; r3 = z; r4 = z; r5 = z; r6 = z; r7 = z
            r8 = z; r9 = z; r10 = z; r11 = z; r12 = z; r13 = z; r14 = z; r15 = z
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 = \
                tile16.load(src, tid, 0, NCOLS,
                            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)
            tile16.store(dst, tid, 0, N,
                         r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15)

    data = np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0
    src.from_numpy(data)

    partial()

    result = dst.to_numpy()
    np.testing.assert_allclose(result[:, :NCOLS], data[:, :NCOLS])
    np.testing.assert_allclose(result[:, NCOLS:], 0.0)


@test_utils.test(arch=qd.cuda)
def test_syr_sub():
    mat = qd.field(dtype=qd.f32, shape=(N, N))
    vec = qd.field(dtype=qd.f32, shape=(N,))
    out = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            v = vec[tid]
            r0 = mat[tid, 0]; r1 = mat[tid, 1]; r2 = mat[tid, 2]; r3 = mat[tid, 3]
            r4 = mat[tid, 4]; r5 = mat[tid, 5]; r6 = mat[tid, 6]; r7 = mat[tid, 7]
            r8 = mat[tid, 8]; r9 = mat[tid, 9]; r10 = mat[tid, 10]; r11 = mat[tid, 11]
            r12 = mat[tid, 12]; r13 = mat[tid, 13]; r14 = mat[tid, 14]; r15 = mat[tid, 15]
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 = \
                tile16.syr_sub(v, r0, r1, r2, r3, r4, r5, r6, r7,
                               r8, r9, r10, r11, r12, r13, r14, r15)
            out[tid, 0] = r0; out[tid, 1] = r1; out[tid, 2] = r2; out[tid, 3] = r3
            out[tid, 4] = r4; out[tid, 5] = r5; out[tid, 6] = r6; out[tid, 7] = r7
            out[tid, 8] = r8; out[tid, 9] = r9; out[tid, 10] = r10; out[tid, 11] = r11
            out[tid, 12] = r12; out[tid, 13] = r13; out[tid, 14] = r14; out[tid, 15] = r15

    rng = np.random.RandomState(123)
    R = rng.randn(N, N).astype(np.float32)
    v = rng.randn(N).astype(np.float32)
    mat.from_numpy(R)
    vec.from_numpy(v)

    run()

    expected = R - np.outer(v, v)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-5)


@test_utils.test(arch=qd.cuda)
def test_ger_sub():
    mat = qd.field(dtype=qd.f32, shape=(N, N))
    vec_a = qd.field(dtype=qd.f32, shape=(N,))
    vec_b = qd.field(dtype=qd.f32, shape=(N,))
    out = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            a = vec_a[tid]
            b = vec_b[tid]
            r0 = mat[tid, 0]; r1 = mat[tid, 1]; r2 = mat[tid, 2]; r3 = mat[tid, 3]
            r4 = mat[tid, 4]; r5 = mat[tid, 5]; r6 = mat[tid, 6]; r7 = mat[tid, 7]
            r8 = mat[tid, 8]; r9 = mat[tid, 9]; r10 = mat[tid, 10]; r11 = mat[tid, 11]
            r12 = mat[tid, 12]; r13 = mat[tid, 13]; r14 = mat[tid, 14]; r15 = mat[tid, 15]
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 = \
                tile16.ger_sub(a, b, r0, r1, r2, r3, r4, r5, r6, r7,
                               r8, r9, r10, r11, r12, r13, r14, r15)
            out[tid, 0] = r0; out[tid, 1] = r1; out[tid, 2] = r2; out[tid, 3] = r3
            out[tid, 4] = r4; out[tid, 5] = r5; out[tid, 6] = r6; out[tid, 7] = r7
            out[tid, 8] = r8; out[tid, 9] = r9; out[tid, 10] = r10; out[tid, 11] = r11
            out[tid, 12] = r12; out[tid, 13] = r13; out[tid, 14] = r14; out[tid, 15] = r15

    rng = np.random.RandomState(456)
    R = rng.randn(N, N).astype(np.float32)
    a = rng.randn(N).astype(np.float32)
    b = rng.randn(N).astype(np.float32)
    mat.from_numpy(R)
    vec_a.from_numpy(a)
    vec_b.from_numpy(b)

    run()

    expected = R - np.outer(a, b)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-5)


@test_utils.test(arch=qd.cuda)
def test_potrf():
    src = qd.field(dtype=qd.f32, shape=(N, N))
    dst = qd.field(dtype=qd.f32, shape=(N, N))
    eps_field = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            eps = eps_field[None]
            r0 = src[tid, 0]; r1 = src[tid, 1]; r2 = src[tid, 2]; r3 = src[tid, 3]
            r4 = src[tid, 4]; r5 = src[tid, 5]; r6 = src[tid, 6]; r7 = src[tid, 7]
            r8 = src[tid, 8]; r9 = src[tid, 9]; r10 = src[tid, 10]; r11 = src[tid, 11]
            r12 = src[tid, 12]; r13 = src[tid, 13]; r14 = src[tid, 14]; r15 = src[tid, 15]
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 = \
                tile16.potrf(tid, eps, r0, r1, r2, r3, r4, r5, r6, r7,
                             r8, r9, r10, r11, r12, r13, r14, r15)
            dst[tid, 0] = r0; dst[tid, 1] = r1; dst[tid, 2] = r2; dst[tid, 3] = r3
            dst[tid, 4] = r4; dst[tid, 5] = r5; dst[tid, 6] = r6; dst[tid, 7] = r7
            dst[tid, 8] = r8; dst[tid, 9] = r9; dst[tid, 10] = r10; dst[tid, 11] = r11
            dst[tid, 12] = r12; dst[tid, 13] = r13; dst[tid, 14] = r14; dst[tid, 15] = r15

    A = _make_spd()
    src.from_numpy(A)
    eps_field[None] = 1e-10

    run()

    result = dst.to_numpy()
    L_expected = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)

    # potrf only writes the lower triangle; compare that part
    L_result = np.tril(result)
    np.testing.assert_allclose(L_result, L_expected, atol=1e-4)


@test_utils.test(arch=qd.cuda)
def test_trsm():
    """Verify trsm solves Q @ L^T = B, i.e. Q = B @ inv(L^T)."""
    l_field = qd.field(dtype=qd.f32, shape=(N, N))
    b_field = qd.field(dtype=qd.f32, shape=(N, N))
    x_field = qd.field(dtype=qd.f32, shape=(N, N))

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            r0 = l_field[tid, 0]; r1 = l_field[tid, 1]; r2 = l_field[tid, 2]; r3 = l_field[tid, 3]
            r4 = l_field[tid, 4]; r5 = l_field[tid, 5]; r6 = l_field[tid, 6]; r7 = l_field[tid, 7]
            r8 = l_field[tid, 8]; r9 = l_field[tid, 9]; r10 = l_field[tid, 10]; r11 = l_field[tid, 11]
            r12 = l_field[tid, 12]; r13 = l_field[tid, 13]; r14 = l_field[tid, 14]; r15 = l_field[tid, 15]
            q0 = b_field[tid, 0]; q1 = b_field[tid, 1]; q2 = b_field[tid, 2]; q3 = b_field[tid, 3]
            q4 = b_field[tid, 4]; q5 = b_field[tid, 5]; q6 = b_field[tid, 6]; q7 = b_field[tid, 7]
            q8 = b_field[tid, 8]; q9 = b_field[tid, 9]; q10 = b_field[tid, 10]; q11 = b_field[tid, 11]
            q12 = b_field[tid, 12]; q13 = b_field[tid, 13]; q14 = b_field[tid, 14]; q15 = b_field[tid, 15]
            q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15 = \
                tile16.trsm(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
                            q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15)
            x_field[tid, 0] = q0; x_field[tid, 1] = q1; x_field[tid, 2] = q2; x_field[tid, 3] = q3
            x_field[tid, 4] = q4; x_field[tid, 5] = q5; x_field[tid, 6] = q6; x_field[tid, 7] = q7
            x_field[tid, 8] = q8; x_field[tid, 9] = q9; x_field[tid, 10] = q10; x_field[tid, 11] = q11
            x_field[tid, 12] = q12; x_field[tid, 13] = q13; x_field[tid, 14] = q14; x_field[tid, 15] = q15

    A = _make_spd(seed=99)
    L = np.linalg.cholesky(A.astype(np.float64)).astype(np.float32)
    rng = np.random.RandomState(77)
    B = rng.randn(N, N).astype(np.float32)

    l_field.from_numpy(L)
    b_field.from_numpy(B)

    run()

    X = x_field.to_numpy()

    # trsm solves Q @ L^T = B, so X @ L^T should reconstruct B
    reconstructed = X @ L.T
    np.testing.assert_allclose(reconstructed, B, atol=1e-3)


@test_utils.test(arch=qd.cuda)
def test_potrf_then_trsm():
    """End-to-end: factorize SPD matrix, then solve a linear system."""
    a_field = qd.field(dtype=qd.f32, shape=(N, N))
    b_field = qd.field(dtype=qd.f32, shape=(N, N))
    x_field = qd.field(dtype=qd.f32, shape=(N, N))
    eps_field = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        qd.loop_config(block_dim=N)
        for tid in range(N):
            eps = eps_field[None]
            # Load A (will become L after potrf)
            r0 = a_field[tid, 0]; r1 = a_field[tid, 1]; r2 = a_field[tid, 2]; r3 = a_field[tid, 3]
            r4 = a_field[tid, 4]; r5 = a_field[tid, 5]; r6 = a_field[tid, 6]; r7 = a_field[tid, 7]
            r8 = a_field[tid, 8]; r9 = a_field[tid, 9]; r10 = a_field[tid, 10]; r11 = a_field[tid, 11]
            r12 = a_field[tid, 12]; r13 = a_field[tid, 13]; r14 = a_field[tid, 14]; r15 = a_field[tid, 15]
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15 = \
                tile16.potrf(tid, eps, r0, r1, r2, r3, r4, r5, r6, r7,
                             r8, r9, r10, r11, r12, r13, r14, r15)
            # Load B
            q0 = b_field[tid, 0]; q1 = b_field[tid, 1]; q2 = b_field[tid, 2]; q3 = b_field[tid, 3]
            q4 = b_field[tid, 4]; q5 = b_field[tid, 5]; q6 = b_field[tid, 6]; q7 = b_field[tid, 7]
            q8 = b_field[tid, 8]; q9 = b_field[tid, 9]; q10 = b_field[tid, 10]; q11 = b_field[tid, 11]
            q12 = b_field[tid, 12]; q13 = b_field[tid, 13]; q14 = b_field[tid, 14]; q15 = b_field[tid, 15]
            # Solve Q @ L^T = B
            q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15 = \
                tile16.trsm(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
                            q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15)
            x_field[tid, 0] = q0; x_field[tid, 1] = q1; x_field[tid, 2] = q2; x_field[tid, 3] = q3
            x_field[tid, 4] = q4; x_field[tid, 5] = q5; x_field[tid, 6] = q6; x_field[tid, 7] = q7
            x_field[tid, 8] = q8; x_field[tid, 9] = q9; x_field[tid, 10] = q10; x_field[tid, 11] = q11
            x_field[tid, 12] = q12; x_field[tid, 13] = q13; x_field[tid, 14] = q14; x_field[tid, 15] = q15

    A = _make_spd(seed=55)
    rng = np.random.RandomState(66)
    B = rng.randn(N, N).astype(np.float32)

    a_field.from_numpy(A)
    b_field.from_numpy(B)
    eps_field[None] = 1e-10

    run()

    X = x_field.to_numpy()
    L_ref = np.linalg.cholesky(A.astype(np.float64))

    # X @ L^T = B  =>  X = B @ inv(L^T)  =>  X = solve(L, B^T)^T
    X_ref = scipy.linalg.solve_triangular(L_ref, B.T.astype(np.float64), lower=True).T.astype(np.float32)
    np.testing.assert_allclose(X, X_ref, atol=1e-3)
