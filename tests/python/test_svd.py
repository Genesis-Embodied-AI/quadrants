import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.data64, fast_math=False)
def test_precision():
    u = qd.field(qd.f64, shape=())
    v = qd.field(qd.f64, shape=())
    w = qd.field(qd.f64, shape=())

    @qd.kernel
    def forward():
        v[None] = qd.sqrt(qd.cast(u[None] + 3.25, qd.f64))
        w[None] = qd.cast(u[None] + 7, qd.f64) / qd.cast(u[None] + 3, qd.f64)

    forward()
    assert v[None] ** 2 == test_utils.approx(3.25, abs=1e-12)
    assert w[None] * 3 == test_utils.approx(7, abs=1e-12)


def mat_equal(A, B, tol=1e-6):
    return np.max(np.abs(A - B)) < tol


def _test_svd(dt, n):
    print(
        f"arch={qd.lang.impl.current_cfg().arch} default_fp={qd.lang.impl.current_cfg().default_fp} fast_math={qd.lang.impl.current_cfg().fast_math} dim={n}"
    )
    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    A_reconstructed = qd.Matrix.field(n, n, dtype=dt, shape=())
    U = qd.Matrix.field(n, n, dtype=dt, shape=())
    UtU = qd.Matrix.field(n, n, dtype=dt, shape=())
    sigma = qd.Matrix.field(n, n, dtype=dt, shape=())
    V = qd.Matrix.field(n, n, dtype=dt, shape=())
    VtV = qd.Matrix.field(n, n, dtype=dt, shape=())

    @qd.kernel
    def run():
        U[None], sigma[None], V[None] = qd.svd(A[None], dt)
        UtU[None] = U[None].transpose() @ U[None]
        VtV[None] = V[None].transpose() @ V[None]
        A_reconstructed[None] = U[None] @ sigma[None] @ V[None].transpose()

    if n == 3:
        A[None] = [[1, 1, 3], [9, -3, 2], [-3, 4, 2]]
    else:
        A[None] = [[1, 1], [2, 3]]

    run()

    tol = 1e-5 if dt == qd.f32 else 1e-12

    assert mat_equal(UtU.to_numpy(), np.eye(n), tol=tol)
    assert mat_equal(VtV.to_numpy(), np.eye(n), tol=tol)
    assert mat_equal(A_reconstructed.to_numpy(), A.to_numpy(), tol=tol)
    for i in range(n):
        for j in range(n):
            if i != j:
                assert sigma[None][i, j] == test_utils.approx(0)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_svd_f32(dim):
    _test_svd(qd.f32, dim)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_svd_f64(dim):
    _test_svd(qd.f64, dim)


@test_utils.test()
def test_transpose_no_loop():
    A = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())
    U = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())
    sigma = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())
    V = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        U[None], sigma[None], V[None] = qd.svd(A[None])

    run()
    # As long as it passes compilation we are good


# ---------------------------------------------------------------------------
# 3×3 SVD sign convention
#
# qipc's ARAP rotation `R = U @ V.T` requires that R is a proper rotation (det R = +1) for any input deformation
# gradient F (which can have det of either sign). The libuipc convention used by qipc enforces this via:
#
#   det(U) = det(V) = +1  always; the sign of det(F) is absorbed into σ (so σ may have one negative diagonal entry
#   when det(F) < 0).
#
# The tests below verify this convention on `qd.svd` for 3×3 inputs covering:
#   - generic positive-det matrix, negative-det matrix, symmetric SPD, identity, near-singular (rank-2), and a
#     near-degenerate-singular-values case.
# ---------------------------------------------------------------------------


def _svd3_sign_convention_test_inputs():
    """A list of (label, np.ndarray (3, 3)) test inputs covering interesting sign / rank / conditioning cases."""
    rng = np.random.default_rng(0xA17F)
    # Random with positive det
    A_pos = rng.standard_normal((3, 3))
    if np.linalg.det(A_pos) < 0:
        A_pos[0] = -A_pos[0]
    # Random with negative det (flip a row of a positive-det matrix)
    A_neg = A_pos.copy()
    A_neg[0] = -A_neg[0]
    # Identity (well-conditioned, det = +1)
    A_id = np.eye(3)
    # Reflection (det = -1)
    A_refl = np.diag([1.0, 1.0, -1.0])
    # SPD (symmetric, all-positive eigenvalues)
    M = rng.standard_normal((3, 3))
    A_spd = M @ M.T + 3 * np.eye(3)
    # Near rank-deficient (column 2 ≈ column 0)
    A_rd = rng.standard_normal((3, 3))
    A_rd[:, 1] = A_rd[:, 0] + 1e-3 * rng.standard_normal(3)
    # Near-degenerate singular values (two close)
    A_dg = rng.standard_normal((3, 3))
    U_q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    V_q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    A_dg = U_q @ np.diag([2.0, 2.0 + 1e-4, 0.5]) @ V_q.T
    return [
        ("identity", A_id),
        ("reflection", A_refl),
        ("random_pos_det", A_pos),
        ("random_neg_det", A_neg),
        ("spd", A_spd),
        ("near_rank_def", A_rd),
        ("near_degenerate_svs", A_dg),
    ]


def _test_svd3_sign_convention(dt):
    print(f"arch={qd.lang.impl.current_cfg().arch} default_fp={qd.lang.impl.current_cfg().default_fp}")
    np_dt = np.float32 if dt == qd.f32 else np.float64

    A = qd.Matrix.field(3, 3, dtype=dt, shape=())
    U = qd.Matrix.field(3, 3, dtype=dt, shape=())
    sigma = qd.Matrix.field(3, 3, dtype=dt, shape=())
    V = qd.Matrix.field(3, 3, dtype=dt, shape=())
    R = qd.Matrix.field(3, 3, dtype=dt, shape=())
    detU = qd.field(dt, shape=())
    detV = qd.field(dt, shape=())
    detR = qd.field(dt, shape=())

    @qd.kernel
    def run():
        U[None], sigma[None], V[None] = qd.svd(A[None], dt)
        R[None] = U[None] @ V[None].transpose()
        detU[None] = U[None].determinant()
        detV[None] = V[None].determinant()
        detR[None] = R[None].determinant()

    tol = 5e-5 if dt == qd.f32 else 1e-10

    for label, A_np in _svd3_sign_convention_test_inputs():
        A.from_numpy(A_np.astype(np_dt))
        run()
        det_A = float(np.linalg.det(A_np))
        det_U = float(detU[None])
        det_V = float(detV[None])
        det_R = float(detR[None])
        sigma_diag = np.array([sigma[None][i, i] for i in range(3)])

        # The libuipc / qipc convention: det(U) = det(V) = +1 for every input. The sign of det(A) is absorbed into σ
        # (so σ may have a negative entry when det(A) < 0).
        assert det_U == test_utils.approx(1.0, abs=tol), (
            f"[{label}] det(U) = {det_U}, expected +1 (det(A) = {det_A:+.3f}); " f"σ = {sigma_diag.tolist()}"
        )
        assert det_V == test_utils.approx(1.0, abs=tol), (
            f"[{label}] det(V) = {det_V}, expected +1 (det(A) = {det_A:+.3f}); " f"σ = {sigma_diag.tolist()}"
        )
        # Direct consequence: R = U @ V.T is a proper rotation.
        assert det_R == test_utils.approx(1.0, abs=tol), (
            f"[{label}] det(U @ V.T) = {det_R}, expected +1; " f"det(U)={det_U:+.3e} det(V)={det_V:+.3e}"
        )
        # Reconstruction must still hold (sanity — duplicates _test_svd).
        A_reconstructed = U.to_numpy() @ sigma.to_numpy() @ V.to_numpy().T
        np.testing.assert_allclose(A_reconstructed, A_np, rtol=tol, atol=tol)


@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_svd3_sign_convention_f32():
    _test_svd3_sign_convention(qd.f32)


@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_svd3_sign_convention_f64():
    _test_svd3_sign_convention(qd.f64)


# ---------------------------------------------------------------------------
# Sort-order contract: every shape of qd.svd must return singular values in descending order (matches NumPy / LAPACK's
# `svd`). For 3×3 the convention allows the smallest entry to be negative (Sifakis absorbs det(A)'s sign); the
# descending check is on direct numeric value (S[0] >= S[1] >= S[2]).
# ---------------------------------------------------------------------------


def _svd_sort_order_test_inputs(n, np_dt):
    rng = np.random.default_rng(0x517D + n)
    inputs = []
    inputs.append(("random_pos_det", rng.standard_normal((n, n)).astype(np_dt)))
    A_neg = rng.standard_normal((n, n)).astype(np_dt)
    A_neg[0] = -A_neg[0]
    inputs.append(("random_neg_det", A_neg))
    if n == 3:
        U_q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        V_q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        # Hand-pick a σ that arrives unsorted so the sort actually does work.
        inputs.append(("unsorted_sigma", (U_q @ np.diag([1.0, 5.0, 2.0]) @ V_q.T).astype(np_dt)))
    return inputs


def _test_svd_sort_order(n, dt):
    np_dt = np.float32 if dt == qd.f32 else np.float64
    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    U = qd.Matrix.field(n, n, dtype=dt, shape=())
    sigma = qd.Matrix.field(n, n, dtype=dt, shape=())
    V = qd.Matrix.field(n, n, dtype=dt, shape=())

    @qd.kernel
    def run():
        U[None], sigma[None], V[None] = qd.svd(A[None], dt)

    tol = 5e-5 if dt == qd.f32 else 1e-10
    for label, A_np in _svd_sort_order_test_inputs(n, np_dt):
        A.from_numpy(A_np)
        run()
        S_diag = np.array([sigma[None][i, i] for i in range(n)])
        diffs = np.diff(S_diag)
        assert np.all(diffs <= tol), f"[{label}] qd.svd n={n} S not descending: S = {S_diag.tolist()}"
        # Reconstruction must still hold after the sort + sign fix-up.
        A_reconstructed = U.to_numpy() @ sigma.to_numpy() @ V.to_numpy().T
        np.testing.assert_allclose(A_reconstructed, A_np, rtol=tol, atol=tol)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_svd_sort_order_f32(dim):
    _test_svd_sort_order(dim, qd.f32)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_svd_sort_order_f64(dim):
    _test_svd_sort_order(dim, qd.f64)
