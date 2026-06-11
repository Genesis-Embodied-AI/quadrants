import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def _eigen_vector_equal(v1, v2, tol):
    if np.linalg.norm(v1) == 0.0:
        assert np.linalg.norm(v2) == 0.0
    else:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        try:
            np.testing.assert_allclose(v1, v2, atol=tol, rtol=tol)
        except AssertionError:
            assert (
                np.allclose(v1, -v2, atol=tol, rtol=tol)
                or np.allclose(v1, 1.0j * v2, atol=tol, rtol=tol)
                or np.allclose(v1, -1.0j * v2, atol=tol, rtol=tol)
            )


def _test_eig2x2_real(dt):
    A = qd.Matrix.field(2, 2, dtype=dt, shape=())
    v = qd.Matrix.field(2, 2, dtype=dt, shape=())
    w = qd.Matrix.field(4, 2, dtype=dt, shape=())

    A[None] = [[1, 1], [2, 3]]

    @qd.kernel
    def eigen_solve():
        v[None], w[None] = qd.eig(A[None])

    tol = 1e-5 if dt == qd.f32 else 1e-12
    dtype = np.float32 if dt == qd.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy()[:, 0].astype(dtype)
    w_ti = w.to_numpy()[0::2, :].astype(dtype)

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_ti)

    np.testing.assert_allclose(v_ti[idx_ti], v_np[idx_np], atol=tol, rtol=tol)
    _eigen_vector_equal(w_ti[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[1]], w_np[:, idx_np[1]], tol)


def _test_eig2x2_complex(dt):
    A = qd.Matrix.field(2, 2, dtype=dt, shape=())
    v = qd.Matrix.field(2, 2, dtype=dt, shape=())
    w = qd.Matrix.field(4, 2, dtype=dt, shape=())

    A[None] = [[1, -1], [1, 1]]

    @qd.kernel
    def eigen_solve():
        v[None], w[None] = qd.eig(A[None])

    tol = 1e-5 if dt == qd.f32 else 1e-12
    dtype = np.float32 if dt == qd.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy().astype(dtype)
    w_ti = w.to_numpy().astype(dtype)
    v_qd_complex = v_ti[:, 0] + v_ti[:, 1] * 1.0j
    w_qd_complex = w_ti[0::2, :] + w_ti[1::2, :] * 1.0j

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_qd_complex)

    np.testing.assert_allclose(v_qd_complex[idx_ti], v_np[idx_np], atol=tol, rtol=tol)
    _eigen_vector_equal(w_qd_complex[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_qd_complex[:, idx_ti[1]], w_np[:, idx_np[1]], tol)


def _test_sym_eig2x2(dt):
    A = qd.Matrix.field(2, 2, dtype=dt, shape=())
    v = qd.Vector.field(2, dtype=dt, shape=())
    w = qd.Matrix.field(2, 2, dtype=dt, shape=())

    A[None] = [[5, 3], [3, 2]]

    @qd.kernel
    def eigen_solve():
        v[None], w[None] = qd.sym_eig(A[None])

    tol = 1e-5 if dt == qd.f32 else 1e-12
    dtype = np.float32 if dt == qd.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy().astype(dtype)
    w_ti = w.to_numpy().astype(dtype)

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_ti)

    np.testing.assert_allclose(v_ti[idx_ti], v_np[idx_np], atol=tol, rtol=tol)
    _eigen_vector_equal(w_ti[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[1]], w_np[:, idx_np[1]], tol)


def _test_sym_eig3x3(dt, a00):
    A = qd.Matrix.field(3, 3, dtype=dt, shape=())
    v = qd.Vector.field(3, dtype=dt, shape=())
    w = qd.Matrix.field(3, 3, dtype=dt, shape=())

    A[None] = [[a00, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0]]

    @qd.kernel
    def eigen_solve():
        v[None], w[None] = qd.sym_eig(A[None])

    tol = 1e-5 if dt == qd.f32 else 1e-12
    dtype = np.float32 if dt == qd.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy().astype(dtype)
    w_ti = w.to_numpy().astype(dtype)

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_ti)

    np.testing.assert_allclose(v_ti[idx_ti], v_np[idx_np], atol=tol, rtol=tol)
    _eigen_vector_equal(w_ti[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[1]], w_np[:, idx_np[1]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[2]], w_np[:, idx_np[2]], tol)


def _test_sym_eig3x3_identity(dt):
    A = qd.Matrix.field(3, 3, dtype=dt, shape=())
    v = qd.Vector.field(3, dtype=dt, shape=())
    w = qd.Matrix.field(3, 3, dtype=dt, shape=())

    A[None] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    @qd.kernel
    def eigen_solve():
        X = qd.Matrix.identity(dt, 3)
        v[None], w[None] = qd.sym_eig(X)

    tol = 1e-5 if dt == qd.f32 else 1e-12
    dtype = np.float32 if dt == qd.f32 else np.float64

    eigen_solve()
    v_np, w_np = np.linalg.eig(A.to_numpy().astype(dtype))
    v_ti = v.to_numpy().astype(dtype)
    w_ti = w.to_numpy().astype(dtype)

    # sort by eigenvalues
    idx_np = np.argsort(v_np)
    idx_ti = np.argsort(v_ti)

    np.testing.assert_allclose(v_ti[idx_ti], v_np[idx_np], atol=tol, rtol=tol)
    _eigen_vector_equal(w_ti[:, idx_ti[0]], w_np[:, idx_np[0]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[1]], w_np[:, idx_np[1]], tol)
    _eigen_vector_equal(w_ti[:, idx_ti[2]], w_np[:, idx_np[2]], tol)


@pytest.mark.parametrize("func", [_test_eig2x2_real, _test_eig2x2_complex])
@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_eig2x2_f32(func):
    func(qd.f32)


@pytest.mark.parametrize("func", [_test_eig2x2_real, _test_eig2x2_complex])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_eig2x2_f64(func):
    func(qd.f64)


@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_sym_eig2x2_f32():
    _test_sym_eig2x2(qd.f32)


@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_sym_eig2x2_f64():
    _test_sym_eig2x2(qd.f64)


@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_sym_eig3x3_identity_f32():
    _test_sym_eig3x3_identity(qd.f32)


@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_sym_eig3x3_identity_f64():
    _test_sym_eig3x3_identity(qd.f64)


@pytest.mark.parametrize("a00", [i for i in range(10)])
@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_sym_eig3x3_f32(a00):
    _test_sym_eig3x3(qd.f32, a00)


@pytest.mark.parametrize("a00", [i for i in range(10)])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_sym_eig3x3_f64(a00):
    _test_sym_eig3x3(qd.f64, a00)


# ---------------------------------------------------------------------------
# Symmetric eigendecomposition for N >= 4 (cyclic Jacobi). Supported sizes are capped at 6×6 (N=4 unrolled, N=5/6
# runtime sweep loop); larger blocks are intentionally unsupported because the inner Givens steps stay unrolled and
# compile time grows steeply.
# ---------------------------------------------------------------------------


def _make_symmetric(M):
    """Return ``(M + M.T) / 2`` cast to the same dtype as ``M``."""
    return ((M + M.T) * 0.5).astype(M.dtype)


def _sym_eig_factory_random(n, dt):
    np_dt = np.float32 if dt == qd.f32 else np.float64
    return _make_symmetric(np.random.default_rng(0xE160 + n).standard_normal((n, n)).astype(np_dt))


def _sym_eig_factory_spd(n, dt):
    np_dt = np.float32 if dt == qd.f32 else np.float64
    A = np.random.default_rng(0x5BD0 + n).standard_normal((n, n)).astype(np_dt)
    return ((A @ A.T) + np.eye(n) * 2.0).astype(np_dt)


def _sym_eig_factory_indefinite(n, dt):
    """Symmetric matrix with mix of positive and negative eigenvalues — exercises make_spd's clamping path."""
    np_dt = np.float32 if dt == qd.f32 else np.float64
    rng = np.random.default_rng(0xCAFE + n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.linspace(-1.0, 2.0, n)
    return (Q @ np.diag(eigs) @ Q.T).astype(np_dt)


def _sym_eig_factory_diagonal(n, dt):
    """Diagonal matrix — eigenvalues / vectors are trivial (sanity check)."""
    np_dt = np.float32 if dt == qd.f32 else np.float64
    return np.diag(np.linspace(1.0, float(n), n).astype(np_dt))


def _sym_eig_factory_repeated_eigs(n, dt):
    """Symmetric with two repeated eigenvalues and one well-separated."""
    np_dt = np.float32 if dt == qd.f32 else np.float64
    rng = np.random.default_rng(0xDEAD + n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.array([1.0] * (n // 2) + [3.0] * (n - n // 2))
    return (Q @ np.diag(eigs) @ Q.T).astype(np_dt)


def _sym_eig_factory_negative_definite(n, dt):
    """Symmetric with all-negative eigenvalues — make_spd should produce a zero matrix."""
    np_dt = np.float32 if dt == qd.f32 else np.float64
    rng = np.random.default_rng(0xBEEF + n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = -np.linspace(0.5, 2.0, n)
    return (Q @ np.diag(eigs) @ Q.T).astype(np_dt)


def _sym_eig_factory_equal_diag_block(n, dt):
    """Equal-diagonal pair (``A[0,0] == A[1,1]``) with a negative off-diagonal — drives the degenerate Jacobi
    rotation where ``|A[p,p] - A[q,q]| < eps`` and ``A[p,q] < 0``, so the rotation angle is pinned to ``tau = -1``."""
    np_dt = np.float32 if dt == qd.f32 else np.float64
    A = np.diag(np.arange(2, 2 + n, dtype=np_dt))
    A[0, 0] = 2.0
    A[1, 1] = 2.0
    A[0, 1] = A[1, 0] = -1.0
    return A


def _test_sym_eig_general(n, dt, factory):
    np_dt = np.float32 if dt == qd.f32 else np.float64
    A_np = factory(n, dt)
    assert np.allclose(A_np, A_np.T, atol=1e-5)

    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    eigvals = qd.Vector.field(n, dtype=dt, shape=())
    eigvecs = qd.Matrix.field(n, n, dtype=dt, shape=())
    A.from_numpy(A_np)

    @qd.kernel
    def run():
        eigvals[None], eigvecs[None] = qd.sym_eig(A[None], dt)

    run()
    eigvals_qd = eigvals.to_numpy().astype(np_dt)
    eigvecs_qd = eigvecs.to_numpy().astype(np_dt)

    eigvals_np = np.linalg.eigvalsh(A_np)
    tol = 5e-3 if dt == qd.f32 else 1e-9

    np.testing.assert_allclose(np.sort(eigvals_qd), np.sort(eigvals_np), rtol=tol, atol=tol)

    Q = eigvecs_qd
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), rtol=tol, atol=tol)
    A_reconstructed = Q @ np.diag(eigvals_qd) @ Q.T
    np.testing.assert_allclose(A_reconstructed, A_np, rtol=tol, atol=tol)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize(
    "factory",
    [
        _sym_eig_factory_random,
        _sym_eig_factory_spd,
        _sym_eig_factory_indefinite,
        _sym_eig_factory_diagonal,
        _sym_eig_factory_repeated_eigs,
    ],
)
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_sym_eig_general_f32(n, factory):
    _test_sym_eig_general(n, qd.f32, factory)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize(
    "factory",
    [
        _sym_eig_factory_random,
        _sym_eig_factory_spd,
        _sym_eig_factory_indefinite,
        _sym_eig_factory_diagonal,
        _sym_eig_factory_repeated_eigs,
    ],
)
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_sym_eig_general_f64(n, factory):
    _test_sym_eig_general(n, qd.f64, factory)


def _test_make_spd(n, dt, factory):
    np_dt = np.float32 if dt == qd.f32 else np.float64
    A_np = factory(n, dt)
    assert np.allclose(A_np, A_np.T, atol=1e-5)

    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    A_spd = qd.Matrix.field(n, n, dtype=dt, shape=())
    A.from_numpy(A_np)

    @qd.kernel
    def run():
        A_spd[None] = qd.make_spd(A[None], dt)

    run()
    A_spd_qd = A_spd.to_numpy().astype(np_dt)
    spd_eigs = np.linalg.eigvalsh(A_spd_qd)
    tol = 5e-3 if dt == qd.f32 else 1e-9

    # Must be symmetric.
    np.testing.assert_allclose(A_spd_qd, A_spd_qd.T, rtol=tol, atol=tol)
    # Must be PSD: eigenvalues >= -tol.
    assert spd_eigs.min() >= -tol, f"min eig of make_spd({factory.__name__}) = {spd_eigs.min()}"

    # Reference: numpy reconstruct with eigenvalues clamped to >= 0.
    eigs_np, vecs_np = np.linalg.eigh(A_np)
    eigs_clamped = np.clip(eigs_np, 0.0, None)
    expected = vecs_np @ np.diag(eigs_clamped) @ vecs_np.T

    np.testing.assert_allclose(A_spd_qd, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize(
    "factory",
    [_sym_eig_factory_indefinite, _sym_eig_factory_random, _sym_eig_factory_spd],
)
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_make_spd_f32(n, factory):
    _test_make_spd(n, qd.f32, factory)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize(
    "factory",
    [_sym_eig_factory_indefinite, _sym_eig_factory_random, _sym_eig_factory_spd],
)
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_make_spd_f64(n, factory):
    _test_make_spd(n, qd.f64, factory)


# ---------------------------------------------------------------------------
# Edge-case + contract tests for sym_eig / make_spd.
# ---------------------------------------------------------------------------


def _test_sym_eig_trivial(n, dt, A_np, expected_eigvals):
    """Run ``qd.sym_eig`` and assert it returns ``expected_eigvals`` (sorted ascending) plus an orthonormal
    eigenvector basis, on the trivial input ``A_np``."""
    np_dt = np.float32 if dt == qd.f32 else np.float64
    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    eigvals = qd.Vector.field(n, dtype=dt, shape=())
    eigvecs = qd.Matrix.field(n, n, dtype=dt, shape=())
    A.from_numpy(A_np.astype(np_dt))

    @qd.kernel
    def run():
        eigvals[None], eigvecs[None] = qd.sym_eig(A[None], dt)

    run()
    tol = 5e-3 if dt == qd.f32 else 1e-9
    eigvals_qd = np.sort(eigvals.to_numpy().astype(np_dt))
    np.testing.assert_allclose(eigvals_qd, np.sort(expected_eigvals.astype(np_dt)), rtol=tol, atol=tol)
    Q = eigvecs.to_numpy().astype(np_dt)
    np.testing.assert_allclose(Q.T @ Q, np.eye(n), rtol=tol, atol=tol)


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize("alpha", [0.0, 1.0, -2.5])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_sym_eig_alpha_identity_f64(n, alpha):
    """``α·I`` has every eigenvalue equal to ``α``. Covers the all-equal / fully-degenerate case that the random /
    repeated factories don't hit (``α=0`` also covers the zero-matrix case)."""
    A_np = (alpha * np.eye(n)).astype(np.float64)
    expected = np.full(n, alpha, dtype=np.float64)
    _test_sym_eig_trivial(n, qd.f64, A_np, expected)


def _test_make_spd_idempotent(n, dt, factory):
    """``make_spd(make_spd(A)) ≈ make_spd(A)`` — defining property of a projector.

    Uses an ndarray-arg parametric kernel so ``qd.make_spd`` is JIT-compiled exactly once and called twice
    (``A → A_spd_1`` and ``A_spd_1 → A_spd_2``). Compiling it twice at the larger sizes on CUDA blows past the per-test
    timeout — one compile fits comfortably.
    """
    np_dt = np.float32 if dt == qd.f32 else np.float64
    mat_t = qd.types.matrix(n, n, dt)
    A = qd.Matrix.ndarray(n, n, dtype=dt, shape=(1,))
    A_spd_1 = qd.Matrix.ndarray(n, n, dtype=dt, shape=(1,))
    A_spd_2 = qd.Matrix.ndarray(n, n, dtype=dt, shape=(1,))
    A.from_numpy(factory(n, dt)[np.newaxis])

    @qd.kernel
    def project(src: qd.types.NDArray[mat_t, 1], dst: qd.types.NDArray[mat_t, 1]):
        dst[0] = qd.make_spd(src[0], dt)

    project(A, A_spd_1)
    project(A_spd_1, A_spd_2)

    tol = 5e-3 if dt == qd.f32 else 1e-9
    np.testing.assert_allclose(
        A_spd_2.to_numpy()[0].astype(np_dt),
        A_spd_1.to_numpy()[0].astype(np_dt),
        rtol=tol,
        atol=tol,
    )


@pytest.mark.parametrize("n", [4, 6])
@pytest.mark.parametrize(
    "factory",
    [_sym_eig_factory_indefinite, _sym_eig_factory_negative_definite, _sym_eig_factory_spd],
)
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_make_spd_idempotent_f64(n, factory):
    _test_make_spd_idempotent(n, qd.f64, factory)


@pytest.mark.parametrize("n", [4, 6])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_make_spd_negative_definite_zero_f64(n):
    """A symmetric matrix with all-negative eigenvalues projects to the zero matrix (``Q · diag(max(λ, 0)) · Qᵀ``
    with all ``λ < 0``)."""
    np_dt = np.float64
    A_np = _sym_eig_factory_negative_definite(n, qd.f64)
    A = qd.Matrix.field(n, n, dtype=qd.f64, shape=())
    A_spd = qd.Matrix.field(n, n, dtype=qd.f64, shape=())
    A.from_numpy(A_np)

    @qd.kernel
    def run():
        A_spd[None] = qd.make_spd(A[None], qd.f64)

    run()
    tol = 1e-9
    np.testing.assert_allclose(A_spd.to_numpy().astype(np_dt), np.zeros((n, n)), rtol=tol, atol=tol)


@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_sym_eig_above_cap_raises():
    """``qd.sym_eig`` only supports ``N <= 6``; calling at ``N = 7`` must raise a clear error rather than silently
    producing wrong results."""
    A = qd.Matrix.field(7, 7, dtype=qd.f64, shape=())
    A.from_numpy(np.eye(7))
    with pytest.raises(Exception, match="up to 6"):

        @qd.kernel
        def run():
            _ = qd.sym_eig(A[None], qd.f64)

        run()


@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_make_spd_above_cap_raises():
    """``qd.make_spd`` shares the cyclic-Jacobi path, so it carries the same ``N <= 6`` cap as ``qd.sym_eig``; calling
    at ``N = 7`` must raise rather than compile the slow unrolled path."""
    A = qd.Matrix.field(7, 7, dtype=qd.f64, shape=())
    A.from_numpy(np.eye(7))
    with pytest.raises(Exception, match="up to 6"):

        @qd.kernel
        def run():
            _ = qd.make_spd(A[None], qd.f64)

        run()


# ---------------------------------------------------------------------------
# CPU coverage for the N > 4 runtime sweep branch and make_spd's dispatch. The parametrized N>=4 / make_spd tests above
# run on ``arch=qd.gpu`` only, so the CPU coverage runner never exercises sym_eig_general's runtime branch or make_spd's
# body. These mirror them on CPU at N=6 (runtime branch); the equal-diagonal factory additionally hits the degenerate
# ``tau = -1`` rotation in both the static (N=2) and runtime (N=6) sweep branches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_sym_eig_factory_random, _sym_eig_factory_indefinite])
@test_utils.test(arch=qd.cpu, require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_sym_eig_general_cpu_f64(factory):
    _test_sym_eig_general(6, qd.f64, factory)


@pytest.mark.parametrize("n", [2, 6])
@test_utils.test(arch=qd.cpu, require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_sym_eig_equal_diag_degenerate_cpu_f64(n):
    """Equal-diagonal pair with a negative off-diagonal hits the ``|diff| < eps`` / ``apq < 0`` degenerate rotation
    (``tau = -1``) in both the static (N=2) and runtime (N=6) sweep branches."""
    _test_sym_eig_general(n, qd.f64, _sym_eig_factory_equal_diag_block)


@test_utils.test(arch=qd.cpu, require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_make_spd_cpu_f64():
    _test_make_spd(6, qd.f64, _sym_eig_factory_indefinite)


# ---------------------------------------------------------------------------
# Sort-order contract: every shape of qd.sym_eig must return eigenvalues in ascending order (matches NumPy / LAPACK's
# `eigh`). The vector at column i must be the eigenvector for `eigvals[i]` (i.e. the sort applies to both).
# ---------------------------------------------------------------------------


def _test_sym_eig_sort_order(n, dt):
    np_dt = np.float32 if dt == qd.f32 else np.float64
    rng = np.random.default_rng(0x501D + n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs_target = np.linspace(-2.0, 3.0, n).astype(np_dt)
    A_np = (Q @ np.diag(eigs_target) @ Q.T).astype(np_dt)

    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    eigvals = qd.Vector.field(n, dtype=dt, shape=())
    eigvecs = qd.Matrix.field(n, n, dtype=dt, shape=())
    A.from_numpy(A_np)

    @qd.kernel
    def run():
        eigvals[None], eigvecs[None] = qd.sym_eig(A[None], dt)

    run()
    eigvals_qd = eigvals.to_numpy().astype(np_dt)
    eigvecs_qd = eigvecs.to_numpy().astype(np_dt)
    tol = 5e-3 if dt == qd.f32 else 1e-9

    diffs = np.diff(eigvals_qd)
    assert np.all(diffs >= -tol), f"qd.sym_eig n={n} not ascending: eigvals = {eigvals_qd.tolist()}"

    for i in range(n):
        v = eigvecs_qd[:, i]
        Av = A_np @ v
        residual = np.linalg.norm(Av - eigvals_qd[i] * v)
        assert residual <= tol * max(
            1.0, abs(eigvals_qd[i])
        ), f"column {i} is not the eigenvector of eigvals[{i}]={eigvals_qd[i]}: residual={residual}"


@pytest.mark.parametrize("n", [2, 3, 6])
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_sym_eig_sort_order_f32(n):
    _test_sym_eig_sort_order(n, qd.f32)


@pytest.mark.parametrize("n", [2, 3, 6])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_sym_eig_sort_order_f64(n):
    _test_sym_eig_sort_order(n, qd.f64)
