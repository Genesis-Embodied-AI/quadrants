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
    if qd.lang.impl.current_cfg().arch == qd.vulkan:
        # `_sym_eig3x3` (Eigen3 `computeDirect` closed form) crashes the NVIDIA Vulkan SPIR-V → NVVM
        # frontend (SIGSEGV inside `libnvidia-gpucomp.so` / `libnvidia-glvkspirv.so`) during pipeline
        # creation on driver 580.76.05. spirv-val accepts the shader and spirv-cross round-trips it to
        # valid GLSL, so the bug is in NVIDIA's compiler when handling the deeply-nested closed-form
        # path (Cardano method + `dsyevq3` Givens-rotation fallback inlined into a single non-offloaded
        # compute kernel with many `OpSelectionMerge` blocks updating Function-scope variables). The
        # `_sym_eig_sort_order` helper also skips this same case (see comment there). `n == 2` and
        # ``n >= 4`` (`sym_eig_general`) compile and run cleanly. Remove this skip once NVIDIA fixes
        # the driver crash (or `_sym_eig3x3` is refactored to a more partitioned codegen pattern).
        pytest.skip("NVIDIA Vulkan driver SIGSEGV in `_sym_eig3x3` SPIR-V codegen (pre-existing)")
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
    if qd.lang.impl.current_cfg().arch == qd.vulkan:
        # Same `_sym_eig3x3` NVIDIA Vulkan SPIR-V codegen SIGSEGV as the random-input case below — see
        # the comment in `_test_sym_eig3x3` for details. The identity matrix specifically hits the
        # `norm <= error` early-return path that funnels into `dsyevq3`'s Givens-rotation sweep, which
        # is one (but not the only) trigger for the driver crash.
        pytest.skip("NVIDIA Vulkan driver SIGSEGV in `_sym_eig3x3` SPIR-V codegen (pre-existing)")
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
# Symmetric eigendecomposition for N >= 4 (Householder + implicit QR). qipc's ABD / contact Hessian make_spd
# projection needs sizes 6, 9, 12.
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


@pytest.mark.parametrize("n", [4, 5, 6, 9, 12])
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


@pytest.mark.parametrize("n", [4, 5, 6, 9, 12])
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


@pytest.mark.parametrize("n", [4, 6, 9, 12])
@pytest.mark.parametrize(
    "factory",
    [_sym_eig_factory_indefinite, _sym_eig_factory_random, _sym_eig_factory_spd],
)
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_make_spd_f32(n, factory):
    _test_make_spd(n, qd.f32, factory)


@pytest.mark.parametrize("n", [4, 6, 9, 12])
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


@pytest.mark.parametrize("n", [4, 6, 9, 12])
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
    (``A → A_spd_1`` and ``A_spd_1 → A_spd_2``). Compiling it twice at N=12 on CUDA blows past the per-test
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


@pytest.mark.parametrize("n", [4, 6, 9, 12])
@pytest.mark.parametrize(
    "factory",
    [_sym_eig_factory_indefinite, _sym_eig_factory_negative_definite, _sym_eig_factory_spd],
)
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_make_spd_idempotent_f64(n, factory):
    _test_make_spd_idempotent(n, qd.f64, factory)


@pytest.mark.parametrize("n", [4, 6, 9, 12])
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
    """``qd.sym_eig`` only supports ``N <= 12``; calling at ``N = 13`` must raise a clear error rather than silently
    producing wrong results."""
    A = qd.Matrix.field(13, 13, dtype=qd.f64, shape=())
    A.from_numpy(np.eye(13))
    with pytest.raises(Exception, match="up to 12"):

        @qd.kernel
        def run():
            _ = qd.sym_eig(A[None], qd.f64)

        run()


# ---------------------------------------------------------------------------
# Sort-order contract: every shape of qd.sym_eig must return eigenvalues in ascending order (matches NumPy / LAPACK's
# `eigh`). The vector at column i must be the eigenvector for `eigvals[i]` (i.e. the sort applies to both).
# ---------------------------------------------------------------------------


def _test_sym_eig_sort_order(n, dt):
    if n == 3 and qd.lang.impl.current_cfg().arch == qd.vulkan:
        # The closed-form 3×3 path (`_sym_eig3x3` → Eigen3 `computeDirect`) segfaults during SPIR-V codegen on the
        # cluster's Vulkan stack (genesis-v1_23 image). Same code runs cleanly on amddesktop's Vulkan, so this is a
        # pre-existing driver / SDK quirk, not a regression from sort-order changes — n=2 and n>=4 work on all
        # backends. Track separately if it matters; remove this skip once the underlying Vulkan codegen is fixed.
        pytest.skip("cluster Vulkan segfaults in _sym_eig3x3 SPIR-V codegen (pre-existing)")
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


@pytest.mark.parametrize("n", [2, 3, 4, 6, 9, 12])
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_sym_eig_sort_order_f32(n):
    _test_sym_eig_sort_order(n, qd.f32)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 9, 12])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_sym_eig_sort_order_f64(n):
    _test_sym_eig_sort_order(n, qd.f64)
