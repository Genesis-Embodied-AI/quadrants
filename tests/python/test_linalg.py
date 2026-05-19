import math

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test()
def test_const_init():
    a = qd.Matrix.field(2, 3, dtype=qd.i32, shape=())
    b = qd.Vector.field(3, dtype=qd.i32, shape=())

    @qd.kernel
    def init():
        a[None] = qd.Matrix([[0, 1, 2], [3, 4, 5]])
        b[None] = qd.Vector([0, 1, 2])

    init()

    for i in range(2):
        for j in range(3):
            assert a[None][i, j] == i * 3 + j

    for j in range(3):
        assert b[None][j] == j


@test_utils.test()
def test_basic_utils():
    a = qd.Vector.field(3, dtype=qd.f32)
    b = qd.Vector.field(2, dtype=qd.f32)
    abT = qd.Matrix.field(3, 2, dtype=qd.f32)
    aNormalized = qd.Vector.field(3, dtype=qd.f32)

    normA = qd.field(qd.f32)
    normSqrA = qd.field(qd.f32)
    normInvA = qd.field(qd.f32)

    qd.root.place(a, b, abT, aNormalized, normA, normSqrA, normInvA)

    @qd.kernel
    def init():
        a[None] = qd.Vector([1.0, 2.0, -3.0])
        b[None] = qd.Vector([4.0, 5.0])
        abT[None] = a[None].outer_product(b[None])

        normA[None] = a[None].norm()
        normSqrA[None] = a[None].norm_sqr()
        normInvA[None] = a[None].norm_inv()

        aNormalized[None] = a[None].normalized()

    init()

    for i in range(3):
        for j in range(2):
            assert abT[None][i, j] == a[None][i] * b[None][j]

    sqrt14 = np.sqrt(14.0)
    invSqrt14 = 1.0 / sqrt14
    assert normSqrA[None] == test_utils.approx(14.0)
    assert normInvA[None] == test_utils.approx(invSqrt14)
    assert normA[None] == test_utils.approx(sqrt14)
    assert aNormalized[None][0] == test_utils.approx(1.0 * invSqrt14)
    assert aNormalized[None][1] == test_utils.approx(2.0 * invSqrt14)
    assert aNormalized[None][2] == test_utils.approx(-3.0 * invSqrt14)


@test_utils.test()
def test_cross():
    a = qd.Vector.field(3, dtype=qd.f32)
    b = qd.Vector.field(3, dtype=qd.f32)
    c = qd.Vector.field(3, dtype=qd.f32)

    a2 = qd.Vector.field(2, dtype=qd.f32)
    b2 = qd.Vector.field(2, dtype=qd.f32)
    c2 = qd.field(dtype=qd.f32)

    qd.root.place(a, b, c, a2, b2, c2)

    @qd.kernel
    def init():
        a[None] = qd.Vector([1.0, 2.0, 3.0])
        b[None] = qd.Vector([4.0, 5.0, 6.0])
        c[None] = a[None].cross(b[None])

        a2[None] = qd.Vector([1.0, 2.0])
        b2[None] = qd.Vector([4.0, 5.0])
        c2[None] = a2[None].cross(b2[None])

    init()
    assert c[None][0] == -3.0
    assert c[None][1] == 6.0
    assert c[None][2] == -3.0
    assert c2[None] == -3.0


@test_utils.test()
def test_dot():
    a = qd.Vector.field(3, dtype=qd.f32)
    b = qd.Vector.field(3, dtype=qd.f32)
    c = qd.field(dtype=qd.f32)

    a2 = qd.Vector.field(2, dtype=qd.f32)
    b2 = qd.Vector.field(2, dtype=qd.f32)
    c2 = qd.field(dtype=qd.f32)

    qd.root.place(a, b, c, a2, b2, c2)

    @qd.kernel
    def init():
        a[None] = qd.Vector([1.0, 2.0, 3.0])
        b[None] = qd.Vector([4.0, 5.0, 6.0])
        c[None] = a[None].dot(b[None])

        a2[None] = qd.Vector([1.0, 2.0])
        b2[None] = qd.Vector([4.0, 5.0])
        c2[None] = a2[None].dot(b2[None])

    init()
    assert c[None] == 32.0
    assert c2[None] == 14.0


def _test_frobenius_inner(n, dt):
    """Frobenius inner product ⟨A, B⟩ = sum_ij A_ij B_ij at size n×n."""
    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    B = qd.Matrix.field(n, n, dtype=dt, shape=())
    out_method = qd.field(dtype=dt, shape=())
    out_self = qd.field(dtype=dt, shape=())

    rng = np.random.default_rng(0xF20B + n + (0 if dt == qd.f32 else 1))
    A_np = rng.standard_normal((n, n)).astype(np.float32 if dt == qd.f32 else np.float64)
    B_np = rng.standard_normal((n, n)).astype(np.float32 if dt == qd.f32 else np.float64)
    A.from_numpy(A_np)
    B.from_numpy(B_np)

    @qd.kernel
    def run():
        out_method[None] = A[None].frobenius_inner(B[None])
        out_self[None] = A[None].frobenius_inner(A[None])

    run()

    expected_AB = float(np.sum(A_np * B_np))
    expected_AA = float(np.sum(A_np * A_np))
    tol = 1e-4 if dt == qd.f32 else 1e-10
    assert out_method[None] == test_utils.approx(expected_AB, rel=tol, abs=tol)
    assert out_self[None] == test_utils.approx(expected_AA, rel=tol, abs=tol)
    assert out_self[None] == test_utils.approx(A.to_numpy().__pow__(2).sum(), rel=tol, abs=tol)


@pytest.mark.parametrize("n", [3, pytest.param(12, marks=pytest.mark.slow)])
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_frobenius_inner_f32(n):
    _test_frobenius_inner(n, qd.f32)


@pytest.mark.parametrize("n", [3, pytest.param(12, marks=pytest.mark.slow)])
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_frobenius_inner_f64(n):
    _test_frobenius_inner(n, qd.f64)


def _test_frobenius_inner_rectangular(rows, cols, dt):
    """Frobenius inner product on non-square matrices (qipc uses 9×12, 12×3, etc.)."""
    A = qd.Matrix.field(rows, cols, dtype=dt, shape=())
    B = qd.Matrix.field(rows, cols, dtype=dt, shape=())
    out = qd.field(dtype=dt, shape=())

    rng = np.random.default_rng(0xFA7E + rows * 31 + cols)
    A_np = rng.standard_normal((rows, cols)).astype(np.float32 if dt == qd.f32 else np.float64)
    B_np = rng.standard_normal((rows, cols)).astype(np.float32 if dt == qd.f32 else np.float64)
    A.from_numpy(A_np)
    B.from_numpy(B_np)

    @qd.kernel
    def run():
        out[None] = A[None].frobenius_inner(B[None])

    run()

    expected = float(np.sum(A_np * B_np))
    tol = 1e-4 if dt == qd.f32 else 1e-10
    assert out[None] == test_utils.approx(expected, rel=tol, abs=tol)


@pytest.mark.parametrize(
    "rows,cols",
    [
        pytest.param(9, 12, marks=pytest.mark.slow),
        pytest.param(12, 3, marks=pytest.mark.slow),
        (2, 4),
    ],
)
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_frobenius_inner_rectangular_f32(rows, cols):
    _test_frobenius_inner_rectangular(rows, cols, qd.f32)


@pytest.mark.parametrize(
    "rows,cols",
    [
        pytest.param(9, 12, marks=pytest.mark.slow),
        pytest.param(12, 3, marks=pytest.mark.slow),
        (2, 4),
    ],
)
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_frobenius_inner_rectangular_f64(rows, cols):
    _test_frobenius_inner_rectangular(rows, cols, qd.f64)


def _test_matmul_chain(dt):
    """3-way matmul chain at qipc IPC sizes: (9×12) · (12×12) · (12×9) → (9×9).

    Verifies that ``Matrix.__matmul__`` compiles and is numerically correct at the largest size qipc needs. Quadrants
    imposes no enforced size cap on matmul, but the unrolled `static(range)` triple loop produces ~1296 FMAs per
    intermediate, so this test catches compile-time blow-up or back-end miscompiles at large sizes.
    """
    np_dt = np.float32 if dt == qd.f32 else np.float64
    A_np = np.random.default_rng(0xCA70).standard_normal((9, 12)).astype(np_dt)
    B_np = np.random.default_rng(0xCA71).standard_normal((12, 12)).astype(np_dt)
    C_np = np.random.default_rng(0xCA72).standard_normal((12, 9)).astype(np_dt)

    A = qd.Matrix.field(9, 12, dtype=dt, shape=())
    B = qd.Matrix.field(12, 12, dtype=dt, shape=())
    C = qd.Matrix.field(12, 9, dtype=dt, shape=())
    AB = qd.Matrix.field(9, 12, dtype=dt, shape=())
    ABC_chained = qd.Matrix.field(9, 9, dtype=dt, shape=())
    ABC_staged = qd.Matrix.field(9, 9, dtype=dt, shape=())

    A.from_numpy(A_np)
    B.from_numpy(B_np)
    C.from_numpy(C_np)

    @qd.kernel
    def run():
        ABC_chained[None] = A[None] @ B[None] @ C[None]
        AB[None] = A[None] @ B[None]
        ABC_staged[None] = AB[None] @ C[None]

    run()

    expected = A_np @ B_np @ C_np
    tol = 5e-4 if dt == qd.f32 else 1e-10

    np.testing.assert_allclose(ABC_chained.to_numpy(), expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(ABC_staged.to_numpy(), expected, rtol=tol, atol=tol)
    np.testing.assert_allclose(AB.to_numpy(), A_np @ B_np, rtol=tol, atol=tol)
    np.testing.assert_allclose(ABC_chained.to_numpy(), ABC_staged.to_numpy(), rtol=tol, atol=tol)


@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_matmul_chain_qipc_sizes_f32():
    _test_matmul_chain(qd.f32)


@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_matmul_chain_qipc_sizes_f64():
    _test_matmul_chain(qd.f64)


@test_utils.test()
def test_transpose():
    dim = 3
    m = qd.Matrix.field(dim, dim, qd.f32)

    qd.root.place(m)

    @qd.kernel
    def transpose():
        mat = m[None].transpose()
        m[None] = mat

    for i in range(dim):
        for j in range(dim):
            m[None][i, j] = i * 2 + j * 7

    transpose()

    for i in range(dim):
        for j in range(dim):
            assert m[None][j, i] == test_utils.approx(i * 2 + j * 7)


def _test_polar_decomp(dim, dt):
    m = qd.Matrix.field(dim, dim, dt)
    r = qd.Matrix.field(dim, dim, dt)
    s = qd.Matrix.field(dim, dim, dt)
    I = qd.Matrix.field(dim, dim, dt)
    D = qd.Matrix.field(dim, dim, dt)

    qd.root.place(m, r, s, I, D)

    @qd.kernel
    def polar():
        R, S = qd.polar_decompose(m[None], dt)
        r[None] = R
        s[None] = S
        m[None] = R @ S
        I[None] = R @ R.transpose()
        D[None] = S - S.transpose()

    def V(i, j):
        return i * 2 + j * 7 + int(i == j) * 3

    for i in range(dim):
        for j in range(dim):
            m[None][i, j] = V(i, j)

    polar()

    tol = 5e-5 if dt == qd.f32 else 1e-12

    for i in range(dim):
        for j in range(dim):
            assert m[None][i, j] == test_utils.approx(V(i, j), abs=tol)
            assert I[None][i, j] == test_utils.approx(int(i == j), abs=tol)
            assert D[None][i, j] == test_utils.approx(0, abs=tol)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(default_fp=qd.f32)
def test_polar_decomp_f32(dim):
    _test_polar_decomp(dim, qd.f32)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64)
def test_polar_decomp_f64(dim):
    _test_polar_decomp(dim, qd.f64)


@test_utils.test()
def test_matrix():
    x = qd.Matrix.field(2, 2, dtype=qd.i32)

    qd.root.dense(qd.i, 16).place(x)

    @qd.kernel
    def inc():
        for i in x:
            delta = qd.Matrix([[3, 0], [0, 0]])
            x[i][1, 1] = x[i][0, 0] + 1
            x[i] = x[i] + delta
            x[i] += delta

    for i in range(10):
        x[i][0, 0] = i

    inc()

    for i in range(10):
        assert x[i][0, 0] == 6 + i
        assert x[i][1, 1] == 1 + i


@pytest.mark.parametrize("n", range(1, 5))
@test_utils.test()
def test_mat_inverse_size(n):
    m = qd.Matrix.field(n, n, dtype=qd.f32, shape=())
    M = np.empty(shape=(n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            M[i, j] = i * j + i * 3 + j + 1 + int(i == j) * 4
    assert np.linalg.det(M) != 0

    m.from_numpy(M)

    @qd.kernel
    def invert():
        m[None] = m[None].inverse()

    invert()

    m_np = m.to_numpy(keep_dims=True)
    np.testing.assert_almost_equal(m_np, np.linalg.inv(M))


# ---------------------------------------------------------------------------
# Matrix.inverse for N up to 12 (LU with partial pivoting).
#
# qipc's ABD diagonal preconditioner needs Matrix.inverse at sizes up to 12×12. The existing closed-form path
# (≤ 4×4) is preserved; sizes 5–12 dispatch to a generic LU-with-partial-pivoting impl.
# ---------------------------------------------------------------------------


def _inverse_diagonally_dominant(n_, dt_):
    """Random N×N with a strong diagonal — well-conditioned, non-symmetric."""
    np_dt = np.float32 if dt_ == qd.f32 else np.float64
    rng = np.random.default_rng(0xD0B7 + n_)
    M = rng.standard_normal((n_, n_)).astype(np_dt)
    M += np.eye(n_, dtype=np_dt) * (n_ + 1)
    return M


def _inverse_spd(n_, dt_):
    """Symmetric positive-definite (qipc's ABD preconditioner is SPD)."""
    np_dt = np.float32 if dt_ == qd.f32 else np.float64
    rng = np.random.default_rng(0x5BD0 + n_)
    A = rng.standard_normal((n_, n_)).astype(np_dt)
    return (A @ A.T + np.eye(n_, dtype=np_dt) * 2.0).astype(np_dt)


def _inverse_pivoting_required(n_, dt_):
    """Permuted upper-triangular: top-left entry is zero so no-pivot LU would fail."""
    np_dt = np.float32 if dt_ == qd.f32 else np.float64
    rng = np.random.default_rng(0xB1C7 + n_)
    M = np.triu(rng.standard_normal((n_, n_)).astype(np_dt))
    np.fill_diagonal(M, np.arange(1.0, n_ + 1.0, dtype=np_dt))
    P = np.eye(n_, dtype=np_dt)
    perm = list(reversed(range(n_)))
    return (P[perm] @ M).astype(np_dt)


def _test_inverse_at_size(n_, dt_, factory):
    m = qd.Matrix.field(n_, n_, dtype=dt_, shape=())
    inv = qd.Matrix.field(n_, n_, dtype=dt_, shape=())
    M = factory(n_, dt_)
    assert np.abs(np.linalg.det(M)) > 1e-6, "test factory produced near-singular input"
    m.from_numpy(M)

    @qd.kernel
    def run():
        inv[None] = m[None].inverse()

    run()

    inv_np = inv.to_numpy()
    expected = np.linalg.inv(M)
    cond = float(np.linalg.cond(M))
    eps = np.finfo(np.float32 if dt_ == qd.f32 else np.float64).eps
    tol = 50 * cond * eps + (1e-5 if dt_ == qd.f32 else 1e-12)

    np.testing.assert_allclose(
        inv_np,
        expected,
        rtol=tol,
        atol=tol,
        err_msg=f"size {n_}, factory {factory.__name__}, cond={cond:.2e}",
    )
    # Round-trip M @ M⁻¹ ≈ I.
    np.testing.assert_allclose(M @ inv_np, np.eye(n_), rtol=tol, atol=tol)


@pytest.mark.parametrize("n", [5, pytest.param(12, marks=pytest.mark.slow)])
@pytest.mark.parametrize(
    "factory",
    [_inverse_diagonally_dominant, _inverse_spd, _inverse_pivoting_required],
)
@test_utils.test(arch=qd.gpu, default_fp=qd.f32, fast_math=False)
def test_inverse_large_f32(n, factory):
    _test_inverse_at_size(n, qd.f32, factory)


@pytest.mark.parametrize("n", [5, pytest.param(12, marks=pytest.mark.slow)])
@pytest.mark.parametrize(
    "factory",
    [_inverse_diagonally_dominant, _inverse_spd, _inverse_pivoting_required],
)
@test_utils.test(require=qd.extension.data64, arch=qd.gpu, default_fp=qd.f64, fast_math=False)
def test_inverse_large_f64(n, factory):
    _test_inverse_at_size(n, qd.f64, factory)


@test_utils.test()
def test_matrix_factories():
    a = qd.Vector.field(3, dtype=qd.i32, shape=3)
    b = qd.Matrix.field(2, 2, dtype=qd.f32, shape=2)
    c = qd.Matrix.field(2, 3, dtype=qd.f32, shape=2)

    @qd.kernel
    def fill():
        b[0] = qd.Matrix.identity(qd.f32, 2)
        b[1] = qd.math.rotation2d(math.pi / 3)
        c[0] = qd.Matrix.zero(qd.f32, 2, 3)
        c[1] = qd.Matrix.one(qd.f32, 2, 3)
        for i in qd.static(range(3)):
            a[i] = qd.Vector.unit(3, i)

    fill()

    for i in range(3):
        for j in range(3):
            assert a[i][j] == int(i == j)

    sqrt3o2 = math.sqrt(3) / 2
    assert b[0].to_numpy() == test_utils.approx(np.eye(2))
    assert b[1].to_numpy() == test_utils.approx(np.array([[0.5, -sqrt3o2], [sqrt3o2, 0.5]]))
    assert c[0].to_numpy() == test_utils.approx(np.zeros((2, 3)))
    assert c[1].to_numpy() == test_utils.approx(np.ones((2, 3)))


@test_utils.test()
def test_init_matrix_from_vectors():
    m1 = qd.Matrix.field(3, 3, dtype=qd.f32, shape=(3))
    m2 = qd.Matrix.field(3, 3, dtype=qd.f32, shape=(3))
    m3 = qd.Matrix.field(3, 3, dtype=qd.f32, shape=(3))
    m4 = qd.Matrix.field(3, 3, dtype=qd.f32, shape=(3))

    @qd.kernel
    def fill():
        for i in range(3):
            a = qd.Vector([1.0, 4.0, 7.0])
            b = qd.Vector([2.0, 5.0, 8.0])
            c = qd.Vector([3.0, 6.0, 9.0])
            m1[i] = qd.Matrix.rows([a, b, c])
            m2[i] = qd.Matrix.cols([a, b, c])
            m3[i] = qd.Matrix.rows([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
            m4[i] = qd.Matrix.cols([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])

    fill()

    for j in range(3):
        for i in range(3):
            assert m1[0][i, j] == int(i + 3 * j + 1)
            assert m2[0][j, i] == int(i + 3 * j + 1)
            assert m3[0][i, j] == int(i + 3 * j + 1)
            assert m4[0][j, i] == int(i + 3 * j + 1)


@test_utils.test()
def test_any_all():
    a = qd.Matrix.field(2, 2, dtype=qd.i32, shape=())
    b = qd.field(dtype=qd.i32, shape=())
    c = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def func():
        b[None] = any(a[None])
        c[None] = all(a[None])

    for i in range(2):
        for j in range(2):
            a[None][0, 0] = i
            a[None][1, 0] = j
            a[None][1, 1] = i
            a[None][0, 1] = j

            func()
            if i == 1 or j == 1:
                assert b[None] == 1
            else:
                assert b[None] == 0

            if i == 1 and j == 1:
                assert c[None] == 1
            else:
                assert c[None] == 0


@test_utils.test()
def test_min_max():
    a = qd.Matrix.field(2, 2, dtype=qd.i32, shape=())
    b = qd.field(dtype=qd.i32, shape=())
    c = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def func():
        b[None] = a[None].max()
        c[None] = a[None].min()

    for i in range(2):
        for j in range(2):
            a[None][0, 0] = i
            a[None][1, 0] = j
            a[None][1, 1] = i
            a[None][0, 1] = j

            func()
            assert b[None] == max(i, j)
            assert c[None] == min(i, j)


# must not throw any error:
@test_utils.test()
def test_matrix_list_assign():
    m = qd.Matrix.field(2, 2, dtype=qd.i32, shape=(2, 2, 1))
    v = qd.Vector.field(2, dtype=qd.i32, shape=(2, 2, 1))

    m[1, 0, 0] = [[4, 3], [6, 7]]
    v[1, 0, 0] = [8, 4]

    assert np.allclose(m.to_numpy()[1, 0, 0, :, :], np.array([[4, 3], [6, 7]]))
    assert np.allclose(v.to_numpy()[1, 0, 0, :], np.array([8, 4]))

    @qd.kernel
    def func():
        m[1, 0, 0] = [[1, 2], [3, 4]]
        v[1, 0, 0] = [5, 6]
        m[1, 0, 0] += [[1, 2], [3, 4]]
        v[1, 0, 0] += [5, 6]

    func()
    assert np.allclose(m.to_numpy()[1, 0, 0, :, :], np.array([[2, 4], [6, 8]]))
    assert np.allclose(v.to_numpy()[1, 0, 0, :], np.array([10, 12]))


@test_utils.test(arch=get_host_arch_list())
def test_vector_xyzw_accessor():
    u = qd.Vector.field(2, dtype=qd.i32, shape=(2, 2, 1))
    v = qd.Vector.field(4, dtype=qd.i32, shape=(2, 2, 1))

    u[1, 0, 0].y = 3
    v[1, 0, 0].z = 0
    v[1, 0, 0].w = 4

    @qd.kernel
    def func():
        u[1, 0, 0].x = 8 * u[1, 0, 0].y
        v[1, 0, 0].z = 1 - v[1, 0, 0].w
        v[1, 0, 0].x = 6

    func()
    assert u[1, 0, 0].x == 24
    assert u[1, 0, 0].y == 3
    assert v[1, 0, 0].z == -3
    assert v[1, 0, 0].w == 4
    assert np.allclose(v.to_numpy()[1, 0, 0, :], np.array([6, 0, -3, 4]))


@test_utils.test(arch=get_host_arch_list())
def test_diag():
    m1 = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())

    @qd.kernel
    def fill():
        m1[None] = qd.Matrix.diag(dim=3, val=1.4)

    fill()

    for i in range(3):
        for j in range(3):
            if i == j:
                assert m1[None][i, j] == test_utils.approx(1.4)
            else:
                assert m1[None][i, j] == 0.0
