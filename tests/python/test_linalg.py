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
