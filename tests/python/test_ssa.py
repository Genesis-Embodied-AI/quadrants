"""
SSA violation edge-case regression test.
1. Ensure working well when computation result is assigned to self.
2. Prevent duplicate-evaluation on expression with side-effect like random.
"""

import math

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test()
def test_matrix_self_assign():
    a = qd.Vector.field(2, qd.f32, ())
    b = qd.Matrix.field(2, 2, qd.f32, ())
    c = qd.Vector.field(2, qd.f32, ())

    @qd.kernel
    def func():
        a[()] = a[()].normalized()
        b[()] = b[()].transpose()
        c[()] = qd.Vector([c[()][1], c[()][0]])

    inv_sqrt2 = 1 / math.sqrt(2)

    a[()] = [1, 1]
    b[()] = [[1, 2], [3, 4]]
    c[()] = [2, 3]
    func()
    assert a[()] == qd.Vector([inv_sqrt2, inv_sqrt2])
    assert b[()] == qd.Matrix([[1, 3], [2, 4]])
    assert c[()] == qd.Vector([3, 2])


@test_utils.test()
def test_random_vector_dup_eval():
    a = qd.Vector.field(2, qd.f32, ())

    @qd.kernel
    def func():
        a[()] = qd.Vector([qd.random(), 1]).normalized()

    for i in range(4):
        func()
        assert a[()].norm_sqr() == test_utils.approx(1)


@test_utils.test()
def test_func_argument_dup_eval():
    @qd.func
    def func(a, t):
        return a * t - a

    @qd.kernel
    def kern(t: qd.f32) -> qd.f32:
        return func(qd.random(), t)

    for i in range(4):
        assert kern(1.0) == 0.0


@test_utils.test()
def test_func_random_argument_dup_eval():
    @qd.func
    def func(a):
        return qd.Vector([qd.cos(a), qd.sin(a)])

    @qd.kernel
    def kern() -> qd.f32:
        return func(qd.random()).norm_sqr()

    for i in range(4):
        assert kern() == test_utils.approx(1.0, rel=5e-5)
