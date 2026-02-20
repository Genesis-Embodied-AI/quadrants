import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(debug=True)
def test_assign_basic():
    @qd.kernel
    def func_basic():
        a = 1
        assert a == 1

    func_basic()


@test_utils.test(debug=True)
def test_assign_unpack():
    @qd.kernel
    def func_unpack():
        (a, b) = (1, 2)
        assert a == 1
        assert b == 2

    func_unpack()


@test_utils.test(debug=True)
def test_assign_chained():
    @qd.kernel
    def func_chained():
        a = b = 1
        assert a == 1
        assert b == 1

    func_chained()


@test_utils.test(debug=True)
def test_assign_chained_unpack():
    @qd.kernel
    def func_chained_unpack():
        (a, b) = (c, d) = (1, 2)
        assert a == 1
        assert b == 2
        assert c == 1
        assert d == 2

    func_chained_unpack()


@test_utils.test(debug=True)
def test_assign_assign():
    @qd.kernel
    def func_assign():
        a = 0
        a = 1
        assert a == 1

    func_assign()


@test_utils.test(debug=True)
def test_assign_ann():
    @qd.kernel
    def func_ann():
        a: qd.i32 = 1
        b: qd.f32 = a
        assert a == 1
        assert b == 1.0

    func_ann()


@test_utils.test()
def test_assign_ann_over():
    @qd.kernel
    def func_ann_over():
        my_int = qd.i32
        d: my_int = 2
        d: qd.f32 = 2.0

    with pytest.raises(qd.QuadrantsCompilationError):
        func_ann_over()


@test_utils.test(debug=True)
def test_assign_chained_involve_self():
    @qd.kernel
    def foo():
        a = 1
        b = 1
        a = b = a + b
        assert a == 2
        assert b == 2

    foo()
