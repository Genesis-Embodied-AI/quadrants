import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize(
    "a,b",
    [
        (10, 3),
        (-10, 3),
        (10, -3),
        (-10, -3),
    ],
)
@test_utils.test()
def test_py_style_mod(a, b):
    z = qd.field(qd.i32, shape=())

    @qd.kernel
    def func(x: qd.i32, y: qd.i32):
        z[None] = x % y

    func(a, b)
    assert z[None] == a % b


@pytest.mark.parametrize(
    "a,b",
    [
        (10, 3),
        (-10, 3),
        (10, -3),
        (-10, -3),
    ],
)
@test_utils.test()
def test_c_style_mod(a, b):
    z = qd.field(qd.i32, shape=())

    @qd.kernel
    def func(x: qd.i32, y: qd.i32):
        z[None] = qd.raw_mod(x, y)

    func(a, b)
    assert z[None] == _c_mod(a, b)


def _c_mod(a, b):
    return a - b * int(float(a) / b)


@test_utils.test()
def test_mod_scan():
    z = qd.field(qd.i32, shape=())
    w = qd.field(qd.i32, shape=())

    @qd.kernel
    def func(x: qd.i32, y: qd.i32):
        z[None] = x % y
        w[None] = qd.raw_mod(x, y)

    for i in range(-10, 11):
        for j in range(-10, 11):
            if j != 0:
                func(i, j)
                assert z[None] == i % j
                assert w[None] == _c_mod(i, j)


@test_utils.test()
def test_py_style_float_const_mod_one():
    @qd.kernel
    def func() -> qd.f32:
        a = 0.5
        return a % 1

    assert func() == 0.5


@test_utils.test()
def test_py_style_unsigned_mod():
    @qd.kernel
    def func() -> qd.u32:
        return qd.u32(3583196299) % qd.u32(524288)

    assert func() == 212107
