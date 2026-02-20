import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


@pytest.mark.parametrize(
    "arg1,a,arg2,b,arg3,c",
    [
        (qd.i32, 10, qd.i32, 3, qd.f32, 3),
        (qd.f32, 10, qd.f32, 3, qd.f32, 3),
        (qd.i32, 10, qd.f32, 3, qd.f32, 3),
        (qd.f32, 10, qd.i32, 3, qd.f32, 3),
        (qd.i32, -10, qd.i32, 3, qd.f32, -4),
        (qd.f32, -10, qd.f32, 3, qd.f32, -4),
        (qd.i32, -10, qd.f32, 3, qd.f32, -4),
        (qd.f32, -10, qd.i32, 3, qd.f32, -4),
        (qd.i32, 10, qd.i32, -3, qd.f32, -4),
        (qd.f32, 10, qd.f32, -3, qd.f32, -4),
        (qd.i32, 10, qd.f32, -3, qd.f32, -4),
        (qd.f32, 10, qd.i32, -3, qd.f32, -4),
    ],
)
@test_utils.test()
def test_floor_div(arg1, a, arg2, b, arg3, c):
    z = qd.field(arg3, shape=())

    @qd.kernel
    def func(x: arg1, y: arg2):
        z[None] = x // y

    func(a, b)
    assert z[None] == c


@pytest.mark.parametrize(
    "arg1,a,arg2,b,arg3,c",
    [
        (qd.i32, 3, qd.i32, 2, qd.f32, 1.5),
        (qd.f32, 3, qd.f32, 2, qd.f32, 1.5),
        (qd.i32, 3, qd.f32, 2, qd.f32, 1.5),
        (qd.f32, 3, qd.i32, 2, qd.f32, 1.5),
        (qd.f32, 3, qd.i32, 2, qd.i32, 1),
        (qd.i32, -3, qd.i32, 2, qd.f32, -1.5),
        (qd.f32, -3, qd.f32, 2, qd.f32, -1.5),
        (qd.i32, -3, qd.f32, 2, qd.f32, -1.5),
        (qd.f32, -3, qd.i32, 2, qd.f32, -1.5),
        (qd.f32, -3, qd.i32, 2, qd.i32, -1),
    ],
)
@test_utils.test()
def test_true_div(arg1, a, arg2, b, arg3, c):
    z = qd.field(arg3, shape=())

    @qd.kernel
    def func(x: arg1, y: arg2):
        z[None] = x / y

    func(a, b)
    assert z[None] == c


@test_utils.test()
def test_div_default_ip():
    impl.get_runtime().set_default_ip(qd.i64)
    z = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 1e15 + 1e9
        z[None] = a // 1e10

    func()
    assert z[None] == 100000


@test_utils.test()
def test_floor_div_pythonic():
    z = qd.field(qd.i32, shape=())

    @qd.kernel
    def func(x: qd.i32, y: qd.i32):
        z[None] = x // y

    for i in range(-10, 11):
        for j in range(-10, 11):
            if j != 0:
                func(i, j)
                assert z[None] == i // j
