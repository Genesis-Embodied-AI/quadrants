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


@pytest.mark.parametrize(
    "x,m",
    [
        (180.0, 180),
        (360.0, 180),
        (540.0, 180),
        (256.0, 64),
    ],
)
@test_utils.test()
def test_exact_div_field_int(x, m):
    # Regression test for quadrants#749. An exactly-divisible float / (runtime int) must be correctly rounded. Under
    # fast_math on AMDGPU the afn (ApproxFunc) flag lowered `fdiv` to an approximate v_rcp_f32 reciprocal, so
    # 180.0 / 180 returned 0.99999994. The divisor is read from a field so it stays a runtime value and is not
    # constant-folded exactly at compile time. No-op on CPU/CUDA, where fdiv ignores afn.
    divisor = qd.field(qd.i32, shape=())
    result = qd.field(qd.f32, shape=())

    @qd.kernel
    def func(a: qd.f32):
        result[None] = a / divisor[None]

    divisor[None] = m
    func(x)
    assert result[None] == x / m


@pytest.mark.parametrize(
    "x,m",
    [
        (180.0, 180),
        (360.0, 180),
        (540.0, 180),
    ],
)
@test_utils.test()
def test_floor_exact_div_field_int(x, m):
    # Companion to test_exact_div_field_int: the approximate reciprocal made floor(a / b) land one cell low
    # (floor(180.0 / 180) == 0.0 instead of 1.0), silently corrupting spatial hashing and grid indexing. See
    # quadrants#749.
    divisor = qd.field(qd.i32, shape=())
    result = qd.field(qd.f32, shape=())

    @qd.kernel
    def func(a: qd.f32):
        result[None] = qd.floor(a / divisor[None], qd.f32)

    divisor[None] = m
    func(x)
    assert result[None] == float(x // m)
