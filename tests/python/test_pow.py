import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsRuntimeError

from tests import test_utils


def _test_pow_f(dt):
    z = qd.field(dt, shape=())

    @qd.kernel
    def func(x: dt, y: dt):
        z[None] = x**y

    for x in [0.5, 1, 1.5, 2, 6.66]:
        for y in [-2, -1, -0.3, 0, 0.5, 1, 1.4, 2.6]:
            func(x, y)
            assert z[None] == pytest.approx(x**y)


def _test_pow_i(dt):
    z = qd.field(dt, shape=())

    @qd.kernel
    def func(x: dt, y: qd.template()):
        z[None] = x**y

    for x in range(-5, 5):
        for y in range(0, 4):
            func(x, y)
            assert z[None] == x**y


@test_utils.test()
def test_pow_f32():
    _test_pow_f(qd.f32)


@test_utils.test(require=qd.extension.data64)
def test_pow_f64():
    _test_pow_f(qd.f64)


@test_utils.test()
def test_pow_i32():
    _test_pow_i(qd.i32)


@test_utils.test(require=qd.extension.data64)
def test_pow_i64():
    _test_pow_i(qd.i64)


def _ipow_negative_exp(dt):
    z = qd.field(dt, shape=())

    @qd.kernel
    def foo(x: dt, y: qd.template()):
        z[None] = x**y

    with pytest.raises(QuadrantsRuntimeError):
        foo(10, -10)


@test_utils.test(
    debug=True,
    advanced_optimization=False,
    exclude=[qd.vulkan, qd.metal],
)
def test_ipow_negative_exp_i32():
    _ipow_negative_exp(qd.i32)


@test_utils.test(
    debug=True,
    advanced_optimization=False,
    require=qd.extension.data64,
    exclude=[qd.vulkan, qd.metal],
)
def test_ipow_negative_exp_i64():
    _ipow_negative_exp(qd.i64)


def _test_pow_int_base_int_exp(dt_base, dt_exp):
    z = qd.field(dt_base, shape=())

    @qd.kernel
    def func(x: dt_base, y: dt_exp):
        z[None] = x**y

    for x in range(-5, 5):
        for y in range(0, 10):
            func(x, y)
            assert z[None] == x**y


@test_utils.test()
def test_pow_int_base_int_exp_32():
    _test_pow_int_base_int_exp(qd.i32, qd.i32)


@pytest.mark.parametrize("dt_base, dt_exp", [(qd.i32, qd.i64), (qd.i64, qd.i64), (qd.i64, qd.i32)])
@test_utils.test(require=qd.extension.data64)
def test_pow_int_base_int_exp_64(dt_base, dt_exp):
    _test_pow_int_base_int_exp(dt_base, dt_exp)


def _test_pow_float_base_int_exp(dt_base, dt_exp):
    z = qd.field(dt_base, shape=())

    @qd.kernel
    def func(x: dt_base, y: dt_exp):
        z[None] = x**y

    for x in [-6.66, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 6.66]:
        for y in range(-10, 10):
            func(x, y)
            assert z[None] == pytest.approx(x**y)


@test_utils.test()
def test_pow_float_base_int_exp_32():
    _test_pow_float_base_int_exp(qd.f32, qd.i32)


@pytest.mark.parametrize("dt_base, dt_exp", [(qd.f64, qd.i32), (qd.f32, qd.i64), (qd.f64, qd.i64)])
@test_utils.test(require=qd.extension.data64)
def test_pow_float_base_int_exp_64(dt_base, dt_exp):
    _test_pow_float_base_int_exp(dt_base, dt_exp)
