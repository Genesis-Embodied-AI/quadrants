import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils

_QD_TYPES = [qd.i8, qd.i16, qd.i32, qd.u8, qd.u16, qd.u32, qd.f32]
_QD_64_TYPES = [qd.i64, qd.u64, qd.f64]


def _test_type_assign_argument(dt):
    x = qd.field(dt, shape=())

    @qd.kernel
    def func(value: dt):
        x[None] = value

    func(3)
    assert x[None] == 3


@pytest.mark.parametrize("dt", _QD_TYPES)
@test_utils.test(exclude=[qd.vulkan])
def test_type_assign_argument(dt):
    _test_type_assign_argument(dt)


@pytest.mark.parametrize("dt", _QD_64_TYPES)
@test_utils.test(exclude=[qd.vulkan], require=qd.extension.data64)
def test_type_assign_argument64(dt):
    _test_type_assign_argument(dt)


def _test_type_operator(dt):
    x = qd.field(dt, shape=())
    y = qd.field(dt, shape=())
    add = qd.field(dt, shape=())
    mul = qd.field(dt, shape=())

    @qd.kernel
    def func():
        add[None] = x[None] + y[None]
        mul[None] = x[None] * y[None]

    for i in range(0, 3):
        for j in range(0, 3):
            x[None] = i
            y[None] = j
            func()
            assert add[None] == x[None] + y[None]
            assert mul[None] == x[None] * y[None]


@pytest.mark.parametrize("dt", _QD_TYPES)
@test_utils.test(exclude=[qd.vulkan])
def test_type_operator(dt):
    _test_type_operator(dt)


@pytest.mark.parametrize("dt", _QD_64_TYPES)
@test_utils.test(exclude=[qd.vulkan], require=qd.extension.data64)
def test_type_operator64(dt):
    _test_type_operator(dt)


def _test_type_field(dt):
    x = qd.field(dt, shape=(3, 2))

    @qd.kernel
    def func(i: qd.i32, j: qd.i32):
        x[i, j] = 3

    for i in range(0, 3):
        for j in range(0, 2):
            func(i, j)
            assert x[i, j] == 3


@pytest.mark.parametrize("dt", _QD_TYPES)
@test_utils.test(exclude=[qd.vulkan])
def test_type_field(dt):
    _test_type_field(dt)


@pytest.mark.parametrize("dt", _QD_64_TYPES)
@test_utils.test(exclude=[qd.vulkan], require=qd.extension.data64)
def test_type_field64(dt):
    _test_type_field(dt)


def _test_overflow(dt, n):
    a = qd.field(dt, shape=())
    b = qd.field(dt, shape=())
    c = qd.field(dt, shape=())

    @qd.kernel
    def func():
        c[None] = a[None] + b[None]

    a[None] = 2**n // 3
    b[None] = 2**n // 3

    func()

    assert a[None] == 2**n // 3
    assert b[None] == 2**n // 3

    if qd.types.is_signed(dt):
        assert c[None] == 2**n // 3 * 2 - (2**n)  # overflows
    else:
        assert c[None] == 2**n // 3 * 2  # does not overflow


@pytest.mark.parametrize(
    "dt,n",
    [
        (qd.i8, 8),
        (qd.u8, 8),
        (qd.i16, 16),
        (qd.u16, 16),
        (qd.i32, 32),
        (qd.u32, 32),
    ],
)
@test_utils.test(exclude=[qd.vulkan])
def test_overflow(dt, n):
    _test_overflow(dt, n)


@pytest.mark.parametrize(
    "dt,n",
    [
        (qd.i64, 64),
        (qd.u64, 64),
    ],
)
@test_utils.test(exclude=[qd.vulkan], require=qd.extension.data64)
def test_overflow64(dt, n):
    _test_overflow(dt, n)


@pytest.mark.parametrize(
    "dt,val",
    [
        (qd.u32, 0xFFFFFFFF),
        (qd.u64, 0xFFFFFFFFFFFFFFFF),
    ],
)
@test_utils.test(require=qd.extension.data64)
def test_uint_max(dt, val):
    # https://github.com/taichi-dev/quadrants/issues/2060
    impl.get_runtime().default_ip = dt
    N = 16
    f = qd.field(dt, shape=N)

    @qd.kernel
    def run():
        for i in f:
            f[i] = val

    run()
    fs = f.to_numpy()
    for f in fs:
        assert f == val
