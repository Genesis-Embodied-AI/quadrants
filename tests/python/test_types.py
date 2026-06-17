import pytest

import quadrants as qd
from quadrants.lang import impl
from quadrants.types import primitive_types

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
    # https://github.com/taichi-dev/taichi/issues/2060
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


@test_utils.test(default_fp=qd.f32)
def test_default_fp_canonical_identity():
    """Regression: get_runtime().default_fp must stay identity-equal to the registered primitive singleton.

    qd.init(default_fp=qd.f32) deep-copies the dtype (misc.py), so without canonicalization in set_default_fp the
    stored default_fp is == qd.f32 (same hash) but has an id outside primitive_types.type_ids.  That silently breaks
    id-based type recognition for any code that resolves a dtype via get_runtime().default_fp and then uses it as a
    type -- e.g. an in-kernel type construction or a kernel/func annotation (this is what broke the simt tile
    proxies).
    """
    dfp = impl.get_runtime().default_fp
    assert dfp == qd.f32
    assert id(dfp) in primitive_types.type_ids

    @qd.kernel
    def use_as_call() -> qd.f32:
        return dfp(2.5)  # type construction; falls through to a raw call (and raises) if id not registered

    assert use_as_call() == pytest.approx(2.5)

    @qd.kernel
    def use_as_annotation(x: dfp) -> dfp:  # annotation must be recognized as a primitive type
        return x * 2.0

    assert use_as_annotation(3.0) == pytest.approx(6.0)


@test_utils.test(default_ip=qd.i32)
def test_default_ip_canonical_identity():
    """Regression: get_runtime().default_ip must stay identity-equal to the registered primitive singleton.

    Same root cause as test_default_fp_canonical_identity, for the default integer type.
    """
    dip = impl.get_runtime().default_ip
    assert dip == qd.i32
    assert id(dip) in primitive_types.type_ids

    @qd.kernel
    def use_as_call() -> qd.i32:
        return dip(7)

    assert use_as_call() == 7

    @qd.kernel
    def use_as_annotation(x: dip) -> dip:
        return x * 2

    assert use_as_annotation(5) == 10
