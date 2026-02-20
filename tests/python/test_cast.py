import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("dtype", [qd.u8, qd.u16, qd.u32])
@test_utils.test()
def test_cast_uint_to_float(dtype):
    @qd.kernel
    def func(a: dtype) -> qd.f32:
        return qd.cast(a, qd.f32)

    @qd.kernel
    def func_sugar(a: dtype) -> qd.f32:
        return qd.f32(a)

    assert func(255) == func_sugar(255) == 255


@pytest.mark.parametrize("dtype", [qd.u8, qd.u16, qd.u32])
@test_utils.test()
def test_cast_float_to_uint(dtype):
    @qd.kernel
    def func(a: qd.f32) -> dtype:
        return qd.cast(a, dtype)

    @qd.kernel
    def func_sugar(a: qd.f32) -> dtype:
        return dtype(a)

    assert func(255) == func_sugar(255) == 255


@test_utils.test()
def test_cast_f32():
    z = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        z[None] = qd.cast(1e9, qd.f32) / qd.cast(1e6, qd.f32) + 1e-3

    func()
    assert z[None] == 1000


@test_utils.test(require=qd.extension.data64)
def test_cast_f64():
    z = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        z[None] = qd.cast(1e13, qd.f64) / qd.cast(1e10, qd.f64) + 1e-3

    func()
    assert z[None] == 1000


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
def test_cast_default_fp(dtype):
    qd.init(default_fp=dtype)

    @qd.kernel
    def func(x: int, y: int) -> float:
        return qd.cast(x, float) * float(y)

    assert func(23, 4) == pytest.approx(23.0 * 4.0)


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64])
def test_cast_default_ip(dtype):
    qd.init(default_ip=dtype)

    @qd.kernel
    def func(x: float, y: float) -> int:
        return qd.cast(x, int) * int(y)

    # make sure that int(4.6) == 4:
    assert func(23.3, 4.6) == 23 * 4
    if dtype == qd.i64:
        large = 1000000000
        assert func(large, 233) == large * 233
        assert func(233, large) == 233 * large


@test_utils.test()
def test_cast_within_while():
    ret = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        t = 10
        while t > 5:
            t = 1.0
            break
        ret[None] = t

    func()


@test_utils.test()
def test_bit_cast():
    x = qd.field(qd.i32, shape=())
    y = qd.field(qd.f32, shape=())
    z = qd.field(qd.i32, shape=())

    @qd.kernel
    def func1():
        y[None] = qd.bit_cast(x[None], qd.f32)

    @qd.kernel
    def func2():
        z[None] = qd.bit_cast(y[None], qd.i32)

    x[None] = 2333
    func1()
    func2()
    assert z[None] == 2333


@test_utils.test(arch=qd.cpu)
def test_int_extension():
    x = qd.field(dtype=qd.i32, shape=2)
    y = qd.field(dtype=qd.u32, shape=2)

    a = qd.field(dtype=qd.i8, shape=1)
    b = qd.field(dtype=qd.u8, shape=1)

    @qd.kernel
    def run_cast_i32():
        x[0] = qd.cast(a[0], qd.i32)
        x[1] = qd.cast(b[0], qd.i32)

    @qd.kernel
    def run_cast_u32():
        y[0] = qd.cast(a[0], qd.u32)
        y[1] = qd.cast(b[0], qd.u32)

    a[0] = -128
    b[0] = -128

    run_cast_i32()
    assert x[0] == -128
    assert x[1] == 128

    run_cast_u32()
    assert y[0] == 0xFFFFFF80
    assert y[1] == 128


@test_utils.test(arch=qd.cpu)
def test_quant_int_extension():
    x = qd.field(dtype=qd.i32, shape=2)
    y = qd.field(dtype=qd.u32, shape=2)

    qi5 = qd.types.quant.int(5, True, qd.i16)
    qu7 = qd.types.quant.int(7, False, qd.u16)

    a = qd.field(dtype=qi5)
    b = qd.field(dtype=qu7)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(a, b)
    qd.root.place(bitpack)

    @qd.kernel
    def run_cast_int():
        x[0] = qd.cast(a[None], qd.i32)
        x[1] = qd.cast(b[None], qd.i32)

    @qd.kernel
    def run_cast_uint():
        y[0] = qd.cast(a[None], qd.u32)
        y[1] = qd.cast(b[None], qd.u32)

    a[None] = -16
    b[None] = -64

    run_cast_int()
    assert x[0] == -16
    assert x[1] == 64

    run_cast_uint()
    assert y[0] == 0xFFFFFFF0
    assert y[1] == 64
