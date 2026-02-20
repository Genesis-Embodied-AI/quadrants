import os
import platform

import pytest

import quadrants as qd

from tests import test_utils

if os.name == "nt":
    pytest.skip(
        "Skipping on windows because fflush issues, " "see https://github.com/taichi-dev/quadrants/issues/6179",
        allow_module_level=True,
    )


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan])
def test_no_debug(capfd):
    capfd.readouterr()

    @qd.kernel
    def foo() -> qd.i32:
        a = qd.i32(1073741824)
        b = qd.i32(1073741824)
        return a + b

    foo()
    qd.sync()
    captured = capfd.readouterr().out
    assert "Addition overflow detected" not in captured
    assert "return a + b" not in captured


def supports_overflow(arch):
    return arch != qd.vulkan or platform.system() != "Darwin"  # Vulkan on macOS does not have a validation layer.


add_table = [
    (qd.i8, 2**6),
    (qd.u8, 2**7),
    (qd.i16, 2**14),
    (qd.u16, 2**15),
    (qd.i32, 2**30),
    (qd.u32, 2**31),
    (qd.i64, 2**62),
    (qd.u64, 2**63),
]


@pytest.mark.parametrize("ty,num", add_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_add_overflow(capfd, ty, num):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(x: ty) -> ty:
        a = ty(x)
        b = ty(num)
        return a + b

    foo(num)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Addition overflow detected" in captured
    assert "return a + b" in captured


@pytest.mark.parametrize("ty,num", add_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_add_no_overflow(capfd, ty, num):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo() -> ty:
        a = ty(num)
        b = ty(num - 1)
        return a + b

    foo()
    qd.sync()
    captured = capfd.readouterr().out
    assert "Addition overflow detected" not in captured
    assert "return a + b" not in captured


sub_table = [
    (qd.i8, 2**6),
    (qd.i16, 2**14),
    (qd.i32, 2**30),
    (qd.i64, 2**62),
]


@pytest.mark.parametrize("ty,num", sub_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_sub_overflow_i(capfd, ty, num):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(num: ty) -> ty:
        a = ty(num)
        b = ty(-num)
        return a - b

    foo(num)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" in captured
    assert "return a - b" in captured


@pytest.mark.parametrize("ty,num", sub_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_sub_no_overflow_i(capfd, ty, num):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(num: ty) -> ty:
        a = ty(num)
        b = ty(-num + 1)
        return a - b

    foo(num)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" not in captured
    assert "return a - b" not in captured


@pytest.mark.parametrize("ty", [qd.u8, qd.u16, qd.u32, qd.u64])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_sub_overflow_u(capfd, ty):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(x: ty, y: ty) -> ty:
        a = ty(x)
        b = ty(y)
        return a - b

    foo(1, 2)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" in captured
    assert "return a - b" in captured


@pytest.mark.parametrize("ty", [qd.u8, qd.u16, qd.u32, qd.u64])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_sub_no_overflow_u(capfd, ty):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(x: ty) -> ty:
        a = ty(x)
        b = ty(x)
        return a - b

    foo(1)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Subtraction overflow detected" not in captured
    assert "return a - b" not in captured


mul_table = [
    (qd.i8, 2**4, 2**3),
    (qd.u8, 2**4, 2**4),
    (qd.i16, 2**8, 2**7),
    (qd.u16, 2**8, 2**8),
    (qd.i32, 2**16, 2**15),
    (qd.u32, 2**16, 2**16),
    (qd.i64, 2**32, 2**31),
    (qd.u64, 2**32, 2**32),
]


@pytest.mark.parametrize("ty,num1,num2", mul_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_mul_overflow(capfd, ty, num1, num2):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    # 64-bit Multiplication overflow detection does not function correctly on old drivers.
    # See https://github.com/taichi-dev/quadrants/issues/6303
    if qd.lang.impl.current_cfg().arch == qd.vulkan and id(ty) in [
        id(qd.i64),
        id(qd.u64),
    ]:
        return
    capfd.readouterr()

    @qd.kernel
    def foo(num1: ty, num2: ty) -> ty:
        a = ty(num1 + 1)
        b = ty(num2 + 1)
        return a * b

    foo(num1, num2)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Multiplication overflow detected" in captured
    assert "return a * b" in captured


@pytest.mark.parametrize("ty,num1,num2", mul_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_mul_no_overflow(capfd, ty, num1, num2):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(num1: ty, num2: ty) -> ty:
        a = ty(num1 + 1)
        b = ty(num2 - 1)
        return a * b

    foo(num1, num2)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Multiplication overflow detected" not in captured
    assert "return a * b" not in captured


shl_table = [
    (qd.i8, 6),
    (qd.u8, 7),
    (qd.i16, 14),
    (qd.u16, 15),
    (qd.i32, 30),
    (qd.u32, 31),
    (qd.i64, 62),
    (qd.u64, 63),
]


@pytest.mark.parametrize("ty,num", shl_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_shl_overflow(capfd, ty, num):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(num: ty) -> ty:
        a = ty(2)
        b = num
        return a << b

    foo(num)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Shift left overflow detected" in captured
    assert "return a << b" in captured


@pytest.mark.parametrize("ty,num", shl_table)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], debug=True)
def test_shl_no_overflow(capfd, ty, num):
    if not supports_overflow(qd.lang.impl.current_cfg().arch):
        pytest.skip("current arch doesnt support overflow")
    capfd.readouterr()

    @qd.kernel
    def foo(num: ty) -> ty:
        a = ty(2)
        b = num - 1
        return a << b

    foo(num)
    qd.sync()
    captured = capfd.readouterr().out
    assert "Shift left overflow detected" not in captured
    assert "return a << b" not in captured
