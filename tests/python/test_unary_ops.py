import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsTypeError

from tests import test_utils


def _test_op(dt, quadrants_op, np_op):
    print("arch={} default_fp={}".format(qd.lang.impl.current_cfg().arch, qd.lang.impl.current_cfg().default_fp))
    n = 4
    val = qd.field(dt, shape=n)

    def f(i):
        return i * 0.1 + 0.4

    @qd.kernel
    def fill():
        for i in range(n):
            val[i] = quadrants_op(qd.func(f)(qd.cast(i, dt)))

    fill()

    # check that it is double precision
    for i in range(n):
        if dt == qd.f64:
            assert abs(np_op(float(f(i))) - val[i]) < 1e-15
        else:
            assert abs(np_op(float(f(i))) - val[i]) < 1e-6 if qd.lang.impl.current_cfg().arch != qd.vulkan else 1e-5


op_pairs = [
    (qd.sin, np.sin),
    (qd.cos, np.cos),
    (qd.asin, np.arcsin),
    (qd.acos, np.arccos),
    (qd.tan, np.tan),
    (qd.tanh, np.tanh),
    (qd.exp, np.exp),
    (qd.log, np.log),
]


@pytest.mark.parametrize("quadrants_op,np_op", op_pairs)
@test_utils.test(default_fp=qd.f32)
def test_trig_f32(quadrants_op, np_op):
    _test_op(qd.f32, quadrants_op, np_op)


@pytest.mark.parametrize("quadrants_op,np_op", op_pairs)
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64)
def test_trig_f64(quadrants_op, np_op):
    _test_op(qd.f64, quadrants_op, np_op)


@test_utils.test(print_full_traceback=False)
def test_bit_not_invalid():
    @qd.kernel
    def test(x: qd.f32) -> qd.i32:
        return ~x

    with pytest.raises(QuadrantsTypeError, match=r"takes integral inputs only"):
        test(1.0)


@test_utils.test(print_full_traceback=False)
def test_logic_not_invalid():
    @qd.kernel
    def test(x: qd.f32) -> qd.i32:
        return not x

    with pytest.raises(QuadrantsTypeError, match=r"takes integral inputs only"):
        test(1.0)


@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
def test_frexp():
    if qd.lang.impl.current_cfg().arch == qd.amdgpu:
        pytest.xfail("BUG: AMDGPU codegen does not lower this op.")

    @qd.kernel
    def get_frac(x: qd.f32) -> qd.f32:
        a, b = qd.frexp(x)
        return a

    assert test_utils.allclose(get_frac(1.4), 0.7)

    @qd.kernel
    def get_exp(x: qd.f32) -> qd.i32:
        a, b = qd.frexp(x)
        return b

    assert get_exp(1.4) == 1


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan])
def test_popcnt():
    if qd.lang.impl.current_cfg().arch == qd.amdgpu:
        pytest.xfail("BUG: AMDGPU codegen does not lower this op.")

    @qd.kernel
    def test_i32(x: qd.int32) -> qd.int32:
        return qd.math.popcnt(x)

    @qd.kernel
    def test_i64(x: qd.int64) -> qd.int32:
        return qd.math.popcnt(x)

    @qd.kernel
    def test_u32(x: qd.uint32) -> qd.int32:
        return qd.math.popcnt(x)

    @qd.kernel
    def test_u64(x: qd.uint64) -> qd.int32:
        return qd.math.popcnt(x)

    assert test_i32(100) == 3
    assert test_i32(1000) == 6
    assert test_i32(10000) == 5
    assert test_i64(100) == 3
    assert test_i64(1000) == 6
    assert test_i64(10000) == 5
    assert test_u32(100) == 3
    assert test_u32(1000) == 6
    assert test_u32(10000) == 5
    assert test_u64(100) == 3
    assert test_u64(1000) == 6
    assert test_u64(10000) == 5


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda, qd.amdgpu, qd.vulkan])
def test_clz():
    @qd.kernel
    def test_i32(x: qd.int32) -> qd.int32:
        return qd.math.clz(x)

    # assert test_i32(0) == 32
    assert test_i32(1) == 31
    assert test_i32(2) == 30
    assert test_i32(3) == 30
    assert test_i32(4) == 29
    assert test_i32(5) == 29
    assert test_i32(1023) == 22
    assert test_i32(1024) == 21


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan])
def test_clz_u32():
    """``qd.math.clz`` on ``u32`` — covers values with the top bit set (which silently broke on SPIR-V before it
    started dispatching ``FindUMsb`` for unsigned inputs, and was rejected outright on CUDA before the intrinsic was
    extended past i32).  Metal is excluded because Metal goes through SPIR-V which has the 64-bit caveat — the 32-bit
    path works on Metal too in principle, included separately if needed."""

    @qd.kernel
    def test_u32(x: qd.uint32) -> qd.int32:
        return qd.math.clz(x)

    assert test_u32(1) == 31
    assert test_u32(0x80000000) == 0
    assert test_u32(0xFFFFFFFF) == 0
    assert test_u32(0x40000000) == 1
    assert test_u32(0x00010000) == 15
    assert test_u32(0x00000001) == 31


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_clz_i64():
    """``qd.math.clz`` on ``i64`` — exercises the 64-bit intrinsic path (``__nv_clzll`` on CUDA, polymorphic
    ``llvm.ctlz`` on AMDGPU).  Vulkan / Metal are excluded because the SPIR-V lowering is hard-coded to 32 bits."""

    @qd.kernel
    def test(x: qd.int64) -> qd.int32:
        return qd.math.clz(x)

    assert test(1) == 63
    assert test(2) == 62
    assert test(1 << 32) == 31
    assert test(1 << 62) == 1


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_clz_u64():
    """``qd.math.clz`` on ``u64`` — covers values with the top bit set (rejected outright on CUDA before the intrinsic
    was extended).  SPIR-V excluded for the same 32-bit reason as `test_clz_i64`."""

    @qd.kernel
    def test(x: qd.uint64) -> qd.int32:
        return qd.math.clz(x)

    assert test(1) == 63
    assert test(1 << 63) == 0
    assert test(0xFFFFFFFFFFFFFFFF) == 0
    assert test(1 << 32) == 31


@test_utils.test(arch=[qd.metal])
def test_popcnt_metal():
    @qd.kernel
    def test_i32(x: qd.int32) -> qd.int32:
        return qd.math.popcnt(x)

    @qd.kernel
    def test_u32(x: qd.uint32) -> qd.int32:
        return qd.math.popcnt(x)

    assert test_i32(100) == 3
    assert test_i32(1000) == 6
    assert test_i32(10000) == 5
    assert test_u32(100) == 3
    assert test_u32(1000) == 6
    assert test_u32(10000) == 5


@test_utils.test()
def test_sign():
    @qd.kernel
    def foo(val: qd.f32) -> qd.f32:
        return qd.math.sign(val)

    assert foo(0.5) == 1.0
    assert foo(-0.5) == -1.0
    assert foo(0.0) == 0.0
