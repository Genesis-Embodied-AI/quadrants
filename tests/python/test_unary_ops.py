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


@test_utils.test()
def test_popcnt():
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


@test_utils.test()
def test_clz():
    @qd.kernel
    def test_i32(x: qd.int32) -> qd.int32:
        return qd.math.clz(x)

    @qd.kernel
    def test_u32(x: qd.uint32) -> qd.int32:
        return qd.math.clz(x)

    assert test_i32(0) == 32
    assert test_i32(1) == 31
    assert test_i32(2) == 30
    assert test_i32(3) == 30
    assert test_i32(4) == 29
    assert test_i32(5) == 29
    assert test_i32(1023) == 22
    assert test_i32(1024) == 21
    # Sign-bit / all-bits-set cases. These exercise the unsigned-MSB semantics (clz must count over the bit pattern,
    # so clz(-1) == 0). Before the FindUMsb fix, the SPIR-V path returned 32 for clz(-1).
    assert test_i32(-1) == 0
    assert test_i32(-2) == 0
    assert test_i32(0x7FFFFFFF) == 1

    # u32 inputs lower to the same intrinsic on every backend (LLVM IR is signless for integers; SPIR-V FindUMsb is
    # unsigned by definition). Pre-generalisation, CUDA / AMDGPU rejected u32 with QD_NOT_IMPLEMENTED and required a
    # bit_cast through qd.i32 as a workaround.
    assert test_u32(0) == 32
    assert test_u32(1) == 31
    assert test_u32(0x7FFFFFFF) == 1
    assert test_u32(0x80000000) == 0
    assert test_u32(0xFFFFFFFF) == 0


# clz on 64-bit ints — runs on every supported backend. CPU / CUDA use native 64-bit leading-zero ops (`__nv_clzll`);
# AMDGPU lowers via llvm.ctlz; SPIR-V (Vulkan / Metal) synthesises the 64-bit case from a hi/lo FindUMsb decomposition.
# u64 routes through the same paths as i64 since the operation is on the bit pattern.
@test_utils.test()
def test_clz_i64():
    @qd.kernel
    def test_i64(x: qd.int64) -> qd.int32:
        return qd.math.clz(x)

    @qd.kernel
    def test_u64(x: qd.uint64) -> qd.int32:
        return qd.math.clz(x)

    assert test_i64(0) == 64
    assert test_i64(1) == 63
    assert test_i64(2) == 62
    assert test_i64(1 << 31) == 32
    assert test_i64(1 << 32) == 31
    assert test_i64(1 << 62) == 1
    # Top bit set on i64 (sign bit) -> 0xFFFFFFFFFFFFFFFF interpreted as -1.
    assert test_i64(-1) == 0
    # Spans both halves: bit 32 set with low half also non-zero.
    assert test_i64((1 << 32) | 0xFF) == 31

    # u64 mirrors i64 with sign-bit / all-bits-set cases that are awkward to express as signed literals.
    assert test_u64(0) == 64
    assert test_u64(1) == 63
    assert test_u64(1 << 32) == 31
    assert test_u64(1 << 63) == 0
    assert test_u64(0xFFFFFFFFFFFFFFFF) == 0
    assert test_u64(0x7FFFFFFFFFFFFFFF) == 1


# Regression sentinel for the i32 return-type normalisation: popcnt / clz must return i32 from the type-checking pass
# onward, not just at codegen time. If the inferred ret_type were the operand's type (i64), promotion of
# `op(x: i64) + i64(1)` to i64 would skip the i32 -> i64 cast, and CUDA / AMDGPU codegen — which truncates the
# libdevice / llvm.ctpop result to i32 — would emit `Add(i32, i64)` and trip an LLVM "operand type mismatch" assertion.
# Direct-return tests above hide this because they don't compose the result with any other operand.
@test_utils.test()
def test_bit_ops_compound_i64():
    @qd.kernel
    def popcnt_plus_one(x: qd.int64) -> qd.int64:
        return qd.math.popcnt(x) + qd.i64(1)

    @qd.kernel
    def clz_plus_one(x: qd.int64) -> qd.int64:
        return qd.math.clz(x) + qd.i64(1)

    assert popcnt_plus_one(0) == 1  # popcnt(0) = 0, +1 = 1
    assert popcnt_plus_one(-1) == 65  # popcnt(0xFFF..F) = 64, +1 = 65 (all-bits-set as signed i64)
    assert clz_plus_one(0) == 65  # clz(0) = 64, +1 = 65
    assert clz_plus_one(1) == 64  # clz(1) = 63, +1 = 64


@test_utils.test()
def test_sign():
    @qd.kernel
    def foo(val: qd.f32) -> qd.f32:
        return qd.math.sign(val)

    assert foo(0.5) == 1.0
    assert foo(-0.5) == -1.0
    assert foo(0.0) == 0.0
