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


# Mirrors test_bit_not_invalid / test_logic_not_invalid: ffs is integer-only at the frontend. Real-typed operand must
# raise QuadrantsTypeError with the same diagnostic, before any codegen is reached - this is what gates ffs on a
# usefully restricted operand domain on every backend.
@test_utils.test(print_full_traceback=False)
def test_ffs_invalid():
    @qd.kernel
    def test(x: qd.f32) -> qd.i32:
        return qd.math.ffs(x)

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


# clz on 64-bit ints - runs on every supported backend. CPU / CUDA use native 64-bit leading-zero ops (`__nv_clzll`);
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


# ffs(x) returns the 1-indexed position of the lowest set bit, with ffs(0) == 0 (CUDA __ffs convention).
# CPU / CUDA / AMDGPU lower to the LLVM ctz intrinsic family; SPIR-V (Vulkan / Metal) lowers to FindILsb.
# Signed and unsigned inputs share an intrinsic since the operation is on the bit pattern.
@test_utils.test()
def test_ffs():
    @qd.kernel
    def test_i32(x: qd.int32) -> qd.int32:
        return qd.math.ffs(x)

    @qd.kernel
    def test_u32(x: qd.uint32) -> qd.int32:
        return qd.math.ffs(x)

    assert test_i32(0) == 0
    assert test_i32(1) == 1
    assert test_i32(2) == 2
    assert test_i32(3) == 1
    assert test_i32(4) == 3
    assert test_i32(8) == 4
    assert test_i32(0x7FFFFFFF) == 1
    # MSB-only / sign-bit / all-bits-set: lowest set bit is the sign bit, so ffs == 32.
    assert test_i32(-(1 << 31)) == 32
    # All bits set -> -1 in two's complement; lowest set bit is bit 0.
    assert test_i32(-1) == 1
    # Non-trivial bit pattern: lowest set bit is at position 4 (1-indexed).
    assert test_i32(0xF0) == 5
    assert test_i32(0xF00) == 9

    # u32 routes through the same intrinsic; signedness is irrelevant since ffs counts trailing zeros of the bit
    # pattern.
    assert test_u32(0) == 0
    assert test_u32(1) == 1
    assert test_u32(0x80000000) == 32
    assert test_u32(0xFFFFFFFF) == 1
    assert test_u32(0xFFFFFFFE) == 2


# ffs on 64-bit ints. CPU / CUDA / AMDGPU lower to llvm.cttz / __nv_ffsll natively; SPIR-V (Vulkan / Metal)
# synthesises from a hi/lo FindILsb decomposition: low half first since "first" = lowest-indexed bit.
@test_utils.test()
def test_ffs_i64():
    @qd.kernel
    def test_i64(x: qd.int64) -> qd.int32:
        return qd.math.ffs(x)

    @qd.kernel
    def test_u64(x: qd.uint64) -> qd.int32:
        return qd.math.ffs(x)

    assert test_i64(0) == 0
    assert test_i64(1) == 1
    assert test_i64(2) == 2
    # First bit at position 32 (in the high half) - exercises the SPIR-V hi/lo decomposition split.
    assert test_i64(1 << 31) == 32
    assert test_i64(1 << 32) == 33
    assert test_i64(1 << 62) == 63
    # All bits set -> -1; lowest set bit is bit 0.
    assert test_i64(-1) == 1
    # Lo half zero, hi half non-zero: result must come from the hi-half arm of the SPV select.
    assert test_i64(1 << 40) == 41
    # Both halves non-zero: low-half arm wins.
    assert test_i64((1 << 40) | (1 << 8)) == 9

    # u64 sign-bit / all-bits-set cases that are awkward to express as signed literals.
    assert test_u64(0) == 0
    assert test_u64(1) == 1
    assert test_u64(1 << 63) == 64
    assert test_u64(0xFFFFFFFFFFFFFFFF) == 1
    assert test_u64(0x8000000000000000) == 64
    # Bit 33 (1-indexed): exercises the low-half-zero, hi-half-non-zero path.
    assert test_u64(1 << 32) == 33


# Regression sentinel for the i32 return-type normalisation: every bit-op (popcnt / clz / ffs) must return i32 from the
# type-checking pass onward, not just at codegen time. If the inferred ret_type were the operand's type (i64),
# promotion of `op(x: i64) + i64(1)` to i64 would skip the i32 → i64 cast, and CUDA / AMDGPU codegen - which truncates
# the libdevice / llvm.ctpop result to i32 - would emit an `Add(i32, i64)` and trip an LLVM "operand type mismatch"
# assertion. Direct-return tests above hide this because they don't compose the result with any other operand.
@test_utils.test()
def test_bit_ops_compound_i64():
    @qd.kernel
    def popcnt_plus_one(x: qd.int64) -> qd.int64:
        return qd.math.popcnt(x) + qd.i64(1)

    @qd.kernel
    def clz_plus_one(x: qd.int64) -> qd.int64:
        return qd.math.clz(x) + qd.i64(1)

    @qd.kernel
    def ffs_plus_one(x: qd.int64) -> qd.int64:
        return qd.math.ffs(x) + qd.i64(1)

    assert popcnt_plus_one(0) == 1  # popcnt(0) = 0, +1 = 1
    assert popcnt_plus_one(-1) == 65  # popcnt(0xFFF..F) = 64, +1 = 65 (all-bits-set as signed i64)
    assert clz_plus_one(0) == 65  # clz(0) = 64, +1 = 65
    assert clz_plus_one(1) == 64  # clz(1) = 63, +1 = 64
    assert ffs_plus_one(0) == 1  # ffs(0) = 0, +1 = 1
    assert ffs_plus_one(1 << 32) == 34  # ffs((1<<32)) = 33, +1 = 34


def _np_fns(mask: int, base: int, offset: int) -> int:
    """Pure-Python reference for qd.math.fns / CUDA __nv_fns (32-bit).

    Mirrors PTX `fns` semantics so the kernel result can be diffed against this trivially.
    """
    NOT_FOUND = 0xFFFFFFFF
    if offset == 0:
        if 0 <= base < 32 and (mask >> base) & 1:
            return base
        return NOT_FOUND
    if offset > 0:
        n = offset
        for pos in range(32):
            if pos < base:
                continue
            if (mask >> pos) & 1:
                n -= 1
                if n == 0:
                    return pos
        return NOT_FOUND
    n = -offset
    for pos in range(31, -1, -1):
        if pos > base:
            continue
        if (mask >> pos) & 1:
            n -= 1
            if n == 0:
                return pos
    return NOT_FOUND


# fns(mask, base, offset) finds the |offset|-th set bit in `mask` starting from `base`, scanning upward (offset > 0),
# downward (offset < 0), or returning `base` if exactly that bit is set (offset == 0). On CUDA this lowers to libdevice
# __nv_fns (single-instruction PTX `fns`); on every other backend it emits a portable @qd.func that loops over the 32
# bit positions. Both paths must agree with the pure-Python reference _np_fns.
@test_utils.test()
def test_fns():
    @qd.kernel
    def fns_kernel(mask: qd.uint32, base: qd.uint32, offset: qd.int32) -> qd.uint32:
        return qd.math.fns(mask, base, offset)

    # Exhaustive sweep over a handful of representative masks. Catches both the upward and downward search arms and
    # the offset == 0 special case in a single loop.
    test_masks = [
        0x00000000,  # empty mask -> always NOT_FOUND
        0xFFFFFFFF,  # all set -> trivial
        0x00000001,  # single bit at position 0
        0x80000000,  # single bit at position 31
        0x0000000F,  # bits 0..3
        0xF0000000,  # bits 28..31
        0xAAAAAAAA,  # alternating bits
        0x12345678,  # arbitrary
    ]
    for mask in test_masks:
        for base in [0, 1, 4, 15, 16, 31]:
            for offset in [-3, -2, -1, 0, 1, 2, 3, 5]:
                expected = _np_fns(mask, base, offset)
                got = fns_kernel(mask, base, offset)
                assert (
                    got == expected
                ), f"fns(mask=0x{mask:08X}, base={base}, offset={offset}): got 0x{got:08X}, expected 0x{expected:08X}"

    # Spot-check a few canonical examples directly so failures point at obvious cases first.
    NOT_FOUND = 0xFFFFFFFF
    # Search upward, multiple bits.
    assert fns_kernel(0xF, 0, 1) == 0
    assert fns_kernel(0xF, 0, 2) == 1
    assert fns_kernel(0xF, 0, 4) == 3
    assert fns_kernel(0xF, 0, 5) == NOT_FOUND
    # base inside the set bits: skip bits below base.
    assert fns_kernel(0xF, 2, 1) == 2
    assert fns_kernel(0xF, 2, 2) == 3
    # Search downward.
    assert fns_kernel(0xF, 5, -1) == 3
    assert fns_kernel(0xF, 5, -4) == 0
    assert fns_kernel(0xF, 5, -5) == NOT_FOUND
    # offset == 0: bit-at-base test.
    assert fns_kernel(0xE, 1, 0) == 1
    assert fns_kernel(0xE, 0, 0) == NOT_FOUND

    # Maximum-magnitude offsets. PTX `fns` admits |offset| up to 32 (the bit width of the mask), which the exhaustive
    # sweep above does not cover. These cases force the search to walk the entire mask before finding (or failing to
    # find) the requested bit, and would catch an off-by-one in the loop bound or the early-exit guard on either
    # implementation.
    assert fns_kernel(0xFFFFFFFF, 0, 32) == 31  # 32nd set bit walking up from 0 in all-set mask
    assert fns_kernel(0xFFFFFFFF, 31, -32) == 0  # 32nd-from-top walking down from 31 in all-set mask
    assert fns_kernel(0xFFFFFFFF, 0, 33) == NOT_FOUND  # only 32 set bits exist
    assert fns_kernel(0xFFFFFFFF, 31, -33) == NOT_FOUND
    # Single-bit masks with large offsets must still return NOT_FOUND rather than walk past the end.
    assert fns_kernel(0x1, 0, 32) == NOT_FOUND
    assert fns_kernel(0x80000000, 31, -32) == NOT_FOUND


@test_utils.test()
def test_sign():
    @qd.kernel
    def foo(val: qd.f32) -> qd.f32:
        return qd.math.sign(val)

    assert foo(0.5) == 1.0
    assert foo(-0.5) == -1.0
    assert foo(0.0) == 0.0
