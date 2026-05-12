import math

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

n = 128


def run_atomic_add_global_case(vartype, step, valproc=lambda x: x):
    x = qd.field(vartype)
    y = qd.field(vartype)
    c = qd.field(vartype)

    qd.root.dense(qd.i, n).place(x, y)
    qd.root.place(c)

    # Make Quadrants correctly infer the type
    # TODO: Quadrants seems to treat numpy.int32 as a float type, fix that.
    init_ck = 0 if vartype == qd.i32 else 0.0

    @qd.kernel
    def func():
        ck = init_ck
        for i in range(n):
            x[i] = qd.atomic_add(c[None], step)
            y[i] = qd.atomic_add(ck, step)

    func()

    assert valproc(c[None]) == n * step
    x_actual = sorted(x.to_numpy())
    y_actual = sorted(y.to_numpy())
    expect = [i * step for i in range(n)]
    for xa, ya, e in zip(x_actual, y_actual, expect):
        print(xa, ya, e)
        assert valproc(xa) == e
        assert valproc(ya) == e


@test_utils.test()
def test_atomic_add_global_i32():
    run_atomic_add_global_case(qd.i32, 42)


@test_utils.test()
def test_atomic_add_global_f32():
    run_atomic_add_global_case(qd.f32, 4.2, valproc=lambda x: test_utils.approx(x, rel=1e-5))


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_atomic_min_max_uint():
    x = qd.field(qd.u64, shape=100)

    @qd.kernel
    def test0():
        for I in x:
            x[I] = 0
        x[1] = qd.cast(1, qd.u64) << 63
        for I in x:
            qd.atomic_max(x[0], x[I])

    test0()
    assert x[0] == 9223372036854775808

    @qd.kernel
    def test1():
        for I in x:
            x[I] = qd.cast(1, qd.u64) << 63
        x[1] = 100
        for I in x:
            qd.atomic_min(x[0], x[I])

    test1()
    assert x[0] == 100


@test_utils.test()
def test_atomic_add_expr_evaled():
    c = qd.field(qd.i32)
    step = 42

    qd.root.place(c)

    @qd.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_add(c[None], step)

    func()

    assert c[None] == n * step


@test_utils.test()
def test_atomic_add_demoted():
    # Ensure demoted atomics do not crash the program.
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)
    step = 42

    qd.root.dense(qd.i, n).place(x, y)

    @qd.kernel
    def func():
        for i in range(n):
            s = i
            # Both adds should get demoted.
            x[i] = qd.atomic_add(s, step)
            y[i] = qd.atomic_add(s, step)

    func()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + step


@test_utils.test()
def test_atomic_add_with_local_store_simplify1():
    # Test for the following LocalStoreStmt simplification case:
    #
    # local store [$a <- ...]
    # atomic add ($a, ...)
    # local store [$a <- ...]
    #
    # Specifically, the second store should not suppress the first one, because
    # atomic_add can return value.
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)
    step = 42

    qd.root.dense(qd.i, n).place(x, y)

    @qd.kernel
    def func():
        for i in range(n):
            # do a local store
            j = i
            x[i] = qd.atomic_add(j, step)
            # do another local store, make sure the previous one is not optimized out
            j = x[i]
            y[i] = j

    func()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i


@test_utils.test()
def test_atomic_add_with_local_store_simplify2():
    # Test for the following LocalStoreStmt simplification case:
    #
    # local store [$a <- ...]
    # atomic add ($a, ...)
    #
    # Specifically, the local store should not be removed, because
    # atomic_add can return its value.
    x = qd.field(qd.i32)
    step = 42

    qd.root.dense(qd.i, n).place(x)

    @qd.kernel
    def func():
        for i in range(n):
            j = i
            x[i] = qd.atomic_add(j, step)

    func()

    for i in range(n):
        assert x[i] == i


@test_utils.test()
def test_atomic_add_with_if_simplify():
    # Make sure IfStmt simplification doesn't move stmts depending on the result
    # of atomic_add()
    x = qd.field(qd.i32)
    step = 42

    qd.root.dense(qd.i, n).place(x)

    boundary = n / 2

    @qd.kernel
    def func():
        for i in range(n):
            if i > boundary:
                # A sequence of commands designed such that atomic_add() is the only
                # thing to decide whether the if branch can be simplified.
                s = i
                j = qd.atomic_add(s, s)
                k = j + s
                x[i] = k
            else:
                # If we look at the IR, this branch should be simplified, since nobody
                # is using atomic_add's result.
                qd.atomic_add(x[i], i)
                x[i] += step

    func()

    for i in range(n):
        expect = i * 3 if i > boundary else (i + step)
        assert x[i] == expect


@test_utils.test()
def test_local_atomic_with_if():
    ret = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def test():
        if True:
            x = 0
            x += 1
            ret[None] = x

    test()
    assert ret[None] == 1


@test_utils.test()
def test_atomic_sub_with_type_promotion():
    if qd.lang.impl.current_cfg().arch in (qd.metal, qd.vulkan):
        pytest.xfail("BUG: SPIR-V codegen does not support unsigned integer negation (OpSNegate).")

    # Test Case 1
    @qd.kernel
    def test_u16_sub_u8() -> qd.uint16:
        x: qd.uint16 = 1000
        y: qd.uint8 = 255

        qd.atomic_sub(x, y)
        return x

    res = test_u16_sub_u8()
    assert res == 745

    # Test Case 2
    @qd.kernel
    def test_u8_sub_u16() -> qd.uint8:
        x: qd.uint8 = 255
        y: qd.uint16 = 100

        qd.atomic_sub(x, y)
        return x

    res = test_u8_sub_u16()
    assert res == 155

    # Test Case 3
    A = qd.field(qd.uint8, shape=())
    B = qd.field(qd.uint16, shape=())

    @qd.kernel
    def test_with_field():
        v: qd.uint16 = 1000
        v -= A[None]
        B[None] = v

    A[None] = 255
    test_with_field()
    assert B[None] == 745


@test_utils.test()
def test_atomic_sub_expr_evaled():
    c = qd.field(qd.i32)
    step = 42

    qd.root.place(c)

    @qd.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_sub(c[None], step)

    func()

    assert c[None] == -n * step


@test_utils.test()
def test_atomic_mul_expr_evaled():
    c = qd.field(qd.i32)
    base = 2

    qd.root.place(c)

    @qd.kernel
    def func():
        c[None] = 1
        for i in range(16):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_mul(c[None], base)

    func()

    assert c[None] == base**16


@test_utils.test()
def test_atomic_max_expr_evaled():
    c = qd.field(qd.i32)
    step = 42

    qd.root.place(c)

    @qd.kernel
    def func():
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_max(c[None], i * step)

    func()

    assert c[None] == (n - 1) * step


@test_utils.test()
def test_atomic_min_expr_evaled():
    c = qd.field(qd.i32)
    step = 42

    qd.root.place(c)

    @qd.kernel
    def func():
        c[None] = 1000
        for i in range(n):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_min(c[None], i * step)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_and_expr_evaled():
    c = qd.field(qd.i32)
    step = 42

    qd.root.place(c)

    max_int = 2147483647

    @qd.kernel
    def func():
        c[None] = 1023
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_and(c[None], max_int - 2**i)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_or_expr_evaled():
    c = qd.field(qd.i32)
    step = 42

    qd.root.place(c)

    @qd.kernel
    def func():
        c[None] = 0
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_or(c[None], 2**i)

    func()

    assert c[None] == 1023


@test_utils.test()
def test_atomic_xor_expr_evaled():
    c = qd.field(qd.i32)
    step = 42

    qd.root.place(c)

    @qd.kernel
    def func():
        c[None] = 1023
        for i in range(10):
            # this is an expr with side effect, make sure it's not optimized out.
            qd.atomic_xor(c[None], 2**i)

    func()

    assert c[None] == 0


@test_utils.test()
def test_atomic_min_rvalue_as_frist_op():
    @qd.kernel
    def func():
        y = qd.Vector([1, 2, 3])
        z = qd.atomic_min([3, 2, 1], y)

    with pytest.raises(qd.QuadrantsSyntaxError) as e:
        func()

    assert "atomic_min" in str(e.value)
    assert "cannot use a non-writable target as the first operand of" in str(e.value)


@test_utils.test()
def test_atomic_max_f32():
    @qd.kernel
    def max_kernel() -> qd.f32:
        x = -1000.0
        for i in range(1, 20):
            qd.atomic_max(x, -qd.f32(i))

        return x

    assert max_kernel() == -1.0


@pytest.mark.parametrize("op", ["add", "sub", "min", "max"])
@pytest.mark.parametrize("dtype", [qd.f16, qd.f32, qd.f64])
@test_utils.test()
def test_atomic_float_ops(op, dtype):
    if qd.cfg.arch in (qd.vulkan, qd.metal):
        caps = qd.lang.impl.get_runtime().prog.get_device_caps()
        # f16 CAS requires 16-bit integer atomics, unsupported on MoltenVK/Metal
        if dtype == qd.f16 and not caps.get(qd._lib.core.DeviceCapability.spirv_has_atomic_float16):
            pytest.skip("Device does not support f16 atomics")
        if dtype == qd.f64 and not caps.get(qd._lib.core.DeviceCapability.spirv_has_float64):
            pytest.skip("Device does not support f64")
    block_dim = 32
    N = block_dim * 4
    SCALE = 0.1523
    atomic_op = getattr(qd, f"atomic_{op}")

    @qd.kernel
    def kern(out: qd.types.ndarray()):
        # Use multiple threads to test concurrent atomicity
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            val = qd.cast(tid * SCALE, dtype)
            atomic_op(out[0], val)

    arr = qd.ndarray(dtype, (1,))
    arr[0] = 0.0
    kern(arr)
    # 4 blocks each contributing SCALE * (0 + 1 + ... + 31)
    nblocks = N // block_dim
    per_block_sum = SCALE * block_dim * (block_dim - 1) / 2.0
    expected = {
        "add": per_block_sum * nblocks,
        "sub": -per_block_sum * nblocks,
        "min": 0.0,
        "max": (block_dim - 1) * SCALE,
    }
    rtol = {qd.f16: 1e-3, qd.f64: 1e-10}.get(dtype, 1e-6)
    assert arr[0] == test_utils.approx(expected[op], rel=rtol)


@test_utils.test()
def test_atomic_mul_f32():
    @qd.kernel
    def mul_kernel() -> qd.f32:
        x = 1.0
        for i in range(1, 8):
            qd.atomic_mul(x, qd.f32(i))

        return x

    assert mul_kernel() == 5040.0


# Pins the doc claim that atomic_mul works (via CAS loop) under multi-thread contention on every GPU backend, for
# both ints and floats including f64. Existing coverage is single-thread only (test_atomic_mul_f32,
# test_atomic_mul_expr_evaled). Values chosen so the product is representable exactly in i32 / f32 / f64.
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_atomic_mul_contention(dtype):
    test_utils.skip_if_f64_unsupported(dtype)
    block_dim = 4
    nblocks = 4
    N = block_dim * nblocks

    arr = qd.ndarray(dtype, (1,))
    arr[0] = 1

    @qd.kernel
    def kern(out: qd.types.ndarray()):
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            qd.atomic_mul(out[0], qd.cast(tid + 1, dtype))

    kern(arr)
    per_block = 1
    for v in range(1, block_dim + 1):
        per_block *= v
    expected = per_block**nblocks
    if dtype == qd.i32:
        assert int(arr[0]) == expected
    else:
        rtol = 1e-6 if dtype == qd.f32 else 1e-12
        assert arr[0] == test_utils.approx(float(expected), rel=rtol)


# Pins the doc claim that floating-point atomic_min / atomic_max use minNum / maxNum-style NaN semantics: when
# exactly one operand is NaN, the non-NaN value is written back. Asserted on every GPU backend and both arg orders.
# CPU is intentionally out of scope here -- the doc explicitly warns that the CPU CAS path uses naive < / >
# comparisons with order-dependent results.
@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_atomic_min_max_nan_seed(op, dtype):
    test_utils.skip_if_f64_unsupported(dtype)
    f = qd.field(dtype, shape=())
    f[None] = float("nan")
    atomic_op = getattr(qd, f"atomic_{op}")

    @qd.kernel
    def kern():
        atomic_op(f[None], qd.cast(7.0, dtype))

    kern()
    assert not math.isnan(float(f[None]))
    assert float(f[None]) == 7.0


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_atomic_min_max_nan_arg(op, dtype):
    test_utils.skip_if_f64_unsupported(dtype)
    f = qd.field(dtype, shape=())
    f[None] = 7.0
    atomic_op = getattr(qd, f"atomic_{op}")

    @qd.kernel
    def kern():
        atomic_op(f[None], qd.cast(qd.math.nan, dtype))

    kern()
    assert not math.isnan(float(f[None]))
    assert float(f[None]) == 7.0


# Pins the doc claim that vector/matrix arguments to atomic ops fan out to one scalar atomic per component (no
# all-or-nothing guarantee across components, but every component must be summed exactly).
@test_utils.test(arch=qd.gpu)
def test_atomic_add_vector_field_fanout():
    N = 256
    f = qd.Vector.field(3, qd.f32, shape=())
    f[None] = qd.Vector([0.0, 0.0, 0.0])

    @qd.kernel
    def kern():
        for _ in range(N):
            qd.atomic_add(f[None], qd.Vector([1.0, 2.0, 3.0]))

    kern()
    assert f[None][0] == test_utils.approx(N * 1.0, rel=1e-5)
    assert f[None][1] == test_utils.approx(N * 2.0, rel=1e-5)
    assert f[None][2] == test_utils.approx(N * 3.0, rel=1e-5)


def _skip_if_no_int64_atomic_rmw(dtype):
    """Skip when the device cannot do general-purpose 64-bit integer atomic RMW.

    Metal sets ``spirv_has_atomic_int64=1`` on Apple7+ / Mac2 in ``metal_device.mm`` (the gate is misnamed
    ``feature_floating_point_atomics``), but MSL only exposes 64-bit atomics as ``atomic_fetch_min/max`` on
    ``uint64`` starting at Apple9 (M3+, A17+); ``atomic_add`` / ``and`` / ``or`` / ``xor`` are not available at
    all. The pipeline create then fails with RhiResult=-1 ("SPIR-V shader was rejected by the backend"). Tightening
    the Metal cap itself is intentionally out of scope here -- the cap is consumed by adstack fallbacks in
    ``runtime/gfx`` and lowering it needs a separate audit -- so we just skip the affected dtypes on Metal instead.
    """
    if dtype not in (qd.i64, qd.u64):
        return
    if qd.cfg.arch == qd.metal:
        pytest.skip("Metal lacks general-purpose 64-bit integer atomic RMW (MSL atomic_long fetch_add/and/or/xor)")
    if qd.cfg.arch == qd.vulkan:
        caps = qd.lang.impl.get_runtime().prog.get_device_caps()
        if not caps.get(qd._lib.core.DeviceCapability.spirv_has_atomic_int64):
            pytest.skip("Vulkan device does not advertise spirv_has_atomic_int64")


# Pins the doc-table claim that atomic_add is "yes" on every integer dtype (i32 / u32 / i64 / u64) on every GPU
# backend. Existing coverage only exercises i32 (run_atomic_add_global_case) and u64 min/max
# (test_atomic_min_max_uint).
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.i64, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_atomic_add_int_contention(dtype):
    _skip_if_no_int64_atomic_rmw(dtype)
    N = 256
    f = qd.field(dtype, shape=())
    f[None] = 0

    @qd.kernel
    def kern():
        for _ in range(N):
            qd.atomic_add(f[None], qd.cast(1, dtype))

    kern()
    assert int(f[None]) == N


# Pins the doc-table claim that atomic_and / atomic_or / atomic_xor are "yes" on every integer dtype on every GPU
# backend. Existing coverage only exercises i32 (test_atomic_{and,or,xor}_expr_evaled).
@pytest.mark.parametrize(
    "op,seed,arg,expected",
    [
        ("and", 0xFF0F, 0x0FF0, 0x0F00),
        ("or", 0x00F0, 0x0F00, 0x0FF0),
        ("xor", 0xFF0F, 0x0FF0, 0xF0FF),
    ],
)
@pytest.mark.parametrize("dtype", [qd.u32, qd.i64, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_atomic_bitwise_int_widths(op, seed, arg, expected, dtype):
    _skip_if_no_int64_atomic_rmw(dtype)
    f = qd.field(dtype, shape=())
    f[None] = seed
    atomic_op = getattr(qd, f"atomic_{op}")

    @qd.kernel
    def kern():
        atomic_op(f[None], qd.cast(arg, dtype))

    kern()
    assert int(f[None]) == expected


# Pins the doc claim that bitwise atomics on float dtypes raise a type error at trace time (atomics page: "Integer
# dtypes only -- passing f32 / f64 raises a type error at trace time"). Enforced by the is_integral check in
# AtomicOpExpression::type_check (quadrants/ir/frontend_ir.cpp) for bit_and / bit_or / bit_xor.
@pytest.mark.parametrize("op", ["and", "or", "xor"])
@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_atomic_bitwise_on_float_field_raises(op, dtype):
    test_utils.skip_if_f64_unsupported(dtype)
    f = qd.field(dtype, shape=())
    f[None] = 0
    atomic_op = getattr(qd, f"atomic_{op}")

    @qd.kernel
    def kern():
        atomic_op(f[None], qd.cast(1, dtype))

    with pytest.raises(qd.lang.exception.QuadrantsCompilationError):
        kern()


@test_utils.test(arch=qd.gpu)
def test_atomic_add_matrix_field_fanout():
    N = 256
    m = qd.Matrix.field(2, 3, qd.f32, shape=())
    m[None] = qd.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    contrib = qd.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    @qd.kernel
    def kern():
        for _ in range(N):
            qd.atomic_add(m[None], contrib)

    kern()
    for i in range(2):
        for j in range(3):
            expected = N * (i * 3 + j + 1)
            assert m[None][i, j] == test_utils.approx(expected, rel=1e-5)


# Pins the documented semantics of qd.atomic_exchange: unconditionally write `val` into `dest` and return the
# old value of `dest`. Single-thread sanity, all primitive dtypes that the codegen path supports today (i32,
# u32, i64, u64, f32, f64). f16 xchg falls through the f16 CAS path in codegen_llvm.cpp's real_type_atomic
# and is deferred (see TODO in real_type_atomic / shared_float_atomic).
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.i64, qd.u64, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_atomic_exchange_returns_old_value(dtype):
    test_utils.skip_if_f64_unsupported(dtype)
    _skip_if_no_int64_atomic_rmw(dtype)
    f = qd.field(dtype, shape=())
    old_out = qd.field(dtype, shape=())
    f[None] = 7

    @qd.kernel
    def kern():
        old_out[None] = qd.atomic_exchange(f[None], qd.cast(42, dtype))

    kern()
    if dtype in (qd.f32, qd.f64, qd.f16):
        assert float(f[None]) == 42.0
        assert float(old_out[None]) == 7.0
    else:
        assert int(f[None]) == 42
        assert int(old_out[None]) == 7


# Stress test: N threads each xchg a unique nonzero value into a single shared slot, capturing the returned
# old value into a per-thread record. Atomicity guarantees that the multiset {final slot value} U {captured
# old values} == {initial value} U {all contributed values}, i.e. no value is lost or duplicated. This is the
# canonical "no values lost" check for a swap primitive and is what distinguishes a correct atomic exchange
# from a racy load+store pair.
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.i64, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_atomic_exchange_swap_under_contention(dtype):
    _skip_if_no_int64_atomic_rmw(dtype)
    N = 256
    INIT = 1_000_000
    slot = qd.field(dtype, shape=())
    olds = qd.field(dtype, shape=(N,))
    slot[None] = INIT

    @qd.kernel
    def kern():
        for i in range(N):
            olds[i] = qd.atomic_exchange(slot[None], qd.cast(i + 1, dtype))

    kern()

    contributed = set(range(1, N + 1)) | {INIT}
    seen = {int(olds[i]) for i in range(N)} | {int(slot[None])}
    assert (
        seen == contributed
    ), f"atomic_exchange lost or duplicated values: missing={contributed - seen}, extra={seen - contributed}"


# Pins that vector-typed atomic_exchange fans out to one scalar OpAtomicExchange per component, mirroring the
# existing fan-out semantics for atomic_add (test_atomic_add_vector_field_fanout). After N exchanges each writing
# the same vector, the slot must equal that vector exactly (last writer wins per component).
@test_utils.test(arch=qd.gpu)
def test_atomic_exchange_vector_field_fanout():
    N = 64
    f = qd.Vector.field(3, qd.f32, shape=())
    f[None] = qd.Vector([0.0, 0.0, 0.0])

    @qd.kernel
    def kern():
        for _ in range(N):
            qd.atomic_exchange(f[None], qd.Vector([1.5, 2.5, 3.5]))

    kern()
    assert f[None][0] == test_utils.approx(1.5, rel=1e-5)
    assert f[None][1] == test_utils.approx(2.5, rel=1e-5)
    assert f[None][2] == test_utils.approx(3.5, rel=1e-5)


# Pins that matrix-typed atomic_exchange fans out to one scalar OpAtomicExchange per component, completing the
# vector + matrix coverage promised in atomics.md ("Vector / matrix arguments fan out per component"). Mirrors
# test_atomic_add_matrix_field_fanout. After N exchanges each writing the same matrix, the slot must equal that
# matrix exactly (last writer wins per element, no all-or-nothing across the 2x3 components).
@test_utils.test(arch=qd.gpu)
def test_atomic_exchange_matrix_field_fanout():
    N = 64
    m = qd.Matrix.field(2, 3, qd.f32, shape=())
    m[None] = qd.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    contrib = qd.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    @qd.kernel
    def kern():
        for _ in range(N):
            qd.atomic_exchange(m[None], contrib)

    kern()
    for i in range(2):
        for j in range(3):
            expected = i * 3 + j + 1
            assert m[None][i, j] == test_utils.approx(expected, rel=1e-5)


# Pins the new xchg branch in demote_atomics.cpp's DemoteAtomics::visit(AtomicOpStmt*). Unlike every other atomic
# op (which demotes to load + binop + store), xchg demotes to a bare load + store(val) because the new value is
# independent of the old. When the destination is a thread-local (a Python variable inside a kernel, not a field),
# the atomic has no contention and the demote pass kicks in. Mirrors test_atomic_add_demoted exactly.
@test_utils.test()
def test_atomic_exchange_demoted():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)
    new_val = 42
    qd.root.dense(qd.i, n).place(x, y)

    @qd.kernel
    def func():
        for i in range(n):
            s = i
            # Both exchanges should get demoted (s is thread-local).
            x[i] = qd.atomic_exchange(s, new_val)
            y[i] = qd.atomic_exchange(s, new_val + 1)

    func()

    for i in range(n):
        # First xchg returns the initial value of s (= i); after it, s == new_val.
        # Second xchg returns new_val (proving the first xchg actually wrote new_val into s).
        assert x[i] == i
        assert y[i] == new_val


# Pins the documented semantics of qd.atomic_cas: returns the prior value of `dest`, swaps in `desired` only when
# the prior value equals `expected`. Single-thread sanity covering both the success path (prior == expected) and
# the failure path (prior != expected) for every integer dtype the codegen path supports today (i32 / u32 /
# i64 / u64). f32 / f64 CAS is currently rejected at trace time -- a separate negative test pins that.
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.i64, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_returns_old_value(dtype):
    _skip_if_no_int64_atomic_rmw(dtype)
    f = qd.field(dtype, shape=())
    out_succ = qd.field(dtype, shape=())
    out_fail = qd.field(dtype, shape=())
    f[None] = 7

    @qd.kernel
    def kern_success():
        # expected == current -> swap fires, return prior (= 7).
        out_succ[None] = qd.atomic_cas(f[None], qd.cast(7, dtype), qd.cast(42, dtype))

    kern_success()
    assert int(out_succ[None]) == 7
    assert int(f[None]) == 42

    @qd.kernel
    def kern_failure():
        # expected != current (current is now 42) -> swap is a no-op, return prior (= 42).
        out_fail[None] = qd.atomic_cas(f[None], qd.cast(99, dtype), qd.cast(123, dtype))

    kern_failure()
    assert int(out_fail[None]) == 42
    assert int(f[None]) == 42


# Stress test for atomic CAS contention: N threads each attempt to flip a slot from INIT to their own unique id.
# Atomicity requires that exactly ONE thread observes prior == INIT (the winner) and all the others observe
# prior == winner_id (their CAS failed because the winner already wrote). Pins both: (a) exactly one thread
# wins, and (b) the slot ends up with the winner's id (no torn writes).
@pytest.mark.parametrize("dtype", [qd.i32, qd.u32, qd.i64, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_contention_single_winner(dtype):
    _skip_if_no_int64_atomic_rmw(dtype)
    N = 256
    INIT = 0
    slot = qd.field(dtype, shape=())
    olds = qd.field(dtype, shape=(N,))
    slot[None] = INIT

    @qd.kernel
    def kern():
        for i in range(N):
            # Each thread tries to flip the slot from INIT to (i + 1). Only the first one to land succeeds;
            # everyone else observes the winner's (i+1) value as the prior.
            olds[i] = qd.atomic_cas(slot[None], qd.cast(INIT, dtype), qd.cast(i + 1, dtype))

    kern()

    final = int(slot[None])
    olds_seen = [int(olds[i]) for i in range(N)]
    winners = [i for i in range(N) if olds_seen[i] == INIT]
    assert len(winners) == 1, f"expected exactly one CAS winner, got {len(winners)}: indices {winners[:5]}..."
    winner_id = winners[0] + 1
    assert final == winner_id, f"slot ended with {final}, expected winner's id {winner_id}"
    for i in range(N):
        if i == winners[0]:
            continue
        # Every loser must have seen the winner's id (the only value the slot ever holds besides INIT).
        assert olds_seen[i] == winner_id, f"loser {i} observed {olds_seen[i]}, expected winner {winner_id}"


# Pins that a user-built CAS retry loop produces the same result as atomic_add. This is the canonical use case
# for exposing CAS at all -- it lets users build atomic RMW operations the framework doesn't expose natively.
# The loop is bounded (no while-True in kernels) so we run a fixed number of attempts per iteration; with N
# iterations and a small contention factor, the loop converges in O(1) attempts on average.
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_loop_increment_matches_atomic_add():
    N = 128
    counter_cas = qd.field(qd.i32, shape=())
    counter_add = qd.field(qd.i32, shape=())
    counter_cas[None] = 0
    counter_add[None] = 0

    @qd.kernel
    def kern():
        # Serialize the outer for-loop so the CAS retry budget below is bounded -- the point of this test is
        # to validate the user-facing CAS-loop pattern, not to stress-test convergence under contention.
        qd.loop_config(serialize=True)
        for _ in range(N):
            # Reference: atomic_add.
            qd.atomic_add(counter_add[None], 1)
            # CAS-loop equivalent of atomic_add: snapshot, compute new = snapshot + 1, try to swap. Under the
            # serialized outer loop there is exactly one in-flight increment, so the CAS always succeeds first
            # try; the bounded retry just demonstrates the pattern users would write.
            done = 0
            for _attempt in range(8):
                if done == 0:
                    expected = counter_cas[None]
                    old = qd.atomic_cas(counter_cas[None], expected, expected + 1)
                    if old == expected:
                        done = 1

    kern()
    # Both counters must reach N. atomic_add is the trustworthy reference; the CAS loop should match it
    # exactly when the loop converges, and lag if it doesn't (which would catch a broken CAS).
    assert int(counter_add[None]) == N
    assert (
        int(counter_cas[None]) == N
    ), f"CAS-loop increment fell behind atomic_add: cas={int(counter_cas[None])}, add={int(counter_add[None])}"


# Pins the new cas branch in demote_atomics.cpp. Demotes to load + cmp_eq + select(cmp, val, load) + store
# when the destination is a thread-local. Mirrors test_atomic_exchange_demoted; uses the success path to verify
# the swap fires, and the failure path to verify the no-op leg keeps the old value.
@test_utils.test()
def test_atomic_cas_demoted():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)
    qd.root.dense(qd.i, n).place(x, y)

    @qd.kernel
    def func():
        for i in range(n):
            s = i
            # Success path: expected == s (= i), so swap fires; old returned == i, s now == 100 + i.
            x[i] = qd.atomic_cas(s, i, 100 + i)
            # Failure path: expected (= -1) != s (now 100 + i); swap is a no-op; old returned == 100 + i;
            # s remains 100 + i. Pins that demoted CAS still returns the prior value on the no-op leg.
            y[i] = qd.atomic_cas(s, -1, 999)

    func()

    for i in range(n):
        assert x[i] == i, f"success-path CAS demoted: expected prior {i}, got {x[i]}"
        assert y[i] == 100 + i, f"failure-path CAS demoted: expected prior {100 + i}, got {y[i]}"


# Pins the doc claim that atomic_cas on float dtypes raises a type error at trace time. f32 / f64 CAS is not
# yet wired up (would need the same uint-bitcast trick xchg uses); the type_check carve-out in
# AtomicOpExpression::type_check rejects it cleanly until the lowering lands.
@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_on_float_field_raises(dtype):
    test_utils.skip_if_f64_unsupported(dtype)
    f = qd.field(dtype, shape=())
    f[None] = 0

    @qd.kernel
    def kern():
        qd.atomic_cas(f[None], qd.cast(0, dtype), qd.cast(1, dtype))

    with pytest.raises(qd.lang.exception.QuadrantsCompilationError):
        kern()


# Pins that atomic_cas on a Vector / Matrix destination is rejected at trace time. The other atomic ops fan
# out to per-component scalar AtomicOpStmts via scalarize / lower_matrix_ptr, but those passes use the 3-arg
# AtomicOpStmt constructor that drops `expected`. Until the scalarizers grow a 4-arg path, refusing tensor
# CAS up front is the correct behaviour. Codex / alanray-tech P1 from PR #690 review.
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_on_vector_field_raises():
    f = qd.Vector.field(3, qd.i32, shape=())
    f[None] = qd.Vector([0, 0, 0])

    @qd.kernel
    def kern():
        qd.atomic_cas(f[None], qd.Vector([0, 0, 0]), qd.Vector([1, 1, 1]))

    with pytest.raises(qd.lang.exception.QuadrantsCompilationError):
        kern()


# Pins that atomic_cas casts `expected` to match the destination element type, so plain Python int literals
# work as the comparator on i64 destinations the same way they do for atomic_add. Without the cast in
# AtomicOpExpression::type_check, this would either trip a backend type-mismatch assertion or silently
# compare-then-corrupt at the codegen layer. Codex / alanray-tech P1 from PR #690 review.
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_expected_int_literal_widens_to_i64():
    _skip_if_no_int64_atomic_rmw(qd.i64)
    f = qd.field(qd.i64, shape=())
    out = qd.field(qd.i64, shape=())
    f[None] = 7

    @qd.kernel
    def kern():
        # `expected` and `desired` are Python int literals (default i32). The frontend must cast them to i64
        # so the CAS operands match the i64 in-memory value.
        out[None] = qd.atomic_cas(f[None], 7, 42)

    kern()
    assert int(out[None]) == 7
    assert int(f[None]) == 42


# Pins that passing a raw Field (instead of `field[None]`) to atomic_cas raises a clear QuadrantsSyntaxError
# instead of a confusing AttributeError on x.ptr. Mirrors @writeback_binary's Field guard for the rest of the
# qd.atomic_* family. alanray-tech nit from PR #690 review.
@test_utils.test()
def test_atomic_cas_raw_field_raises_clear_error():
    f = qd.field(qd.i32, shape=())
    f[None] = 0

    @qd.kernel
    def kern():
        qd.atomic_cas(f, 0, 1)

    with pytest.raises(qd.lang.exception.QuadrantsCompilationError):
        kern()


# Pins that atomic_cas works on qd.ndarray elements, not just qd.field. Ndarrays go through a different
# access path (typed-NDArray kernel argument with physical-pointer subscript), so the surface needs to be
# exercised separately. Uses a contention pattern (single winner among N threads) to also verify atomicity
# on the ndarray surface, matching the field-side test_atomic_cas_contention_single_winner.
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_on_ndarray():
    N = 128
    INIT = 0

    @qd.kernel
    def kern(slot: qd.types.NDArray, olds: qd.types.NDArray) -> None:
        for i in range(N):
            olds[i] = qd.atomic_cas(slot[0], INIT, i + 1)

    slot = np.array([INIT], dtype=np.int32)
    olds = np.zeros(N, dtype=np.int32)
    kern(slot, olds)
    qd.sync()

    olds_seen = list(olds)
    winners = [i for i in range(N) if olds_seen[i] == INIT]
    assert len(winners) == 1, f"expected exactly one CAS winner, got {len(winners)}: indices {winners[:5]}..."
    winner_id = winners[0] + 1
    assert int(slot[0]) == winner_id, f"slot ended with {int(slot[0])}, expected winner's id {winner_id}"


# Pins that the documented workaround for tensor-CAS rejection actually works: extract a scalar component
# from a Vector field with `vec_field[None][i]` and CAS on that scalar. The MatrixPtrStmt-derived lvalue
# must reach codegen as a normal scalar AtomicOpStmt with `expected` populated. Without this, the docs
# would tell users to do something we don't actually support.
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_on_vector_field_scalar_component():
    f = qd.Vector.field(3, qd.i32, shape=())
    f[None] = qd.Vector([10, 20, 30])
    out = qd.field(qd.i32, shape=(3,))

    @qd.kernel
    def kern():
        # CAS each component independently. The expected-value matches in every case, so all three swaps fire.
        out[0] = qd.atomic_cas(f[None][0], 10, 100)
        out[1] = qd.atomic_cas(f[None][1], 20, 200)
        out[2] = qd.atomic_cas(f[None][2], 30, 300)

    kern()
    assert int(out[0]) == 10 and int(f[None][0]) == 100
    assert int(out[1]) == 20 and int(f[None][1]) == 200
    assert int(out[2]) == 30 and int(f[None][2]) == 300


# Pins that the `expected`-cast in AtomicOpExpression::type_check handles signed -> unsigned widening too.
# `qd.atomic_cas(u32_field[None], -1, 0)` passes -1 as a Python int (default i32). The cast must convert
# it through cast_value to u32 (=0xFFFFFFFF), letting the swap-from-sentinel pattern work on unsigned
# fields. Same shape for u64. Complements test_atomic_cas_expected_int_literal_widens_to_i64 which only
# covers the positive i64 case.
@pytest.mark.parametrize("dtype", [qd.u32, qd.u64])
@test_utils.test(arch=qd.gpu)
def test_atomic_cas_expected_signed_int_literal_casts_to_unsigned(dtype):
    _skip_if_no_int64_atomic_rmw(dtype)
    sentinel = (1 << 32) - 1 if dtype == qd.u32 else (1 << 64) - 1
    f = qd.field(dtype, shape=())
    out = qd.field(dtype, shape=())
    f[None] = sentinel

    @qd.kernel
    def kern():
        # -1 is a Python int (i32 default). After signed -> unsigned cast it becomes the all-ones sentinel,
        # so the CAS fires and the slot is cleared to 0.
        out[None] = qd.atomic_cas(f[None], -1, 0)

    kern()
    assert int(out[None]) == sentinel
    assert int(f[None]) == 0
