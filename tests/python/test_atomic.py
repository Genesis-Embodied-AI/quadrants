import math

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


# Pins the doc claim that atomic_mul works (via CAS loop) under
# multi-thread contention on every GPU backend, for both ints and
# floats including f64. Existing coverage is single-thread only
# (test_atomic_mul_f32, test_atomic_mul_expr_evaled). Values chosen
# so the product is representable exactly in i32 / f32 / f64.
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


# Pins the doc claim that floating-point atomic_min / atomic_max use
# minNum / maxNum-style NaN semantics: when exactly one operand is NaN,
# the non-NaN value is written back. Asserted on every GPU backend and
# both arg orders. CPU is intentionally out of scope here -- the doc
# explicitly warns that the CPU CAS path uses naive </</> > comparisons
# with order-dependent results.
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


# Pins the doc claim that vector/matrix arguments to atomic ops fan out to one
# scalar atomic per component (no all-or-nothing guarantee across components,
# but every component must be summed exactly).
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

    Metal sets ``spirv_has_atomic_int64=1`` on Apple7+ / Mac2 in
    ``metal_device.mm`` (the gate is misnamed ``feature_floating_point_atomics``),
    but MSL only exposes 64-bit atomics as ``atomic_fetch_min/max`` on ``uint64``
    starting at Apple9 (M3+, A17+); ``atomic_add`` / ``and`` / ``or`` / ``xor``
    are not available at all. The pipeline create then fails with RhiResult=-1
    ("SPIR-V shader was rejected by the backend"). Tightening the Metal cap
    itself is intentionally out of scope here -- the cap is consumed by adstack
    fallbacks in ``runtime/gfx`` and lowering it needs a separate audit -- so we
    just skip the affected dtypes on Metal instead.
    """
    if dtype not in (qd.i64, qd.u64):
        return
    if qd.cfg.arch == qd.metal:
        pytest.skip("Metal lacks general-purpose 64-bit integer atomic RMW (MSL atomic_long fetch_add/and/or/xor)")
    if qd.cfg.arch == qd.vulkan:
        caps = qd.lang.impl.get_runtime().prog.get_device_caps()
        if not caps.get(qd._lib.core.DeviceCapability.spirv_has_atomic_int64):
            pytest.skip("Vulkan device does not advertise spirv_has_atomic_int64")


# Pins the doc-table claim that atomic_add is "yes" on every integer
# dtype (i32 / u32 / i64 / u64) on every GPU backend. Existing coverage
# only exercises i32 (run_atomic_add_global_case) and u64 min/max
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


# Pins the doc-table claim that atomic_and / atomic_or / atomic_xor are
# "yes" on every integer dtype on every GPU backend. Existing coverage
# only exercises i32 (test_atomic_{and,or,xor}_expr_evaled).
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


# Pins the doc claim that bitwise atomics on float dtypes raise a type
# error at trace time (atomics page: "Integer dtypes only -- passing
# f32 / f64 raises a type error at trace time"). Enforced by the
# is_integral check in AtomicOpExpression::type_check
# (quadrants/ir/frontend_ir.cpp) for bit_and / bit_or / bit_xor.
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
