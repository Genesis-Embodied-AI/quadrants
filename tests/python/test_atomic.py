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


@test_utils.test(arch=[qd.cpu, qd.cuda])
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


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_atomic_sub_with_type_promotion():
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


@test_utils.test()
def test_atomic_mul_f32():
    @qd.kernel
    def mul_kernel() -> qd.f32:
        x = 1.0
        for i in range(1, 8):
            qd.atomic_mul(x, qd.f32(i))

        return x

    assert mul_kernel() == 5040.0
