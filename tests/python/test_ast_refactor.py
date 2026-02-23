import platform

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl
from quadrants.lang.util import has_pytorch

from tests import test_utils

u = platform.uname()


@test_utils.test()
def test_namedexpr():
    @qd.kernel
    def foo() -> qd.i32:
        b = 2 + (a := 5)
        b += a
        return b

    assert foo() == 12


@test_utils.test()
def test_binop():
    @qd.kernel
    def foo(x: qd.i32, y: qd.i32, a: qd.template()):
        a[0] = x + y
        a[1] = x - y
        a[2] = x * y
        a[3] = impl.ti_float(x) / y
        a[4] = x // y
        a[5] = x % y
        a[6] = x**y
        a[7] = x << y
        a[8] = x >> y
        a[9] = x | y
        a[10] = x ^ y
        a[11] = x & y

    x = 37
    y = 3
    a = qd.field(qd.f32, shape=(12,))
    b = qd.field(qd.f32, shape=(12,))

    a[0] = x + y
    a[1] = x - y
    a[2] = x * y
    a[3] = x / y
    a[4] = x // y
    a[5] = x % y
    a[6] = x**y
    a[7] = x << y
    a[8] = x >> y
    a[9] = x | y
    a[10] = x ^ y
    a[11] = x & y

    foo(x, y, b)

    for i in range(12):
        assert a[i] == test_utils.approx(b[i])


@test_utils.test()
def test_augassign():
    @qd.kernel
    def foo(x: qd.i32, y: qd.i32, a: qd.template(), b: qd.template()):
        for i in a:
            a[i] = x
        a[0] += y
        a[1] -= y
        a[2] *= y
        a[3] //= y
        a[4] %= y
        a[5] **= y
        a[6] <<= y
        a[7] >>= y
        a[8] |= y
        a[9] ^= y
        a[10] &= y
        b[0] = x
        b[0] /= y

    x = 37
    y = 3
    a = qd.field(qd.i32, shape=(11,))
    b = qd.field(qd.i32, shape=(11,))
    c = qd.field(qd.f32, shape=(1,))
    d = qd.field(qd.f32, shape=(1,))

    a[0] = x + y
    a[1] = x - y
    a[2] = x * y
    a[3] = x // y
    a[4] = x % y
    a[5] = x**y
    a[6] = x << y
    a[7] = x >> y
    a[8] = x | y
    a[9] = x ^ y
    a[10] = x & y
    c[0] = x / y

    foo(x, y, b, d)

    for i in range(11):
        assert a[i] == b[i]
    assert c[0] == test_utils.approx(d[0])


@test_utils.test()
def test_unaryop():
    @qd.kernel
    def foo(x: qd.i32, a: qd.template()):
        a[0] = +x
        a[1] = -x
        a[2] = not x
        a[3] = ~x

    x = 1234
    a = qd.field(qd.i32, shape=(4,))
    b = qd.field(qd.i32, shape=(4,))

    a[0] = +x
    a[1] = -x
    a[2] = not x
    a[3] = ~x

    foo(x, b)

    for i in range(4):
        assert a[i] == b[i]


@test_utils.test()
def test_boolop():
    @qd.kernel
    def foo(a: qd.template()):
        a[0] = 0 and 0
        a[1] = 0 and 1
        a[2] = 1 and 0
        a[3] = 1 and 1
        a[4] = 0 or 0
        a[5] = 0 or 1
        a[6] = 1 or 0
        a[7] = 1 or 1
        a[8] = 1 and 1 and 1 and 1
        a[9] = 1 and 1 and 1 and 0
        a[10] = 0 or 0 or 0 or 0
        a[11] = 0 or 0 or 1 or 0

    a = qd.field(qd.i32, shape=(12,))
    b = qd.field(qd.i32, shape=(12,))

    a[0] = 0 and 0
    a[1] = 0 and 1
    a[2] = 1 and 0
    a[3] = 1 and 1
    a[4] = 0 or 0
    a[5] = 0 or 1
    a[6] = 1 or 0
    a[7] = 1 or 1
    a[8] = 1 and 1 and 1 and 1
    a[9] = 1 and 1 and 1 and 0
    a[10] = 0 or 0 or 0 or 0
    a[11] = 0 or 0 or 1 or 0

    foo(b)

    for i in range(12):
        assert a[i] == b[i]


@test_utils.test()
def test_single_compare():
    @qd.kernel
    def foo(a: qd.template(), b: qd.template(), c: qd.template()):
        for i in qd.static(range(3)):
            c[i * 6] = a[i] == b[i]
            c[i * 6 + 1] = a[i] != b[i]
            c[i * 6 + 2] = a[i] < b[i]
            c[i * 6 + 3] = a[i] <= b[i]
            c[i * 6 + 4] = a[i] > b[i]
            c[i * 6 + 5] = a[i] >= b[i]

    a = qd.Vector([1, 1, 2])
    b = qd.Vector([2, 1, 1])
    c = qd.field(qd.i32, shape=(18,))
    d = qd.field(qd.i32, shape=(18,))

    for i in range(3):
        c[i * 6] = a[i] == b[i]
        c[i * 6 + 1] = a[i] != b[i]
        c[i * 6 + 2] = a[i] < b[i]
        c[i * 6 + 3] = a[i] <= b[i]
        c[i * 6 + 4] = a[i] > b[i]
        c[i * 6 + 5] = a[i] >= b[i]

    foo(a, b, d)
    for i in range(18):
        assert c[i] == d[i]


@test_utils.test()
def test_chain_compare():
    @qd.kernel
    def foo(a: qd.i32, b: qd.i32, c: qd.template()):
        c[0] = a == b == a
        c[1] = a == b != a
        c[2] = a != b == a
        c[3] = a < b > a
        c[4] = a > b < a
        c[5] = a < b < a
        c[6] = a > b > a
        c[7] = a == a == a == a
        c[8] = a == a == a != a
        c[9] = a < b > a < b
        c[10] = a > b > a < b

    a = 1
    b = 2
    c = qd.field(qd.i32, shape=(11,))
    d = qd.field(qd.i32, shape=(11,))

    c[0] = a == b == a
    c[1] = a == b != a
    c[2] = a != b == a
    c[3] = a < b > a
    c[4] = a > b < a
    c[5] = a < b < a
    c[6] = a > b > a
    c[7] = a == a == a == a
    c[8] = a == a == a != a
    c[9] = a < b > a < b
    c[10] = a > b > a < b

    foo(a, b, d)
    for i in range(11):
        assert c[i] == d[i]


@test_utils.test()
def test_return():
    @qd.kernel
    def foo(x: qd.i32) -> qd.i32:
        return x + 1

    assert foo(1) == 2


@test_utils.test()
def test_format_print():
    a = qd.field(qd.i32, shape=(10,))

    @qd.kernel
    def foo():
        a[0] = 1.0
        a[5] = 2.0
        print("Test if the string.format and fstring print works")
        print("string.format: a[0]={}, a[5]={}".format(a[0], a[5]))
        print(f"fstring: a[0]={a[0]}, a[5]={a[5]}")


@test_utils.test(print_preprocessed_ir=True)
def test_if():
    @qd.kernel
    def foo(x: qd.i32) -> qd.i32:
        ret = 0
        if x:
            ret = 1
        else:
            ret = 0
        return ret

    assert foo(1)
    assert not foo(0)


@test_utils.test(print_preprocessed_ir=True)
def test_static_if():
    @qd.kernel
    def foo(x: qd.template()) -> qd.i32:
        ret = 0
        if qd.static(x):
            ret = 1
        else:
            ret = 0
        return ret

    assert foo(1)
    assert not foo(0)


@test_utils.test(print_preprocessed_ir=True)
def test_struct_for():
    a = qd.field(qd.i32, shape=(10,))

    @qd.kernel
    def foo(x: qd.i32):
        for i in a:
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        assert a[i] == 5


@test_utils.test(print_preprocessed_ir=True)
def test_grouped_struct_for():
    a = qd.field(qd.i32, shape=(4, 4))

    @qd.kernel
    def foo(x: qd.i32):
        for I in qd.grouped(a):
            a[I] = x

    x = 5
    foo(x)
    for i in range(4):
        for j in range(4):
            assert a[i, j] == 5


@test_utils.test(print_preprocessed_ir=True)
def test_static_for():
    a = qd.field(qd.i32, shape=(10,))

    @qd.kernel
    def foo(x: qd.i32):
        for i in qd.static(range(10)):
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        assert a[i] == 5


@test_utils.test(print_preprocessed_ir=True)
def test_static_grouped_for():
    a = qd.field(qd.i32, shape=(4, 4))

    @qd.kernel
    def foo(x: qd.i32):
        for i in qd.static(qd.grouped(qd.ndrange((1, 3), (1, 3)))):
            a[i] = x

    x = 5
    foo(x)
    for i in range(4):
        for j in range(4):
            if 1 <= i < 3 and 1 <= j < 3:
                assert a[i, j] == 5
            else:
                assert a[i, j] == 0


@test_utils.test(print_preprocessed_ir=True)
def test_range_for_single_argument():
    a = qd.field(qd.i32, shape=(10,))

    @qd.kernel
    def foo(x: qd.i32):
        for i in range(5):
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        if i < 5:
            assert a[i] == 5
        else:
            assert a[i] == 0


@test_utils.test(print_preprocessed_ir=True)
def test_range_for_two_arguments():
    a = qd.field(qd.i32, shape=(10,))

    @qd.kernel
    def foo(x: qd.i32):
        for i in range(3, 7):
            a[i] = x

    x = 5
    foo(x)
    for i in range(10):
        if 3 <= i < 7:
            assert a[i] == 5
        else:
            assert a[i] == 0


@test_utils.test()
def test_range_for_three_arguments():
    a = qd.field(qd.i32, shape=(10,))

    with pytest.raises(qd.QuadrantsCompilationError, match="Range should have 1 or 2 arguments, found 3"):

        @qd.kernel
        def foo(x: qd.i32):
            for i in range(3, 7, 2):
                a[i] = x

        x = 5
        foo(x)


@test_utils.test(print_preprocessed_ir=True)
def test_ndrange_for():
    x = qd.field(qd.f32, shape=(16, 32, 64))

    @qd.kernel
    def func():
        for i, j, k in qd.ndrange((4, 10), (3, 8), 17):
            x[i, j, k] = i + j * 10 + k * 100

    func()
    for i in range(16):
        for j in range(32):
            for k in range(64):
                if 4 <= i < 10 and 3 <= j < 8 and k < 17:
                    assert x[i, j, k] == i + j * 10 + k * 100
                else:
                    assert x[i, j, k] == 0


@test_utils.test(print_preprocessed_ir=True)
def test_grouped_ndrange_for():
    x = qd.field(qd.i32, shape=(6, 6, 6))
    y = qd.field(qd.i32, shape=(6, 6, 6))

    @qd.kernel
    def func():
        lower = qd.Vector([0, 1, 2])
        upper = qd.Vector([3, 4, 5])
        for I in qd.grouped(qd.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
            x[I] = I[0] + I[1] + I[2]
        for i in range(0, 3):
            for j in range(1, 4):
                for k in range(2, 5):
                    y[i, j, k] = i + j + k

    func()

    for i in range(6):
        for j in range(6):
            for k in range(6):
                assert x[i, j, k] == y[i, j, k]


@test_utils.test(print_preprocessed_ir=True)
def test_static_for_break():
    n = 10

    @qd.kernel
    def foo(a: qd.template()):
        for i in qd.static(range(n)):
            a[i] = 3
            if qd.static(i >= 5):
                break
                a[i] = 10
            a[i] = 5

    a = qd.field(qd.i32, shape=(n,))
    foo(a)
    for i in range(n):
        if i < 5:
            assert a[i] == 5
        elif i == 5:
            assert a[i] == 3
        else:
            assert a[i] == 0


@test_utils.test(print_preprocessed_ir=True)
def test_static_grouped_for_break():
    n = 4

    @qd.kernel
    def foo(a: qd.template()):
        for I in qd.static(qd.grouped(qd.ndrange(n, n))):
            a[I] = 3
            if qd.static(I[0] >= 3):
                break
                a[I] = 10
            a[I] = 5

    a = qd.field(qd.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            elif i == 3 and j == 0:
                assert a[i, j] == 3
            else:
                assert a[i, j] == 0


@test_utils.test(print_preprocessed_ir=True)
def test_static_for_continue():
    n = 10

    @qd.kernel
    def foo(a: qd.template()):
        for i in qd.static(range(n)):
            a[i] = 3
            if qd.static(i >= 5):
                continue
                a[i] = 10
            a[i] = 5

    a = qd.field(qd.i32, shape=(n,))
    foo(a)
    for i in range(n):
        if i < 5:
            assert a[i] == 5
        else:
            assert a[i] == 3


@test_utils.test(print_preprocessed_ir=True)
def test_static_grouped_for_continue():
    n = 4

    @qd.kernel
    def foo(a: qd.template()):
        for I in qd.static(qd.grouped(qd.ndrange(n, n))):
            a[I] = 3
            if qd.static(I[0] >= 3):
                continue
                a[I] = 10
            a[I] = 5

    a = qd.field(qd.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            else:
                assert a[i, j] == 3


@test_utils.test(print_preprocessed_ir=True)
def test_for_break():
    n = 4

    @qd.kernel
    def foo(a: qd.template()):
        for i in range(n):
            for j in range(n):
                a[i, j] = 3
                if i >= 3:
                    break
                    a[i, j] = 10
                a[i, j] = 5

    a = qd.field(qd.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            elif i == 3 and j == 0:
                assert a[i, j] == 3
            else:
                assert a[i, j] == 0


@test_utils.test(print_preprocessed_ir=True)
def test_for_continue():
    n = 4

    @qd.kernel
    def foo(a: qd.template()):
        for i in range(n):
            for j in range(n):
                a[i, j] = 3
                if i >= 3:
                    continue
                    a[i, j] = 10
                a[i, j] = 5

    a = qd.field(qd.i32, shape=(n, n))
    foo(a)
    for i in range(n):
        for j in range(n):
            if i < 3:
                assert a[i, j] == 5
            else:
                assert a[i, j] == 3


@test_utils.test()
def test_while():
    x = qd.field(qd.f32)

    N = 1

    qd.root.dense(qd.i, N).place(x)

    @qd.kernel
    def func():
        i = 0
        s = 0
        while i < 10:
            s += i
            i += 1
        x[0] = s

    func()
    assert x[0] == 45


@test_utils.test()
def test_while_break():
    ret = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        i = 0
        s = 0
        while True:
            s += i
            i += 1
            if i > 10:
                break
        ret[None] = s

    func()
    assert ret[None] == 55


@test_utils.test()
def test_while_continue():
    ret = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        i = 0
        s = 0
        while i < 10:
            i += 1
            if i % 2 == 0:
                continue
            s += i
        ret[None] = s

    func()
    assert ret[None] == 25


@test_utils.test(print_preprocessed_ir=True)
def test_func():
    @qd.func
    def bar(x):
        return x * x, -x

    a = qd.field(qd.i32, shape=(10,))
    b = qd.field(qd.i32, shape=(10,))

    @qd.kernel
    def foo():
        for i in a:
            a[i], b[i] = bar(i)

    foo()
    for i in range(10):
        assert a[i] == i * i
        assert b[i] == -i


@test_utils.test(print_preprocessed_ir=True)
def test_func_in_python_func():
    @qd.func
    def bar(x: qd.template()):
        if qd.static(x):
            mat = bar(x // 2)
            mat = mat @ mat
            if qd.static(x % 2):
                mat = mat @ qd.Matrix([[1, 1], [1, 0]])
            return mat
        else:
            return qd.Matrix([[1, 0], [0, 1]])

    def fibonacci(x):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        return impl.subscript(ast_builder, bar(x), 1, 0)

    @qd.kernel
    def foo(x: qd.template()) -> qd.i32:
        return fibonacci(x)

    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    for i in range(10):
        assert foo(i) == fib[i]


@test_utils.test(print_preprocessed_ir=True)
def test_ifexp():
    @qd.kernel
    def foo(x: qd.i32) -> qd.i32:
        return 1 if x else 0

    assert foo(1) == 1
    assert foo(0) == 0


@test_utils.test(print_preprocessed_ir=True)
def test_static_ifexp():
    @qd.kernel
    def foo(x: qd.template()) -> qd.i32:
        return 1 if qd.static(x) else 0

    assert foo(1) == 1
    assert foo(0) == 0


@test_utils.test()
def test_static_assign():
    a = qd.field(qd.i32, shape=(1,))
    b = qd.field(qd.i32, shape=(1,))

    @qd.kernel
    def foo(xx: qd.template(), yy: qd.template()) -> qd.i32:
        x, y = qd.static(xx, yy)
        x[0] -= 1
        y[0] -= 1
        return x[0] + y[0]

    a[0] = 2
    b[0] = 3
    assert foo(a, b) == 3


@test_utils.test()
def test_static_assign_element():
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Static assign cannot be used on elements in arrays",
    ):

        @qd.kernel
        def foo():
            a = qd.static([1, 2, 3])
            a[0] = qd.static(2)

        foo()


@test_utils.test()
def test_recreate_variable():
    with pytest.raises(qd.QuadrantsCompilationError, match="Recreating variables is not allowed"):

        @qd.kernel
        def foo():
            a = 1
            a = qd.static(2)

        foo()


@test_utils.test()
def test_quadrants_other_than_ti():
    import quadrants as tc

    @tc.func
    def bar(x: tc.template()):
        if tc.static(x):
            mat = bar(x // 2)
            mat = mat @ mat
            if tc.static(x % 2):
                mat = mat @ tc.Matrix([[1, 1], [1, 0]])
            return mat
        else:
            return tc.Matrix([[1, 0], [0, 1]])

    def fibonacci(x):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        return impl.subscript(ast_builder, bar(x), 1, 0)

    @tc.kernel
    def foo(x: tc.template()) -> tc.i32:
        return fibonacci(x)

    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    for i in range(10):
        assert foo(i) == fib[i]


@pytest.mark.skipif(
    u.system == "linux" and u.machine in ("arm64", "aarch64"),
    reason="assert not currently supported on linux arm64 or aarch64",
)
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message():
    @qd.kernel
    def func():
        x = 20
        assert 10 <= x < 20, "Foo bar"

    with pytest.raises(RuntimeError, match="Foo bar"):
        func()


@pytest.mark.skipif(
    u.system == "linux" and u.machine in ("arm64", "aarch64"),
    reason="assert not currently supported on linux arm64 or aarch64",
)
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_assert_message_formatted():
    x = qd.field(dtype=int, shape=16)
    x[10] = 42

    @qd.kernel
    def assert_formatted():
        for i in x:
            assert x[i] == 0, "x[%d] expect=%d got=%d" % (i, 0, x[i])

    @qd.kernel
    def assert_float():
        y = 0.5
        assert y < 0, "y = %f" % y

    with pytest.raises(RuntimeError, match=r"x\[10\] expect=0 got=42"):
        assert_formatted()
    # TODO: note that we are not fully polished to be able to recover from
    # assertion failures...
    with pytest.raises(RuntimeError, match=r"y = 0.5"):
        assert_float()

    # success case
    x[10] = 0
    assert_formatted()


@test_utils.test(print_full_traceback=False)
def test_dict():
    @qd.kernel
    def foo(x: qd.template()) -> qd.i32:
        a = {1: 2, 3: 4}
        b = {5: 6, **a}
        return b[x]

    assert foo(1) == 2
    with pytest.raises(qd.QuadrantsCompilationError):
        foo(2)


@test_utils.test()
def test_single_listcomp():
    @qd.func
    def identity(dt, n: qd.template()):
        return qd.Matrix([[qd.cast(int(i == j), dt) for j in range(n)] for i in range(n)])

    @qd.kernel
    def foo(n: qd.template()) -> qd.i32:
        a = identity(qd.i32, n)
        b = [i[0] for i in a]
        ret = 0
        for i in qd.static(range(n)):
            ret += b[i]
        return ret

    assert foo(5) == 1


@test_utils.test()
def test_listcomp():
    @qd.func
    def identity(dt, n: qd.template()):
        return qd.Matrix([[qd.cast(int(i == j), dt) for j in range(n)] for i in range(n)])

    @qd.kernel
    def foo(n: qd.template()) -> qd.i32:
        a = identity(qd.i32, n)
        b = [j for i in a for j in i]
        ret = 0
        for i in qd.static(range(n)):
            for j in qd.static(range(n)):
                ret += i * j * b[i * n + j]
        return ret

    assert foo(5) == 1 + 4 + 9 + 16


@test_utils.test()
def test_dictcomp():
    @qd.kernel
    def foo(n: qd.template()) -> qd.i32:
        a = {i: i * i for i in range(n) if i % 3 if i % 2}
        ret = 0
        for i in qd.static(range(n)):
            if qd.static(i % 3):
                if qd.static(i % 2):
                    ret += a[i]
        return ret

    assert foo(10) == 1 * 1 + 5 * 5 + 7 * 7


@test_utils.test(print_full_traceback=False)
def test_dictcomp_fail():
    @qd.kernel
    def foo(n: qd.template(), m: qd.template()) -> qd.i32:
        a = {i: i * i for i in range(n) if i % 3 if i % 2}
        return a[m]

    with pytest.raises(qd.QuadrantsCompilationError):
        foo(5, 2)

    with pytest.raises(qd.QuadrantsCompilationError):
        foo(5, 3)


@pytest.mark.skipif(not has_pytorch(), reason="Pytorch not installed.")
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_ndarray():
    n = 4
    m = 7

    @qd.kernel
    def run(x: qd.types.ndarray(dtype=qd.types.matrix(1, 1, qd.i32)), y: qd.types.ndarray()):
        for i in qd.static(range(n)):
            for j in qd.static(range(m)):
                x[i, j][0, 0] += i + j + y[i, j]

    a = qd.Matrix.ndarray(1, 1, qd.i32, shape=(n, m))
    for i in range(n):
        for j in range(m):
            a[i, j][0, 0] = i * j
    b = np.ones((n, m), dtype=np.int32)
    run(a, b)
    for i in range(n):
        for j in range(m):
            assert a[i, j][0, 0] == i * j + i + j + 1


@test_utils.test(arch=qd.cpu)
def test_sparse_matrix_builder():
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i, j in qd.static(qd.ndrange(n, n)):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


@test_utils.test()
def test_func_default_value():
    @qd.func
    def bar(s, t=1):
        return s + t

    @qd.kernel
    def foo() -> qd.i32:
        return bar(1)

    assert foo() == 2


@test_utils.test()
def test_func_default_value_fail():
    with pytest.raises(qd.QuadrantsCompilationError):

        @qd.func
        def bar(s, t=1):
            return s + t

        @qd.kernel
        def foo() -> qd.i32:
            return bar(1, 2, 3)

        foo()


@test_utils.test(print_full_traceback=False)
def test_raise():
    dim = 1
    m = qd.Matrix.field(dim, dim, qd.f32)
    qd.root.place(m)

    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Polar decomposition only supports 2D and 3D matrices.",
    ):

        @qd.kernel
        def foo():
            qd.polar_decompose(m, qd.f32)

        foo()


@test_utils.test()
def test_default_template_args_on_func():
    @qd.func
    def bar(a: qd.template() = 123):
        return a

    @qd.kernel
    def foo() -> qd.i32:
        return bar()

    assert foo() == 123


@test_utils.test()
def test_grouped_static_for_cast():
    @qd.kernel
    def foo() -> qd.f32:
        ret = 0.0
        for I in qd.static(qd.grouped(qd.ndrange((4, 5), (3, 5), 5))):
            tmp = I.cast(float)
            ret += tmp[2] / 2
        return ret

    assert foo() == test_utils.approx(10)
