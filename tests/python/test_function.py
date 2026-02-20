import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_function_without_return():
    x = qd.field(qd.i32, shape=())

    @qd.real_func
    def foo(val: qd.i32):
        x[None] += val

    @qd.kernel
    def run():
        foo(40)
        foo(2)

    x[None] = 0
    run()
    assert x[None] == 42


@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True)
def test_function_with_return():
    x = qd.field(qd.i32, shape=())

    @qd.real_func
    def foo(val: qd.i32) -> qd.i32:
        x[None] += val
        return val

    @qd.kernel
    def run():
        a = foo(40)
        foo(2)
        assert a == 40

    x[None] = 0
    run()
    assert x[None] == 42


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_call_expressions():
    x = qd.field(qd.i32, shape=())

    @qd.real_func
    def foo(val: qd.i32) -> qd.i32:
        if x[None] > 10:
            x[None] += 1
        x[None] += val
        return 0

    @qd.kernel
    def run():
        assert foo(15) == 0
        assert foo(10) == 0

    x[None] = 0
    run()
    assert x[None] == 26


@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True)
def test_default_templates():
    @qd.func
    def func1(x: qd.template()):
        x = 1

    @qd.func
    def func2(x: qd.template()):
        x += 1

    @qd.func
    def func3(x):
        x = 1

    @qd.func
    def func4(x):
        x += 1

    @qd.func
    def func1_field(x: qd.template()):
        x[None] = 1

    @qd.func
    def func2_field(x: qd.template()):
        x[None] += 1

    @qd.func
    def func3_field(x):
        x[None] = 1

    @qd.func
    def func4_field(x):
        x[None] += 1

    v = qd.field(dtype=qd.i32, shape=())

    @qd.kernel
    def run_func():
        a = 0
        func1(a)
        assert a == 1
        b = 0
        func2(b)
        assert b == 1
        c = 0
        func3(c)
        assert c == 0
        d = 0
        func4(d)
        assert d == 0

        v[None] = 0
        func1_field(v)
        assert v[None] == 1
        v[None] = 0
        func2_field(v)
        assert v[None] == 1
        v[None] = 0
        func3_field(v)
        assert v[None] == 1
        v[None] = 0
        func4_field(v)
        assert v[None] == 1

    run_func()


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_experimental_templates():
    x = qd.field(qd.i32, shape=())
    y = qd.field(qd.i32, shape=())
    answer = qd.field(qd.i32, shape=8)

    @qd.kernel
    def kernel_inc(x: qd.template()):
        x[None] += 1

    def run_kernel():
        x[None] = 10
        y[None] = 20
        kernel_inc(x)
        assert x[None] == 11
        assert y[None] == 20
        kernel_inc(y)
        assert x[None] == 11
        assert y[None] == 21

    @qd.real_func
    def inc(x: qd.template()):
        x[None] += 1

    @qd.kernel
    def run_func(a: qd.u1):
        x[None] = 10
        y[None] = 20
        if a:
            inc(x)
        answer[0] = x[None]
        answer[1] = y[None]
        if a:
            inc(y)
        answer[2] = x[None]
        answer[3] = y[None]

    def verify():
        assert answer[0] == 11
        assert answer[1] == 20
        assert answer[2] == 11
        assert answer[3] == 21

    run_kernel()
    run_func(True)
    verify()


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_missing_arg_annotation():
    with pytest.raises(qd.QuadrantsSyntaxError, match="must be type annotated"):

        @qd.real_func
        def add(a, b: qd.i32) -> qd.i32:
            return a + b


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_missing_return_annotation():
    with pytest.raises(qd.QuadrantsCompilationError, match="return value must be annotated"):

        @qd.real_func
        def add(a: qd.i32, b: qd.i32):
            return a + b

        @qd.kernel
        def run():
            add(30, 2)

        run()


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_different_argument_type():
    @qd.real_func
    def add(a: qd.f32, b: qd.f32) -> qd.f32:
        return a + b

    @qd.kernel
    def run() -> qd.i32:
        return add(1, 2)

    assert run() == 3


@pytest.mark.run_in_serial
@test_utils.test(arch=[qd.cpu, qd.cuda], cuda_stack_limit=8192)
def test_recursion():
    @qd.real_func
    def sum(f: qd.template(), l: qd.i32, r: qd.i32) -> qd.i32:
        if l == r:
            return f[l]
        else:
            return sum(f, l, (l + r) // 2) + sum(f, (l + r) // 2 + 1, r)

    f = qd.field(qd.i32, shape=100)
    for i in range(100):
        f[i] = i

    @qd.kernel
    def get_sum() -> qd.i32:
        return sum(f, 0, 99)

    assert get_sum() == 99 * 50


@pytest.mark.run_in_serial
@test_utils.test(arch=[qd.cpu, qd.cuda], cuda_stack_limit=32768)
def test_deep_recursion():
    @qd.real_func
    def sum_func(n: qd.i32) -> qd.i32:
        if n == 0:
            return 0
        return sum_func(n - 1) + n

    @qd.kernel
    def sum(n: qd.i32) -> qd.i32:
        return sum_func(n)

    assert sum(100) == 5050


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_multiple_return():
    x = qd.field(qd.i32, shape=())

    @qd.real_func
    def foo(val: qd.i32) -> qd.i32:
        if x[None] > 10:
            if x[None] > 20:
                return 1
            x[None] += 1
        x[None] += val
        return 0

    @qd.kernel
    def run():
        assert foo(15) == 0
        assert foo(10) == 0
        assert foo(100) == 1

    x[None] = 0
    run()
    assert x[None] == 26


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_return_in_for():
    @qd.real_func
    def foo() -> qd.i32:
        for i in range(10):
            return 42

    @qd.kernel
    def bar() -> qd.i32:
        return foo()

    assert bar() == 42


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_return_in_while():
    @qd.real_func
    def foo() -> qd.i32:
        i = 1
        while i:
            return 42

    @qd.kernel
    def bar() -> qd.i32:
        return foo()

    assert bar() == 42


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_return_in_if_in_for():
    @qd.real_func
    def foo(a: qd.i32) -> qd.i32:
        s = 0
        for i in range(100):
            if i == a + 1:
                return s
            s = s + i
        return s

    @qd.kernel
    def bar(a: qd.i32) -> qd.i32:
        return foo(a)

    assert bar(10) == 11 * 5
    assert bar(200) == 99 * 50


@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True)
def test_ref():
    @qd.real_func
    def foo(a: qd.ref(qd.f32)):
        a = 7

    @qd.kernel
    def bar():
        a = 5.0
        foo(a)
        assert a == 7

    bar()


@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True)
def test_ref_atomic():
    # FIXME: failed test on Pascal (and potentially older) architecture.
    # Please remove this guardiance when you fix this issue
    cur_arch = qd.lang.impl.get_runtime().prog.config().arch
    if cur_arch == qd.cuda and qd.lang.impl.get_cuda_compute_capability() < 70:
        pytest.skip(
            "Skip this test on Pascal (and potentially older) architecture, ask turbo0628/Proton for more information"
        )

    @qd.real_func
    def foo(a: qd.ref(qd.f32)):
        a += a

    @qd.kernel
    def bar():
        a = 5.0
        foo(a)
        assert a == 10.0

    bar()


@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True, print_full_traceback=False)
def test_func_ndarray_arg():
    vec3 = qd.types.vector(3, qd.f32)

    @qd.func
    def test(a: qd.types.ndarray(ndim=1)):
        a[0] = [100, 100, 100]

    @qd.kernel
    def test_k(x: qd.types.ndarray(ndim=1)):
        test(x)

    @qd.func
    def test_error_func(a: qd.types.ndarray(dtype=qd.math.vec2, ndim=1)):
        a[0] = [100, 100]

    @qd.kernel
    def test_error(x: qd.types.ndarray(ndim=1)):
        test_error_func(x)

    arr = qd.ndarray(vec3, shape=(4))
    arr[0] = [20, 20, 20]
    test_k(arr)

    assert arr[0] == [20, 20, 20]

    with pytest.raises(qd.QuadrantsCompilationError, match=r"Invalid value for argument a"):
        test_error(arr)


@test_utils.test(debug=True)
def test_func_matrix_arg():
    vec3 = qd.types.vector(3, qd.f32)

    @qd.func
    def test(a: vec3):
        a[0] = 100

    @qd.kernel
    def test_k():
        x = qd.Matrix([3, 4, 5])
        x[0] = 20
        test(x)

        assert x[0] == 20

    test_k()


@test_utils.test()
def test_func_matrix_arg_with_error():
    vec3 = qd.types.vector(3, qd.f32)

    @qd.func
    def test(a: vec3):
        a[0] = 100

    @qd.kernel
    def test_error():
        x = qd.Matrix([3, 4])
        test(x)

    with pytest.raises(qd.QuadrantsSyntaxError, match=r"is expected to be a Matrix with n 3, but got 2"):
        test_error()


@test_utils.test(debug=True)
def test_func_struct_arg():
    @qd.dataclass
    class C:
        i: int

    @qd.func
    def f(c: C):
        return c.i

    @qd.kernel
    def k():
        c = C(i=2)
        assert f(c) == 2

    k()


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_matrix_arg():
    @qd.real_func
    def mat_arg(a: qd.math.mat2, b: qd.math.vec2) -> float:
        return a[0, 0] + a[0, 1] + a[1, 0] + a[1, 1] + b[0] + b[1]

    b = qd.Vector.field(n=2, dtype=float, shape=())
    b[()][0] = 5
    b[()][1] = 6

    @qd.kernel
    def foo() -> float:
        a = qd.math.mat2(1, 2, 3, 4)
        return mat_arg(a, b[()])

    assert foo() == pytest.approx(21)


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_matrix_return():
    @qd.real_func
    def mat_ret() -> qd.math.mat2:
        return qd.math.mat2(1, 2, 3, 4)

    @qd.kernel
    def foo() -> qd.math.mat2:
        return mat_ret()

    assert (foo() == qd.math.mat2(1, 2, 3, 4)).all()


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_struct_ret():
    s = qd.types.struct(a=qd.i16, b=qd.f64)

    @qd.real_func
    def bar() -> s:
        return s(a=123, b=qd.f64(1.2345e300))

    @qd.kernel
    def foo() -> qd.f64:
        a = bar()
        return a.a * a.b

    assert foo() == pytest.approx(123 * 1.2345e300)


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_struct_ret_with_matrix():
    s0 = qd.types.struct(a=qd.math.vec3, b=qd.i16)
    s1 = qd.types.struct(a=qd.f32, b=s0)

    @qd.real_func
    def bar() -> s1:
        return s1(a=1, b=s0(a=qd.Vector([100, 0.2, 3], dt=qd.f32), b=65537))

    @qd.kernel
    def foo() -> qd.f32:
        s = bar()
        return s.a + s.b.a[0] + s.b.a[1] + s.b.a[2] + s.b.b

    assert foo() == pytest.approx(105.2)


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_break_in_real_func():
    @qd.real_func
    def bar() -> int:
        a = 0
        for i in range(10):
            if i == 5:
                break
            a += 1
        return a

    @qd.kernel
    def foo() -> int:
        return bar()

    assert foo() == 5


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_continue_in_real_func():
    @qd.real_func
    def bar() -> int:
        a = 0
        for i in range(10):
            if i % 2 == 0:
                continue
            a += 1
        return a

    @qd.kernel
    def foo() -> int:
        return bar()

    assert foo() == 5
