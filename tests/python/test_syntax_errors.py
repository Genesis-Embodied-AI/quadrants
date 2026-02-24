import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_try():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        try:
            a = 0
        except:
            a = 1

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_for_else():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        for i in range(10):
            pass
        else:
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_while_else():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        while True:
            pass
        else:
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_raise():
    @qd.kernel
    def foo():
        raise Exception()

    with pytest.raises(qd.QuadrantsSyntaxError, match='Unsupported node "Raise"') as e:
        foo()


@test_utils.test()
def test_loop_var_range():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        i = 0
        for i in range(10):
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_loop_var_struct():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        i = 0
        for i in x:
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_loop_var_struct():
    x = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def func():
        j = 0
        for i, j in x:
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_func_def_in_kernel():
    @qd.kernel
    def kernel():
        @qd.func
        def func():
            return 1

        print(func())

    with pytest.raises(qd.QuadrantsCompilationError):
        kernel()


@test_utils.test()
def test_func_def_in_func():
    @qd.func
    def func():
        @qd.func
        def func2():
            return 1

        return func2()

    @qd.kernel
    def kernel():
        print(func())

    with pytest.raises(qd.QuadrantsCompilationError):
        kernel()


@test_utils.test(arch=qd.cpu)
def test_kernel_bad_argument_annotation():
    with pytest.raises(qd.QuadrantsSyntaxError, match="annotation"):

        @qd.kernel
        def kernel(x: "bar"):
            print(x)


@test_utils.test(arch=qd.cpu)
def test_func_bad_argument_annotation():
    with pytest.raises(qd.QuadrantsSyntaxError, match="annotation"):

        @qd.func
        def func(x: "foo"):
            print(x)


@test_utils.test()
def test_nested_static():
    @qd.kernel
    def func():
        for i in qd.static(qd.static(range(1))):
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_nested_grouped():
    @qd.kernel
    def func():
        for i in qd.grouped(qd.grouped(range(1))):
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_nested_ndrange():
    @qd.kernel
    def func():
        for i in qd.ndrange(qd.ndrange(1)):
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_static_grouped_struct_for():
    val = qd.field(qd.i32)

    qd.root.dense(qd.ij, (1, 1)).place(val)

    @qd.kernel
    def test():
        for I in qd.static(qd.grouped(val)):
            pass

    with pytest.raises(qd.QuadrantsCompilationError):
        test()


@test_utils.test()
def test_is():
    b = qd.field(qd.i32, shape=())
    c = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = b is c

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_is_not():
    b = qd.field(qd.i32, shape=())
    c = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = b is not c

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_in():
    b = qd.field(qd.i32, shape=())
    c = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = b in c

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_not_in():
    b = qd.field(qd.i32, shape=())
    c = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = b not in c

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_expr_set():
    @qd.kernel
    def func():
        x = {2, 4, 6}

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test()
def test_redefining_template_args():
    @qd.kernel
    def foo(a: qd.template()):
        a = 5

    with pytest.raises(qd.QuadrantsSyntaxError, match='Kernel argument "a" is immutable in the kernel'):
        foo(1)


@test_utils.test(print_full_traceback=False)
def test_break_in_outermost_for():
    @qd.kernel
    def foo():
        for i in range(10):
            break

    with pytest.raises(qd.QuadrantsSyntaxError, match="Cannot break in the outermost loop"):
        foo()


@test_utils.test()
def test_funcdef_in_kernel():
    @qd.kernel
    def foo():
        def bar():
            pass

    with pytest.raises(qd.QuadrantsSyntaxError, match="Function definition is not allowed in 'qd.kernel'"):
        foo()


@test_utils.test()
def test_funcdef_in_func():
    @qd.func
    def foo():
        def bar():
            pass

    @qd.kernel
    def baz():
        foo()

    with pytest.raises(qd.QuadrantsSyntaxError, match="Function definition is not allowed in 'qd.func'"):
        baz()


@test_utils.test()
def test_continue_in_static_for_in_non_static_if():
    @qd.kernel
    def test_static_loop():
        for i in qd.static(range(5)):
            x = 0.1
            if x == 0.0:
                continue

    with pytest.raises(qd.QuadrantsSyntaxError, match="You are trying to `continue` a static `for` loop"):
        test_static_loop()


@test_utils.test()
def test_break_in_static_for_in_non_static_if():
    @qd.kernel
    def test_static_loop():
        for i in qd.static(range(5)):
            x = 0.1
            if x == 0.0:
                break

    with pytest.raises(qd.QuadrantsSyntaxError, match="You are trying to `break` a static `for` loop"):
        test_static_loop()
