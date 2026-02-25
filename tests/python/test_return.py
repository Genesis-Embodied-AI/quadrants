import sys
from typing import Tuple

import pytest
from pytest import approx

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_return_without_type_hint():
    @qd.kernel
    def kernel():
        return 1

    with pytest.raises(qd.QuadrantsCompilationError):
        kernel()


def test_const_func_ret():
    qd.init()

    @qd.kernel
    def func1() -> qd.f32:
        return 3

    @qd.kernel
    def func2() -> qd.i32:
        return 3.3  # return type mismatch, will be auto-casted into qd.i32

    assert func1() == test_utils.approx(3)
    assert func2() == 3


@pytest.mark.parametrize(
    "dt1,dt2,dt3,castor",
    [
        (qd.i32, qd.f32, qd.f32, float),
        (qd.f32, qd.i32, qd.f32, float),
        (qd.i32, qd.f32, qd.i32, int),
        (qd.f32, qd.i32, qd.i32, int),
    ],
)
@test_utils.test()
def test_binary_func_ret(dt1, dt2, dt3, castor):
    @qd.kernel
    def func(a: dt1, b: dt2) -> dt3:
        return a * b

    if qd.types.is_integral(dt1):
        xs = list(range(4))
    else:
        xs = [0.2, 0.4, 0.8, 1.0]

    if qd.types.is_integral(dt2):
        ys = list(range(4))
    else:
        ys = [0.2, 0.4, 0.8, 1.0]

    for x, y in zip(xs, ys):
        assert func(x, y) == test_utils.approx(castor(x * y))


@test_utils.test()
def test_return_in_static_if():
    @qd.kernel
    def foo(a: qd.template()) -> qd.i32:
        if qd.static(a == 1):
            return 1
        elif qd.static(a == 2):
            return 2
        return 3

    assert foo(1) == 1
    assert foo(2) == 2
    assert foo(123) == 3


@test_utils.test()
def test_func_multiple_return():
    @qd.func
    def safe_sqrt(a):
        if a > 0:
            return qd.sqrt(a)
        else:
            return 0.0

    @qd.kernel
    def kern(a: float):
        print(safe_sqrt(a))

    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Return inside non-static if/for is not supported",
    ):
        kern(-233)


@test_utils.test()
def test_return_inside_static_for():
    @qd.kernel
    def foo() -> qd.i32:
        a = 0
        for i in qd.static(range(10)):
            a += i * i
            if qd.static(i == 8):
                return a

    assert foo() == 204


@test_utils.test()
def test_return_inside_non_static_for():
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Return inside non-static if/for is not supported",
    ):

        @qd.kernel
        def foo() -> qd.i32:
            for i in range(10):
                return i

        foo()


@test_utils.test()
def test_kernel_no_return():
    with pytest.raises(
        qd.QuadrantsSyntaxError,
        match="Kernel has a return type but does not have a return statement",
    ):

        @qd.kernel
        def foo() -> qd.i32:
            pass

        foo()


@test_utils.test()
def test_func_no_return():
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Function has a return type but does not have a return statement",
    ):

        @qd.func
        def bar() -> qd.i32:
            pass

        @qd.kernel
        def foo() -> qd.i32:
            return bar()

        foo()


@test_utils.test()
def test_void_return():
    @qd.kernel
    def foo():
        return

    foo()


@test_utils.test()
def test_return_none():
    @qd.kernel
    def foo():
        return None

    foo()


@test_utils.test(exclude=[qd.metal, qd.vulkan])
def test_return_uint64():
    @qd.kernel
    def foo() -> qd.u64:
        return qd.u64(2**64 - 1)

    assert foo() == 2**64 - 1


@test_utils.test(exclude=[qd.metal, qd.vulkan])
def test_return_uint64_vec():
    @qd.kernel
    def foo() -> qd.types.vector(2, qd.u64):
        return qd.Vector([qd.u64(2**64 - 1), qd.u64(2**64 - 1)])

    assert foo()[0] == 2**64 - 1


@test_utils.test()
def test_struct_ret_with_matrix():
    s0 = qd.types.struct(a=qd.math.vec3, b=qd.i16)
    s1 = qd.types.struct(a=qd.f32, b=s0)

    @qd.kernel
    def foo() -> s1:
        return s1(a=1, b=s0(a=qd.math.vec3([100, 0.2, 3]), b=65537))

    ret = foo()
    assert ret.a == approx(1)
    assert ret.b.a[0] == approx(100)
    assert ret.b.a[1] == approx(0.2)
    assert ret.b.a[2] == approx(3)
    assert ret.b.b == 1


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_tuple_ret_39():
    s0 = qd.types.struct(a=qd.math.vec3, b=qd.i16)

    @qd.real_func
    def foo() -> tuple[qd.f32, s0]:
        return 1, s0(a=qd.math.vec3([100, 0.2, 3]), b=65537)

    @qd.kernel
    def bar() -> tuple[qd.f32, s0]:
        return foo()

    ret_a, ret_b = bar()
    assert ret_a == approx(1)
    assert ret_b.a[0] == approx(100)
    assert ret_b.a[1] == approx(0.2)
    assert ret_b.a[2] == approx(3)
    assert ret_b.b == 1


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_tuple_ret_typing_tuple():
    s0 = qd.types.struct(a=qd.math.vec3, b=qd.i16)

    @qd.real_func
    def foo() -> Tuple[qd.f32, s0]:
        return 1, s0(a=qd.math.vec3([100, 0.2, 3]), b=65537)

    @qd.kernel
    def bar() -> Tuple[qd.f32, s0]:
        return foo()

    ret_a, ret_b = bar()
    assert ret_a == approx(1)
    assert ret_b.a[0] == approx(100)
    assert ret_b.a[1] == approx(0.2)
    assert ret_b.a[2] == approx(3)
    assert ret_b.b == 1


@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True)
def test_real_func_tuple_ret():
    s0 = qd.types.struct(a=qd.math.vec3, b=qd.i16)

    @qd.real_func
    def foo() -> (qd.f32, s0):
        return 1, s0(a=qd.math.vec3([100, 0.2, 3]), b=65537)

    @qd.kernel
    def bar() -> (qd.f32, s0):
        return foo()

    # bar()
    ret_a, ret_b = bar()
    assert ret_a == approx(1)
    assert ret_b.a[0] == approx(100)
    assert ret_b.a[1] == approx(0.2)
    assert ret_b.a[2] == approx(3)
    assert ret_b.b == 1


@test_utils.test(print_full_traceback=False)
def test_return_type_mismatch_1():
    with pytest.raises(qd.QuadrantsCompilationError):

        @qd.kernel
        def foo() -> qd.i32:
            return qd.math.vec3([1, 2, 3])

        foo()


@test_utils.test(print_full_traceback=False)
def test_return_type_mismatch_2():
    with pytest.raises(qd.QuadrantsCompilationError):

        @qd.kernel
        def foo() -> qd.math.vec4:
            return qd.math.vec3([1, 2, 3])

        foo()


@test_utils.test(print_full_traceback=False)
def test_return_type_mismatch_3():
    sphere_type = qd.types.struct(center=qd.math.vec3, radius=float)
    circle_type = qd.types.struct(center=qd.math.vec2, radius=float)
    sphere_type_ = qd.types.struct(center=qd.math.vec3, radius=int)

    @qd.kernel
    def foo() -> sphere_type:
        return circle_type(center=qd.math.vec2([1, 2]), radius=2)

    @qd.kernel
    def bar() -> sphere_type:
        return sphere_type_(center=qd.math.vec3([1, 2, 3]), radius=2)

    with pytest.raises(qd.QuadrantsCompilationError):
        foo()

    with pytest.raises(qd.QuadrantsCompilationError):
        bar()


@test_utils.test()
def test_func_scalar_return_cast():
    @qd.func
    def bar(a: qd.f32) -> qd.i32:
        return a

    @qd.kernel
    def foo(a: qd.f32) -> qd.f32:
        return bar(a)

    assert foo(1.5) == 1.0


@test_utils.test()
def test_return_struct_field():
    tp = qd.types.struct(a=qd.i32)

    f = tp.field(shape=1)

    @qd.func
    def bar() -> tp:
        return f[0]

    @qd.kernel
    def foo() -> tp:
        return bar()

    assert foo().a == 0


@test_utils.test(exclude=[qd.amdgpu])
def test_ret_4k():
    vec1024 = qd.types.vector(1024, qd.i32)

    @qd.kernel
    def foo() -> vec1024:
        ret = vec1024(0)
        for i in range(1024):
            ret[i] = i
        return ret

    ret = foo()
    for i in range(1024):
        assert ret[i] == i
