import enum
import math

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_pure_validation_prim():
    a = 2

    @qd.kernel
    def k1():
        print(a)

    k1()

    @qd.pure
    @qd.kernel
    def k1b(a: qd.i32):
        print(a)

    k1b(a)

    @qd.pure
    @qd.kernel
    def k1c(a: qd.Template):
        print(a)

    k1c(a)

    @qd.pure
    @qd.kernel
    def k2():
        print(a)

    with pytest.raises(qd.QuadrantsCompilationError):
        k2()


@test_utils.test()
def test_pure_validation_field():
    a = qd.field(qd.i32, (10,))

    @qd.kernel
    def k1_f():
        print(a[0])

    k1_f()

    @qd.pure
    @qd.kernel
    def k1c_f(a: qd.Template):
        print(a[0])

    k1c_f(a)

    @qd.pure
    @qd.kernel
    def k2_f():
        print(a[0])

    with pytest.raises(qd.QuadrantsCompilationError):
        k2_f()


@test_utils.test()
def test_pure_validation_field_child():
    a = qd.field(qd.i32, (10,))

    @qd.func
    def k1_f():
        print(a[0])

    @qd.kernel
    def k1():
        k1_f()

    k1()

    @qd.func
    def k1c_f(a: qd.Template):
        print(a[0])

    @qd.pure
    @qd.kernel
    def k1c(a: qd.Template):
        k1c_f(a)

    k1c(a)

    @qd.func
    def k2_f():
        print(a[0])

    @qd.pure
    @qd.kernel
    def k2():
        k2_f()

    with pytest.raises(qd.QuadrantsCompilationError):
        k2()


@test_utils.test()
def test_pure_validation_data_oriented_not_param():
    # When the data oriented arrives into the kernel without being passed in as a parameter,
    # to the kernel, that's a pure violation
    @qd.data_oriented
    class MyDataOriented:
        def __init__(self) -> None:
            self.b = qd.field(qd.i32, (10,))

    @qd.pure
    @qd.kernel
    def k1() -> None:
        my_do.b[0] = 5

    my_do = MyDataOriented()
    with pytest.raises(qd.QuadrantsCompilationError):
        k1()


@test_utils.test()
def test_pure_validation_data_oriented_as_param():
    # When the data oriented arrives into the kernel as a parameter,
    # to the kernel, that's ok
    @qd.data_oriented
    class MyDataOriented:
        def __init__(self) -> None:
            self.b = qd.field(qd.i32, (10,))

    @qd.pure
    @qd.kernel
    def k1(my_data_oriented: qd.template()) -> None:
        my_data_oriented.b[0] = 5

    my_do = MyDataOriented()
    k1(my_do)


@test_utils.test()
def test_pure_validation_enum():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @qd.kernel(pure=True)
    def k1() -> qd.i32:
        return MyEnum.bar

    v = k1()
    assert v == 2


@test_utils.test()
def test_pure_validation_builtin_values_inf():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @qd.kernel(pure=True)
    def k1() -> qd.f32:
        return qd.math.inf

    v = k1()
    assert v > 1e8


@test_utils.test()
def test_pure_validation_builtin_values_ti_pi():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @qd.kernel(pure=True)
    def k1() -> qd.f32:
        return qd.math.pi

    v = k1()
    assert int(k1() * 100) == 314


@test_utils.test()
def test_pure_validation_builtin_values_np_pi():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @qd.kernel(pure=True)
    def k1() -> qd.f32:
        return np.pi

    v = k1()
    assert int(k1() * 100) == 314


@test_utils.test()
def test_pure_validation_builtin_values_math_pi():
    class MyEnum(enum.IntEnum):
        foo = 1
        bar = 2

    @qd.kernel(pure=True)
    def k1() -> qd.f32:
        return math.pi

    assert int(k1() * 100) == 314


@test_utils.test()
def test_pure_validation_actual_violation_exceptoin():
    a = 5

    @qd.kernel(pure=True)
    def k1() -> qd.f32:
        return a

    with pytest.raises(qd.QuadrantsCompilationError):
        k1()


@test_utils.test()
def test_pure_validation_actual_violation_warning():
    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch
    qd.init(arch=arch, offline_cache=False)

    SOME_GLOBAL = 5

    @qd.kernel(pure=True)
    def k1() -> qd.f32:
        return SOME_GLOBAL

    with pytest.warns(UserWarning, match=r"\[PURE\.VIOLATION\]"):
        assert k1() == 5

    class Foo:
        def __init__(self) -> None:
            self.BAR = 23

    foo = Foo()

    @qd.kernel(pure=True)
    def k2() -> qd.f32:
        return foo.BAR

    with pytest.warns(UserWarning, match=r"\[PURE\.VIOLATION\]"):
        assert k2() == 23
