from typing import Any

import pytest

import quadrants as ti

from tests import test_utils


@pytest.fixture
def ti_type(use_ndarray: bool) -> Any:
    if use_ndarray:
        return ti.ndarray
    return ti.field


@pytest.fixture
def ti_annotation(use_ndarray: bool) -> Any:
    if use_ndarray:
        return ti.types.ndarray()
    return ti.template()


@pytest.mark.parametrize("use_ndarray", [False, True])
@test_utils.test()
def test_class_method(ti_type: Any, ti_annotation: Any) -> None:
    shape = (20,)

    @ti.data_oriented
    class InnerClass:
        def __init__(self):
            self.a = ti_type(ti.i32, shape)
            self.b = ti_type(ti.i32, shape)

    @ti.data_oriented
    class MyClass:
        def __init__(self):
            self.inner = InnerClass()
            self.c = ti_type(ti.i32, shape)
            self.d = ti_type(ti.i32, shape)
            self.e = ti_type(ti.i32, shape)

        @ti.func
        def f1(self_unused, c: ti_annotation, inner: ti.template()):
            c[0] = 4

        @ti.kernel
        def test(self, c: ti_annotation, inner: ti.template()):
            self.f1(c, inner=inner)

        def run(self):
            self.test(self.c, self.inner)

    obj = MyClass()
    obj.run()
