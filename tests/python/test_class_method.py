from typing import Any

import pytest

import quadrants as qd

from tests import test_utils


@pytest.fixture
def qd_type(use_ndarray: bool) -> Any:
    if use_ndarray:
        return qd.ndarray
    return qd.field


@pytest.fixture
def qd_annotation(use_ndarray: bool) -> Any:
    if use_ndarray:
        return qd.types.ndarray()
    return qd.template()


@pytest.mark.parametrize("use_ndarray", [False, True])
@test_utils.test()
def test_class_method(qd_type: Any, qd_annotation: Any) -> None:
    shape = (20,)

    @qd.data_oriented
    class InnerClass:
        def __init__(self):
            self.a = qd_type(qd.i32, shape)
            self.b = qd_type(qd.i32, shape)

    @qd.data_oriented
    class MyClass:
        def __init__(self):
            self.inner = InnerClass()
            self.c = qd_type(qd.i32, shape)
            self.d = qd_type(qd.i32, shape)
            self.e = qd_type(qd.i32, shape)

        @qd.func
        def f1(self_unused, c: qd_annotation, inner: qd.template()):
            c[0] = 4

        @qd.kernel
        def test(self, c: qd_annotation, inner: qd.template()):
            self.f1(c, inner=inner)

        def run(self):
            self.test(self.c, self.inner)

    obj = MyClass()
    obj.run()
