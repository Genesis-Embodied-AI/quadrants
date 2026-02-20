import pytest

import quadrants as qd

from tests import test_utils


@qd.func
def some_sub_func(a: qd.template, b: qd.Template) -> None:
    a[None] = b[None] + 2


@qd.kernel
def some_kernel(a: qd.template, b: qd.Template) -> None:
    a[None] = b[None] + 2
    some_sub_func(a, b)


@test_utils.test()
def test_template_no_braces():
    """
    Check that we can use qd.Template as an annotation for kernels and funcs.
    """
    a = qd.field(int, shape=())
    b = qd.field(int, shape=())
    b[None] = 5
    some_kernel(a, b)
    assert a[None] == 5 + 2


@pytest.mark.parametrize("raise_on_templated_floats", [False, True])
@test_utils.test()
def test_template_raise_on_floats(raise_on_templated_floats: bool) -> None:
    arch = getattr(qd, qd.cfg.arch.name)
    qd.init(arch=arch, raise_on_templated_floats=raise_on_templated_floats)

    @qd.kernel
    def k1(a: qd.Template) -> None:
        print(a)

    k1(123)
    if raise_on_templated_floats:
        with pytest.raises(ValueError):
            k1(1.23)
    else:
        k1(1.23)


@pytest.mark.parametrize("raise_on_templated_floats", [False, True])
@test_utils.test()
def test_template_raise_on_data_oriented_floats(raise_on_templated_floats: bool) -> None:
    arch = getattr(qd, qd.cfg.arch.name)
    qd.init(arch=arch, raise_on_templated_floats=raise_on_templated_floats)

    @qd.data_oriented
    class DataOrientedWithoutFloat:
        def __init__(self) -> None:
            self.an_int = 123
            self.a_bool = True

    @qd.data_oriented
    class DataOrientedWithFloat:
        def __init__(self) -> None:
            self.an_int = 123
            self.a_float = 1.23

    @qd.kernel(fastcache=True)
    def k1(a: qd.Template) -> None: ...

    my_do1 = DataOrientedWithoutFloat()
    k1(my_do1)
    my_do2 = DataOrientedWithFloat()
    if raise_on_templated_floats:
        with pytest.raises(ValueError):
            k1(my_do2)
    else:
        k1(my_do2)


@pytest.mark.parametrize("raise_on_templated_floats", [False, True])
@test_utils.test()
def test_template_raise_on_global_floats(raise_on_templated_floats: bool) -> None:
    arch = getattr(qd, qd.cfg.arch.name)
    qd.init(arch=arch, raise_on_templated_floats=raise_on_templated_floats)

    c = 123
    d = 1.23

    @qd.kernel()
    def k1c() -> None:
        print(c)

    @qd.kernel()
    def k1d() -> None:
        print(d)

    k1c()
    if raise_on_templated_floats:
        with pytest.raises(qd.QuadrantsCompilationError):
            k1d()
    else:
        k1d()
