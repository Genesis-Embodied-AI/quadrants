import pytest

import quadrants as qd

from tests.helpers import test_utils


@qd.func
def some_sub_func(a: qd.Template, b: qd.Template) -> None:
    a[()] = b[()] + 2


@qd.kernel
def some_kernel(a: qd.Template, b: qd.Template) -> None:
    a[()] = b[()] + 2
    some_sub_func(a, b)


@test_utils.test()
def test_template_no_braces():
    """
    Check that qd.Template annotations work in kernels and funcs.
    """
    a = qd.field(int, shape=())
    b = qd.field(int, shape=())
    b[()] = 5
    some_kernel(a, b)
    assert a[()] == 5 + 2


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
            self.scratch = qd.ndarray(qd.i32, shape=(1,))

    @qd.data_oriented
    class DataOrientedWithFloat:
        def __init__(self) -> None:
            self.an_int = 123
            self.a_float = 1.23
            self.scratch = qd.ndarray(qd.i32, shape=(1,))

    # The kernels have to actually read the primitive members: pruning keeps unread members out of the hash, so the
    # raise-on-templated-floats guard never sees ``a.a_float`` unless it is read.
    @qd.kernel(fastcache=True)
    def k1f(a: qd.Template) -> None:
        a.scratch[0] = qd.cast(a.a_float, qd.i32)

    @qd.kernel(fastcache=True)
    def k1i(a: qd.Template) -> None:
        a.scratch[0] = a.an_int

    my_do1 = DataOrientedWithoutFloat()
    k1i(my_do1)
    my_do2 = DataOrientedWithFloat()
    if raise_on_templated_floats:
        with pytest.raises(ValueError):
            k1f(my_do2)
    else:
        k1f(my_do2)


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


@test_utils.test()
def test_pyfunc_template_scalars_in_static():
    """Check that a pyfunc called from a kernel can use boolean and integer qd.Template arguments in qd.static."""
    values = qd.field(qd.i32, shape=4)

    @qd.pyfunc
    def add(enabled: qd.Template, count: qd.Template):
        if qd.static(enabled):
            for i in qd.static(range(count)):
                values[i] += 1

    @qd.kernel
    def run(enabled: qd.Template, count: qd.Template):
        add(enabled, count)

    for enabled, count in [(True, 4), (False, 4), (True, 2), (False, 2), (True, 4)]:
        values.fill(0)
        run(enabled, count)
        assert values.to_numpy().tolist() == [int(enabled and i < count) for i in range(4)]


@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@pytest.mark.parametrize("config", [[2, 3], (2, 3), {0: 2, 1: 3}], ids=["list", "tuple", "dict"])
@test_utils.test()
def test_template_compile_time_container(decorate, config):
    """Check that funcs and pyfuncs can use elements of a list, tuple, or dict passed as qd.Template in qd.static."""

    @decorate
    def total(items: qd.Template):
        result = 0
        for i in qd.static(range(items[0])):
            result += qd.static(items[1])
        return result

    @qd.kernel
    def run() -> qd.i32:
        return total(config)

    assert run() == 6


@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@test_utils.test()
def test_template_callable(decorate):
    """Check that funcs and pyfuncs can call a function passed as a qd.Template argument."""

    @qd.func
    def increment(value):
        return value + 1

    @decorate
    def apply(operation: qd.Template, value: qd.i32):
        return operation(value)

    @qd.kernel
    def run(value: qd.i32) -> qd.i32:
        return apply(increment, value)

    assert run(41) == 42


@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@test_utils.test()
def test_template_reference_mutation(decorate):
    """Check that changing a vector or struct passed as qd.Template changes the caller's value."""

    @decorate
    def update(vector: qd.Template, struct: qd.Template):
        vector[0] += 2
        struct.value += 3

    @qd.kernel
    def run() -> qd.i32:
        vector = qd.Vector([20, 30])
        struct = qd.Struct(value=40)
        update(vector, struct)
        return vector[0] + struct.value

    assert run() == 65
