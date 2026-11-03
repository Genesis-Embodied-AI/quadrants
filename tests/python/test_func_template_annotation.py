import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("annotation", [qd.Template, qd.template()], ids=["class", "legacy"])
@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@test_utils.test()
def test_template_compile_time_scalars(annotation, decorate):
    """Keep boolean flags and integer loop bounds compile-time across changing and repeated kernel calls."""
    values = qd.field(qd.i32, shape=4)

    @decorate
    def add(enabled: annotation, count: annotation):
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


@pytest.mark.parametrize("annotation", [qd.Template, qd.template()], ids=["class", "legacy"])
@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@pytest.mark.parametrize("config", [[2, 3], (2, 3), {0: 2, 1: 3}], ids=["list", "tuple", "dict"])
@test_utils.test()
def test_template_compile_time_container(annotation, decorate, config):
    """Preserve compile-time elements of lists, tuples, and dictionaries passed to funcs and kernel-side pyfuncs."""

    @decorate
    def total(items: annotation):
        result = 0
        for i in qd.static(range(items[0])):
            result += qd.static(items[1])
        return result

    @qd.kernel
    def run() -> qd.i32:
        return total(config)

    assert run() == 6


@pytest.mark.parametrize("annotation", [qd.Template, qd.template()], ids=["class", "legacy"])
@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@test_utils.test()
def test_template_callable(annotation, decorate):
    """Pass a callable through a template parameter and invoke it from a func or kernel-side pyfunc."""

    @qd.func
    def increment(value):
        return value + 1

    @decorate
    def apply(operation: annotation, value: qd.i32):
        return operation(value)

    @qd.kernel
    def run(value: qd.i32) -> qd.i32:
        return apply(increment, value)

    assert run(41) == 42


@pytest.mark.parametrize("annotation", [qd.Template, qd.template()], ids=["class", "legacy"])
@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@test_utils.test()
def test_template_reference_mutation(annotation, decorate):
    """Verify that mutations to templated runtime scalars, vectors, and structs reach the caller."""

    @decorate
    def increment(value: annotation):
        value += 1

    @decorate
    def update(vector: annotation, struct: annotation):
        vector[0] += 2
        struct.value += 3

    @qd.kernel
    def run() -> qd.i32:
        scalar = 10
        vector = qd.Vector([20, 30])
        struct = qd.Struct(value=40)
        increment(scalar)
        update(vector, struct)
        return scalar + vector[0] + struct.value

    assert run() == 76


@pytest.mark.parametrize("annotation", [qd.Template, qd.template()], ids=["class", "legacy"])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_real_func_template_runtime_arguments(annotation):
    """Keep interleaved template parameters out of real-function runtime arguments while specializing their values."""

    @qd.real_func
    def apply(enabled: annotation, value: qd.i32, offset: annotation, scale: qd.i32) -> qd.i32:
        result = value
        if qd.static(enabled):
            result = value * scale + offset
        return result

    @qd.kernel
    def run(enabled: qd.Template, offset: qd.Template, value: qd.i32) -> qd.i32:
        return apply(enabled, value, offset, 2)

    for enabled, offset, value in [(True, 3, 10), (False, 3, 10), (True, 7, 20), (True, 3, 11)]:
        assert run(enabled, offset, value) == (value * 2 + offset if enabled else value)


@pytest.mark.parametrize("annotation", [qd.Template, qd.template()], ids=["class", "legacy"])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_real_func_template_only_arguments(annotation):
    """Call a real function with no runtime arguments, specializing on a field and changing or repeated constants."""
    values = qd.field(qd.i32, shape=())

    @qd.real_func
    def add(target: annotation, amount: annotation):
        target[()] += amount

    @qd.kernel
    def run():
        add(values, 3)
        add(values, 7)
        add(values, 3)

    run()
    assert values[()] == 13
