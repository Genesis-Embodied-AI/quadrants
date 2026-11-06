import pytest

import quadrants as qd

from tests.helpers import test_utils


@test_utils.test()
def test_legacy_template_kernel():
    """Keep legacy kernel annotations working for field references and compile-time flags."""
    values = qd.field(qd.i32, shape=())

    @qd.kernel
    def run(target: qd.template(), enabled: qd.template()):
        if qd.static(enabled):
            target[()] += 1

    for enabled, expected in [(True, 1), (False, 1), (True, 2)]:
        run(values, enabled)
        assert values[()] == expected


@pytest.mark.parametrize("decorate", [qd.func, qd.pyfunc], ids=["func", "pyfunc"])
@test_utils.test()
def test_legacy_template_func(decorate):
    """Preserve legacy compile-time flags and scalar references in funcs and kernel-side pyfuncs."""

    @decorate
    def add(value: qd.template(), enabled: qd.template()):
        if qd.static(enabled):
            value += 2

    @qd.kernel
    def run(enabled: qd.Template) -> qd.i32:
        value = 40
        add(value, enabled)
        return value

    assert run(True) == 42
    assert run(False) == 40
    assert run(True) == 42


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_legacy_template_real_func():
    """Check that real functions accept legacy qd.template() arguments mixed with ordinary typed arguments."""
    values = qd.field(qd.i32, shape=())

    @qd.real_func
    def add(target: qd.template(), enabled: qd.template(), value: qd.i32):
        if qd.static(enabled):
            target[()] += value

    @qd.kernel
    def run(enabled: qd.Template, value: qd.i32):
        add(values, enabled, value)

    for enabled, value, expected in [(True, 3, 3), (False, 7, 3), (True, 2, 5)]:
        run(enabled, value)
        assert values[()] == expected
