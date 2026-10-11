import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_is_is_not_unsupported_in_quadrants_scope():
    with pytest.raises(qd.QuadrantsSyntaxError, match='Operator "is" in Quadrants scope is not supported'):

        @qd.kernel
        def func_is(a: qd.template()):
            a is None

        func_is(None)

    with pytest.raises(qd.QuadrantsSyntaxError, match='Operator "is not" in Quadrants scope is not supported'):

        @qd.kernel
        def func_is_not(a: qd.template()):
            a is not None

        func_is_not(None)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_deprecate_experimental_real_func():
    with pytest.warns(
        DeprecationWarning,
        match="qd.experimental.real_func is deprecated because it is no longer experimental. "
        "Use qd.real_func instead.",
    ):

        @qd.experimental.real_func
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
