import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_remove_is_is_not():
    with pytest.raises(qd.QuadrantsSyntaxError, match='Operator "is" in Quadrants scope is not supported'):

        @qd.kernel
        def func():
            qd.static(1 is 2)

        func()


@test_utils.test(arch=[qd.cpu, qd.cuda])
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
