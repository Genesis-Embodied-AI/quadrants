import pytest

import quadrants as ti

from tests import test_utils


@test_utils.test()
def test_remove_is_is_not():
    with pytest.raises(ti.QuadrantsSyntaxError, match='Operator "is" in Quadrants scope is not supported'):

        @ti.kernel
        def func():
            ti.static(1 is 2)

        func()


@test_utils.test(arch=[ti.cpu, ti.cuda])
def test_deprecate_experimental_real_func():
    with pytest.warns(
        DeprecationWarning,
        match="ti.experimental.real_func is deprecated because it is no longer experimental. "
        "Use ti.real_func instead.",
    ):

        @ti.experimental.real_func
        def foo(a: ti.i32) -> ti.i32:
            s = 0
            for i in range(100):
                if i == a + 1:
                    return s
                s = s + i
            return s

        @ti.kernel
        def bar(a: ti.i32) -> ti.i32:
            return foo(a)

        assert bar(10) == 11 * 5
        assert bar(200) == 99 * 50
