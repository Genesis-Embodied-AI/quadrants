import platform

import pytest

import quadrants as qd

from tests import test_utils

u = platform.uname()
if u.system == "linux" and u.machine in ("arm64", "aarch64"):
    pytest.skip(
        "This module is only for linux on arm64, which doesn't support assert currently", allow_module_level=True
    )


@test_utils.test()
def test_assert_ignored():
    """
    On linux arm, assert is just a `nop` currently (otherwise it crashes). This test checks that:
    - assert doesn't cause a crash
    - that the test expression is still executed
    """
    a = qd.field(qd.i32, shape=())

    @qd.func
    def f1() -> bool:
        a[()] = 5
        return False

    @qd.kernel
    def k1() -> None:
        assert f1()
        assert False
        assert True

    k1()
    assert a[()] == 5
