import pytest

import quadrants as qd


@pytest.mark.tryfirst
def test_without_init():
    # We want to check if Quadrants works well without ``qd.init()``.
    # But in test ``qd.init()`` will always be called in last ``@qd.all_archs``.
    # So we have to create a new Quadrants instance, i.e. test in a sandbox.
    assert qd.cfg.arch == qd.cpu

    x = qd.field(qd.i32, (2, 3))
    assert x.shape == (2, 3)

    x[1, 2] = 4
    assert x[1, 2] == 4
