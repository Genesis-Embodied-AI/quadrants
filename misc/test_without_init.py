import pytest

import quadrants as ti


@pytest.mark.tryfirst
def test_without_init():
    # We want to check if Quadrants works well without ``ti.init()``.
    # But in test ``ti.init()`` will always be called in last ``@ti.all_archs``.
    # So we have to create a new Quadrants instance, i.e. test in a sandbox.
    assert ti.cfg.arch == ti.cpu

    x = ti.field(ti.i32, (2, 3))
    assert x.shape == (2, 3)

    x[1, 2] = 4
    assert x[1, 2] == 4
