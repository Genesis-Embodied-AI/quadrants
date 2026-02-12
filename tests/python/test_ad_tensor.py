import functools

import quadrants as ti

from tests import test_utils

has_autograd = False

try:
    import autograd.numpy as np  # noqa: F401
    from autograd import grad  # noqa: F401

    has_autograd = True
except:
    pass


def if_has_autograd(func):
    # functools.wraps is nececssary for pytest parametrization to work
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if has_autograd:
            func(*args, **kwargs)

    return wrapper


@if_has_autograd
@test_utils.test()
def test_ad_tensor_store_load():
    x = ti.Vector.field(4, dtype=ti.f32, shape=(), needs_grad=True)
    y = ti.Vector.field(4, dtype=ti.f32, shape=(), needs_grad=True)

    @ti.kernel
    def test(tmp: ti.f32):
        b = ti.Vector([tmp, tmp, tmp, tmp])
        b[0] = tmp * 4
        y[None] = b * x[None]

    y.grad.fill(2.0)
    test.grad(10)

    assert (x.grad.to_numpy() == [80.0, 20.0, 20.0, 20.0]).all()
