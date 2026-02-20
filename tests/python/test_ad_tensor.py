import functools

import quadrants as qd

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
    x = qd.Vector.field(4, dtype=qd.f32, shape=(), needs_grad=True)
    y = qd.Vector.field(4, dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def test(tmp: qd.f32):
        b = qd.Vector([tmp, tmp, tmp, tmp])
        b[0] = tmp * 4
        y[None] = b * x[None]

    y.grad.fill(2.0)
    test.grad(10)

    assert (x.grad.to_numpy() == [80.0, 20.0, 20.0, 20.0]).all()
