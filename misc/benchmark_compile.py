import autograd.numpy as np
from autograd import grad
from pytest import approx

import quadrants as qd


@qd.test()
def grad_test(tifunc, npfunc=None):
    if npfunc is None:
        npfunc = tifunc

    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x, x.grad, y, y.grad)

    @qd.kernel
    def func():
        for i in x:
            y[i] = tifunc(x[i])

    v = 0.2

    y.grad[0] = 1
    x[0] = v
    func()
    func.grad()

    assert y[0] == approx(npfunc(v))
    assert x.grad[0] == approx(grad(npfunc)(v))


def test_unary():
    import time

    t = time.time()
    grad_test(lambda x: qd.sqrt(x), lambda x: np.sqrt(x))
    grad_test(lambda x: qd.exp(x), lambda x: np.exp(x))
    grad_test(lambda x: qd.log(x), lambda x: np.log(x))
    qd.core.print_profile_info()
    print("Total time {:.3f}s".format(time.time() - t))


test_unary()
