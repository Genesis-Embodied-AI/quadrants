from autograd import grad

import quadrants as qd
from quadrants._testing import approx


# Note: test happens at v = 0.2
def grad_test(tifunc, npfunc=None, default_fp=qd.f32):
    if npfunc is None:
        npfunc = tifunc

    @qd.test(default_fp=default_fp)
    def impl():
        print(f"arch={qd.cfg.arch} default_fp={qd.cfg.default_fp}")
        x = qd.field(default_fp)
        y = qd.field(default_fp)

        qd.root.dense(qd.i, 1).place(x, x.grad, y, y.grad)

        @qd.kernel
        def func():
            for i in x:
                y[i] = tifunc(x[i])

        v = 0.234

        y.grad[0] = 1
        x[0] = v
        func()
        func.grad()

        assert y[0] == approx(npfunc(v))
        assert x.grad[0] == approx(grad(npfunc)(v))

    impl()


def test_poly():
    import time

    t = time.time()
    grad_test(lambda x: x)
    grad_test(lambda x: -x)
    grad_test(lambda x: x * x)
    grad_test(lambda x: x**2)
    grad_test(lambda x: x * x * x)
    grad_test(lambda x: x * x * x * x)
    grad_test(lambda x: 0.4 * x * x - 3)
    grad_test(lambda x: (x - 3) * (x - 1))
    grad_test(lambda x: (x - 3) * (x - 1) + x * x)
    qd.core.print_profile_info()
    print("total_time", time.time() - t)


test_poly()
