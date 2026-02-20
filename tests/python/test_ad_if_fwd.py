import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_ad_if_simple_fwd():
    x = qd.field(qd.f32, shape=())
    y = qd.field(qd.f32, shape=())
    qd.root.lazy_dual()

    @qd.kernel
    def func():
        if x[None] > 0.0:
            y[None] = x[None]

    x[None] = 1
    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0]):
        func()

    assert y.dual[None] == 1


@test_utils.test()
def test_ad_if():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)
    qd.root.lazy_dual()

    @qd.kernel
    def func(i: qd.i32):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 2 * x[i]

    x[0] = 0
    x[1] = 1
    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0, 1.0]):
        func(0)
        func(1)
    assert y.dual[0] == 2
    assert y.dual[1] == 1


@test_utils.test()
def test_ad_if_nested():
    n = 20
    x = qd.field(qd.f32, shape=n)
    y = qd.field(qd.f32, shape=n)
    z = qd.field(qd.f32, shape=n)
    qd.root.lazy_dual()

    @qd.kernel
    def func():
        for i in x:
            if x[i] < 2:
                if x[i] == 0:
                    y[i] = 0
                else:
                    y[i] = z[i] * 1
            else:
                if x[i] == 2:
                    y[i] = z[i] * 2
                else:
                    y[i] = z[i] * 3

    z.fill(1)

    for i in range(n):
        x[i] = i % 4

    func()
    for i in range(n):
        assert y[i] == i % 4

    with qd.ad.FwdMode(loss=y, param=z, seed=[1.0 for _ in range(n)]):
        func()

    for i in range(n):
        assert y.dual[i] == i % 4


@test_utils.test()
def test_ad_if_mutable():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)

    qd.root.lazy_dual()

    @qd.kernel
    def func(i: qd.i32):
        t = x[i]
        if t > 0:
            y[i] = t
        else:
            y[i] = 2 * t

    x[0] = 0
    x[1] = 1

    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0, 1.0]):
        func(0)
        func(1)

    assert y.dual[0] == 2
    assert y.dual[1] == 1


@test_utils.test()
def test_ad_if_parallel():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)

    qd.root.lazy_dual()

    @qd.kernel
    def func():
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1

    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0, 1.0]):
        func()

    assert y.dual[0] == 2
    assert y.dual[1] == 1


@test_utils.test(require=[qd.extension.data64], default_fp=qd.f64)
def test_ad_if_parallel_f64():
    x = qd.field(qd.f64, shape=2)
    y = qd.field(qd.f64, shape=2)

    qd.root.lazy_dual()

    @qd.kernel
    def func():
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1

    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0, 1.0]):
        func()

    assert y.dual[0] == 2
    assert y.dual[1] == 1


@test_utils.test()
def test_ad_if_parallel_complex():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)

    qd.root.lazy_dual()

    @qd.kernel
    def func():
        qd.loop_config(parallelize=1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2

    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0, 1.0]):
        func()

    assert y.dual[0] == 0
    assert y.dual[1] == -0.25


@test_utils.test(require=[qd.extension.data64], default_fp=qd.f64)
def test_ad_if_parallel_complex_f64():
    x = qd.field(qd.f64, shape=2)
    y = qd.field(qd.f64, shape=2)

    qd.root.lazy_dual()

    @qd.kernel
    def func():
        qd.loop_config(parallelize=1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2

    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0, 1.0]):
        func()

    assert y.dual[0] == 0
    assert y.dual[1] == -0.25
