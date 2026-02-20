import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_ad_fwd_add():
    N = 5
    x = qd.field(qd.f32, shape=N)
    loss = qd.field(qd.f32, shape=N)
    qd.root.lazy_dual()

    for i in range(N):
        x[i] = i

    @qd.kernel
    def ad_fwd_add():
        loss[1] += 2 * x[3]

    with qd.ad.FwdMode(loss=loss, param=x, seed=[0, 0, 0, 1, 0]):
        ad_fwd_add()

    assert loss.dual[1] == 2


@test_utils.test()
def test_ad_fwd_multiply():
    N = 5
    x = qd.field(qd.f32, shape=N)
    loss = qd.field(qd.f32, shape=N)
    qd.root.lazy_dual()

    for i in range(N):
        x[i] = i

    @qd.kernel
    def ad_fwd_multiply():
        loss[1] += x[3] * x[4]

    with qd.ad.FwdMode(loss=loss, param=x, seed=[0, 0, 0, 1, 1]):
        ad_fwd_multiply()

    assert loss.dual[1] == 7


@test_utils.test()
def test_multiple_calls():
    N = 5
    a = qd.field(float, shape=N)
    b = qd.field(float, shape=N)
    loss_1 = qd.field(float, shape=())
    loss_2 = qd.field(float, shape=())
    qd.root.lazy_dual()

    for i in range(N):
        a[i] = i
        b[i] = i

    @qd.kernel
    def multiple_calls():
        loss_1[None] += 3 * b[1] ** 2 + 5 * a[3] ** 2
        loss_2[None] += 4 * b[2] ** 2 + 6 * a[4] ** 2

    with qd.ad.FwdMode(loss=loss_1, param=a, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_1.dual[None] == 30

    with qd.ad.FwdMode(loss=loss_1, param=b, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_1.dual[None] == 6

    with qd.ad.FwdMode(loss=loss_2, param=b, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_2.dual[None] == 16

    with qd.ad.FwdMode(loss=loss_2, param=a, seed=[1.0 for _ in range(N)]):
        multiple_calls()
    assert loss_2.dual[None] == 48


@test_utils.test()
def test_handle_shape_accessed_by_zero():
    a = qd.field(float)
    b = qd.field(float)
    qd.root.dense(qd.i, 1).place(a, b, a.dual, b.dual)

    @qd.kernel
    def func():
        pass

    with qd.ad.FwdMode(loss=b, param=a):
        func()


@test_utils.test()
def test_handle_shape_accessed_by_none():
    c = qd.field(float, shape=())
    d = qd.field(float, shape=())
    qd.root.lazy_dual()

    @qd.kernel
    def func():
        pass

    with qd.ad.FwdMode(loss=d, param=c):
        func()


@test_utils.test()
def test_clear_all_dual_field():
    x = qd.field(float, shape=(), needs_dual=True)
    y = qd.field(float, shape=(), needs_dual=True)
    loss = qd.field(float, shape=(), needs_dual=True)

    x[None] = 2.0
    y[None] = 3.0

    @qd.kernel
    def clear_dual_test():
        y[None] = x[None] ** 2
        loss[None] += y[None]

    for _ in range(5):
        with qd.ad.FwdMode(loss=loss, param=x):
            clear_dual_test()
        assert y.dual[None] == 4.0
