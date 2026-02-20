import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_normal_grad():
    x = qd.field(qd.f32)
    loss = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(loss)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in range(n):
            loss[None] += x[i] ** 2

    for i in range(n):
        x[i] = i

    with qd.ad.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == i * 2


@test_utils.test()
def test_stop_grad():
    x = qd.field(qd.f32)
    loss = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(loss)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in range(n):
            qd.stop_grad(x)
            loss[None] += x[i] ** 2

    for i in range(n):
        x[i] = i

    with qd.ad.Tape(loss):
        func()

    for i in range(n):
        assert x.grad[i] == 0


@test_utils.test()
def test_stop_grad2():
    x = qd.field(qd.f32)
    loss = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(loss)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        # Two loops, one with stop grad on without
        for i in range(n):
            qd.stop_grad(x)
            loss[None] += x[i] ** 2
        for i in range(n):
            loss[None] += x[i] ** 2

    for i in range(n):
        x[i] = i

    with qd.ad.Tape(loss):
        func()

    # If without stop, grad x.grad[i] = i * 4
    for i in range(n):
        assert x.grad[i] == i * 2
