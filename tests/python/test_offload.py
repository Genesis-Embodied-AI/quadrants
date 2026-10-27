import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_running_loss():
    return
    steps = 16

    total_loss = qd.field(qd.f32)
    running_loss = qd.field(qd.f32)
    additional_loss = qd.field(qd.f32)

    qd.root.place(total_loss)
    qd.root.dense(qd.i, steps).place(running_loss)
    qd.root.place(additional_loss)
    qd.root.lazy_grad()

    @qd.kernel
    def compute_loss():
        total_loss[()] = 0.0
        for i in range(steps):
            qd.atomic_add(total_loss[()], running_loss[i] * 2)
        qd.atomic_add(total_loss[()], additional_loss[()] * 3)

    compute_loss()

    assert total_loss.grad[()] == 1
    for i in range(steps):
        assert running_loss[i] == 2
    assert additional_loss.grad[()] == 3


@test_utils.test()
def test_reduce_separate():
    a = qd.field(qd.f32, shape=(16))
    b = qd.field(qd.f32, shape=(4))
    c = qd.field(qd.f32, shape=())

    qd.root.lazy_grad()

    @qd.kernel
    def reduce1():
        for i in range(16):
            b[i // 4] += a[i]

    @qd.kernel
    def reduce2():
        for i in range(4):
            c[()] += b[i]

    c.grad[()] = 1
    reduce2.grad()
    reduce1.grad()

    for i in range(4):
        assert b.grad[i] == 1
    for i in range(16):
        assert a.grad[i] == 1


@test_utils.test()
def test_reduce_merged():
    a = qd.field(qd.f32, shape=(16))
    b = qd.field(qd.f32, shape=(4))
    c = qd.field(qd.f32, shape=())

    qd.root.lazy_grad()

    @qd.kernel
    def reduce():
        for i in range(16):
            b[i // 4] += a[i]

        for i in range(4):
            c[()] += b[i]

    c.grad[()] = 1
    reduce.grad()

    for i in range(4):
        assert b.grad[i] == 1
    for i in range(16):
        assert a.grad[i] == 1
