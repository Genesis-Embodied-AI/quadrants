import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_ad_reduce():
    N = 16

    x = qd.field(dtype=qd.f32, shape=N, needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def func():
        for i in x:
            loss[None] += x[i] ** 2

    total_loss = 0
    for i in range(N):
        x[i] = i
        total_loss += i * i

    loss.grad[None] = 1
    func()
    func.grad()

    assert total_loss == test_utils.approx(loss[None])
    for i in range(N):
        assert x.grad[i] == test_utils.approx(i * 2)
