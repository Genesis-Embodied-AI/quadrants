import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_abs():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    N = 16

    qd.root.dense(qd.i, N).place(x)
    qd.root.dense(qd.i, N).place(y)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in range(N):
            x[i] = abs(y[i])

    for i in range(N):
        y[i] = i - 10
        x.grad[i] = 1

    func()
    func.grad()

    def sgn(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    for i in range(N):
        assert x[i] == abs(y[i])
        assert y.grad[i] == sgn(y[i])


@test_utils.test()
def test_abs_fwd():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    N = 16

    qd.root.dense(qd.i, N).place(x)
    qd.root.dense(qd.i, N).place(y)
    qd.root.lazy_dual()

    @qd.kernel
    def func():
        for i in range(N):
            x[i] = abs(y[i])

    for i in range(N):
        y[i] = i - 10

    with qd.ad.FwdMode(loss=x, param=y, seed=[1.0 for _ in range(N)]):
        func()

    def sgn(x):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    for i in range(N):
        assert x[i] == abs(y[i])
        assert x.dual[i] == sgn(y[i])


@test_utils.test(require=qd.extension.data64)
def test_abs_i64():
    @qd.kernel
    def foo(x: qd.i64) -> qd.i64:
        return abs(x)

    for x in [-(2**40), 0, 2**40]:
        assert foo(x) == abs(x)


@test_utils.test()
def test_abs_u32():
    @qd.kernel
    def foo(x: qd.u32) -> qd.u32:
        return abs(x)

    for x in [0, 2**20]:
        assert foo(x) == abs(x)
