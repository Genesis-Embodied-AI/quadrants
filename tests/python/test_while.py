import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_while():
    x = qd.field(qd.f32)

    N = 1

    qd.root.dense(qd.i, N).place(x)

    @qd.kernel
    def func():
        i = 0
        s = 0
        while i < 10:
            s += i
            i += 1
        x[0] = s

    func()
    assert x[0] == 45


@test_utils.test()
def test_break():
    ret = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        i = 0
        s = 0
        while True:
            s += i
            i += 1
            if i > 10:
                break
        ret[None] = s

    func()
    assert ret[None] == 55
