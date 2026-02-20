import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_1D():
    N = 2
    x = qd.field(qd.f32)
    qd.root.dense(qd.i, N).place(x)

    x[0] = 42
    assert x[0] == 42
    assert x[1] == 0
