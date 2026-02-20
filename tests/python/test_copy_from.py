import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_scalar():
    n = 16

    x = qd.field(qd.i32, shape=n)
    y = qd.field(qd.i32, shape=n)

    x[1] = 2

    y[0] = 1
    y[2] = 3

    x.copy_from(y)

    assert x[0] == 1
    assert x[1] == 0
    assert x[2] == 3

    assert y[0] == 1
    assert y[1] == 0
    assert y[2] == 3


@test_utils.test()
def test_struct():
    @qd.dataclass
    class C:
        i: int
        f: float

    n = 16

    x = C.field(shape=n)
    y = C.field(shape=n)

    x[1].i = 2
    x[2].i = 4

    y[0].f = 1.0
    y[2].i = 3

    x.copy_from(y)

    assert x[0].f == 1.0
    assert x[1].i == 0
    assert x[2].i == 3

    assert y[0].f == 1.0
    assert y[1].i == 0
    assert y[2].i == 3
