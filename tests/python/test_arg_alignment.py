import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_ret_write():
    @qd.kernel
    def func(a: qd.i16) -> qd.f32:
        return 3.0

    assert func(255) == 3.0


@test_utils.test()
def test_arg_read():
    x = qd.field(qd.i32, shape=())

    @qd.kernel
    def func(a: qd.i8, b: qd.i32):
        x[None] = b

    func(255, 2)
    assert x[None] == 2
