import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_pass_by_value():
    @qd.func
    def set_val(x, i):
        x = i

    ret = qd.field(qd.i32, shape=())

    @qd.kernel
    def task():
        set_val(ret[None], 112)

    task()
    assert ret[None] == 0
