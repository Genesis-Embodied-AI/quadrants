import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_argument_error():
    x = qd.field(qd.i32)

    qd.root.place(x)

    try:

        @qd.kernel
        def set_i32_notype(v):
            pass

    except qd.QuadrantsSyntaxError:
        pass

    try:

        @qd.kernel
        def set_i32_args(*args):
            pass

    except qd.QuadrantsSyntaxError:
        pass

    try:

        @qd.kernel
        def set_i32_kwargs(**kwargs):
            pass

    except qd.QuadrantsSyntaxError:
        pass

    @qd.kernel
    def set_i32(v: qd.i32):
        x[None] = v

    set_i32(123)
    assert x[None] == 123
