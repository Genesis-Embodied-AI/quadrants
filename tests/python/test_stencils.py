import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_simple():
    # Note: access simplification does not work in this case. Maybe worth fixing.
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.i, n).place(x, y)

    @qd.kernel
    def run():
        for i in range(n - 1):
            x[i] = 1
            y[i + 1] = 2

    run()

    for i in range(n - 1):
        assert x[i] == 1
        assert y[i + 1] == 2
