import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse, exclude=qd.metal)
def test_no_activate():
    x = qd.field(qd.f32)

    n = 1024

    d = qd.root.dynamic(qd.i, n, chunk_size=32)
    d.place(x)

    @qd.kernel
    def initialize():
        for i in range(n):
            x[i] = 1

    @qd.kernel
    def func():
        qd.no_activate(d)
        for i in range(n // 2):
            x[i * 2 + 1] += 1

    initialize()

    func()

    for i in range(n):
        assert x[i] == i % 2 + 1
