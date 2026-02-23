import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_pointer():
    e = qd.Vector.field(2, dtype=int, shape=16)

    e[0] = qd.Vector([0, 0])

    a = qd.field(float, shape=512)
    b = qd.field(dtype=float)
    qd.root.pointer(qd.i, 32).dense(qd.i, 16).place(b)

    @qd.kernel
    def test():
        for i in a:
            a[i] = i
        for i in a:
            b[i] += a[i]

    test()
    qd.sync()

    b_np = b.to_numpy()
    for i in range(512):
        assert b_np[i] == i
