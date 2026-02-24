import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_rescale():
    a = qd.field(qd.f32)
    b = qd.field(qd.f32)
    qd.root.dense(qd.ij, 4).dense(qd.ij, 4).place(a)
    qd.root.dense(qd.ij, 4).place(b)

    @qd.kernel
    def set_b():
        for I in qd.grouped(a):
            Ib = qd.rescale_index(a, b, I)
            b[Ib] += 1.0

    @qd.kernel
    def set_a():
        for I in qd.grouped(b):
            Ia = qd.rescale_index(b, a, I)
            a[Ia] = 1.0

    set_a()
    set_b()

    for i in range(0, 4):
        for j in range(0, 4):
            assert b[i, j] == 16

    for i in range(0, 16):
        for j in range(0, 16):
            if i % 4 == 0 and j % 4 == 0:
                assert a[i, j] == 1
            else:
                assert a[i, j] == 0
