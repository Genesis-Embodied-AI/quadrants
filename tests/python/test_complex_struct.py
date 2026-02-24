import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_complex_dense():
    a = qd.field(qd.i32, shape=(4, 4))
    b = qd.field(qd.i32, shape=(16, 16))
    c = qd.field(qd.i32, shape=(16, 4))
    d = qd.field(qd.i32, shape=(4, 4, 4))

    w = qd.field(qd.i32)
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)
    z = qd.field(qd.i32)

    blk = qd.root.dense(qd.ij, 4)
    blk.place(w)
    blk.dense(qd.ij, 2).dense(qd.ij, 2).place(x)
    blk.dense(qd.i, 4).place(y)
    blk.dense(qd.k, 4).place(z)

    @qd.kernel
    def set_w():
        for I in qd.grouped(qd.ndrange(4, 4)):
            w[I] = 1

    @qd.kernel
    def set_x():
        for I in qd.grouped(qd.ndrange(16, 16)):
            x[I] = 2

    @qd.kernel
    def set_y():
        for I in qd.grouped(qd.ndrange(16, 4)):
            y[I] = 3

    @qd.kernel
    def set_z():
        for I in qd.grouped(qd.ndrange(4, 4, 4)):
            z[I] = 4

    @qd.kernel
    def set_a():
        for I in qd.grouped(w):
            a[I] = w[I]

    @qd.kernel
    def set_b():
        for I in qd.grouped(x):
            b[I] = x[I]

    @qd.kernel
    def set_c():
        for I in qd.grouped(y):
            c[I] = y[I]

    @qd.kernel
    def set_d():
        for I in qd.grouped(z):
            d[I] = z[I]

    set_w()
    set_x()
    set_y()
    set_z()

    set_a()
    set_b()
    set_c()
    set_d()

    for i in range(4):
        for j in range(4):
            assert a[i, j] == 1

    for i in range(16):
        for j in range(16):
            assert b[i, j] == 2

    for i in range(16):
        for j in range(4):
            assert c[i, j] == 3

    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert d[i, j, k] == 4


@test_utils.test(require=qd.extension.sparse)
def test_complex_pointer():
    a = qd.field(qd.i32, shape=(4, 4))
    b = qd.field(qd.i32, shape=(16, 16))
    c = qd.field(qd.i32, shape=(16, 4))
    d = qd.field(qd.i32, shape=(4, 4, 4))

    w = qd.field(qd.i32)
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)
    z = qd.field(qd.i32)

    blk = qd.root.pointer(qd.ij, 4)
    blk.place(w)
    blk.pointer(qd.ij, 2).dense(qd.ij, 2).place(x)
    blk.dense(qd.i, 4).place(y)
    blk.dense(qd.k, 4).place(z)

    @qd.kernel
    def set_w():
        for I in qd.grouped(qd.ndrange(4, 4)):
            w[I] = 1

    @qd.kernel
    def set_x():
        for I in qd.grouped(qd.ndrange(16, 16)):
            x[I] = 2

    @qd.kernel
    def set_y():
        for I in qd.grouped(qd.ndrange(16, 4)):
            y[I] = 3

    @qd.kernel
    def set_z():
        for I in qd.grouped(qd.ndrange(4, 4, 4)):
            z[I] = 4

    @qd.kernel
    def set_a():
        for I in qd.grouped(w):
            a[I] = w[I]

    @qd.kernel
    def set_b():
        for I in qd.grouped(x):
            b[I] = x[I]

    @qd.kernel
    def set_c():
        for I in qd.grouped(y):
            c[I] = y[I]

    @qd.kernel
    def set_d():
        for I in qd.grouped(z):
            d[I] = z[I]

    set_w()
    set_x()
    set_y()
    set_z()

    set_a()
    set_b()
    set_c()
    set_d()

    for i in range(4):
        for j in range(4):
            assert a[i, j] == 1

    for i in range(16):
        for j in range(16):
            assert b[i, j] == 2

    for i in range(16):
        for j in range(4):
            assert c[i, j] == 3

    for i in range(4):
        for j in range(4):
            for k in range(4):
                assert d[i, j, k] == 4
