import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_kernel_template_basic():
    x = qd.field(qd.i32)
    y = qd.field(qd.f32)

    n = 16

    qd.root.dense(qd.i, n).place(x, y)

    @qd.kernel
    def inc(a: qd.template(), b: qd.template()):
        for i in a:
            a[i] += b

    inc(x, 1)
    inc(y, 2)

    for i in range(n):
        assert x[i] == 1
        assert y[i] == 2

    @qd.kernel
    def inc2(z: qd.i32, a: qd.template(), b: qd.i32):
        for i in a:
            a[i] += b + z

    inc2(10, x, 1)
    for i in range(n):
        assert x[i] == 12


@test_utils.test()
def test_kernel_template_gradient():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    z = qd.field(qd.f32)
    loss = qd.field(qd.f32)

    qd.root.dense(qd.i, 16).place(x, y, z)
    qd.root.place(loss)
    qd.root.lazy_grad()

    @qd.kernel
    def double(a: qd.template(), b: qd.template()):
        for i in range(16):
            b[i] = a[i] * 2 + 1

    @qd.kernel
    def compute_loss():
        for i in range(16):
            qd.atomic_add(loss[None], z[i])

    for i in range(16):
        x[i] = i

    with qd.ad.Tape(loss):
        double(x, y)
        double(y, z)
        compute_loss()

    for i in range(16):
        assert z[i] == i * 4 + 3
        assert x.grad[i] == 4


@test_utils.test()
def test_func_template():
    a = [qd.field(dtype=qd.f32) for _ in range(2)]
    b = [qd.field(dtype=qd.f32) for _ in range(2)]

    for l in range(2):
        qd.root.dense(qd.ij, 16).place(a[l], b[l])

    @qd.func
    def sample(x: qd.template(), l: qd.template(), I):
        return x[l][I]

    @qd.kernel
    def fill(l: qd.template()):
        for I in qd.grouped(a[l]):
            a[l][I] = l

    @qd.kernel
    def aTob(l: qd.template()):
        for I in qd.grouped(b[l]):
            b[l][I] = sample(a, l, I)

    for l in range(2):
        fill(l)
        aTob(l)

    for l in range(2):
        for i in range(16):
            for j in range(16):
                assert b[l][i, j] == l


@test_utils.test()
def test_func_template2():
    a = qd.field(dtype=qd.f32)
    b = qd.field(dtype=qd.f32)

    qd.root.dense(qd.ij, 16).place(a, b)

    @qd.func
    def sample(x: qd.template(), I):
        return x[I]

    @qd.kernel
    def fill():
        for I in qd.grouped(a):
            a[I] = 1.0

    @qd.kernel
    def aTob():
        for I in qd.grouped(b):
            b[I] = sample(a, I)

    for l in range(2):
        fill()
        aTob()

    for i in range(16):
        for j in range(16):
            assert b[i, j] == 1.0
