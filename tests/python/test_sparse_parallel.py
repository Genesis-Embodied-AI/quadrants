import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse)
def test_pointer():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    qd.root.pointer(qd.i, n).dense(qd.i, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def activate():
        for i in range(n):
            x[i * n] = 0

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    activate()
    func()
    assert s[None] == n * n


@test_utils.test(require=qd.extension.sparse)
def test_pointer2():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    qd.root.pointer(qd.i, n).dense(qd.i, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def activate():
        for i in range(n * n):
            x[i] = i

    @qd.kernel
    def func():
        for i in x:
            s[None] += i

    activate()
    func()
    N = n * n
    assert s[None] == N * (N - 1) / 2


@test_utils.test(require=qd.extension.sparse)
def test_nested_struct_fill_and_clear():
    a = qd.field(dtype=qd.f32)
    N = 512

    ptr = qd.root.pointer(qd.ij, [N, N])
    ptr.dense(qd.ij, [8, 8]).place(a)

    @qd.kernel
    def fill():
        for i, j in qd.ndrange(N * 8, N * 8):
            a[i, j] = 2.0

    @qd.kernel
    def clear():
        for i, j in a.parent():
            qd.deactivate(ptr, qd.rescale_index(a, ptr, [i, j]))

    def task():
        fill()
        clear()

    for i in range(10):
        task()
        qd.sync()
