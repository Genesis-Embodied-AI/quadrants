import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_1d():
    N = 16

    x = qd.field(qd.f32, shape=(N,))
    y = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def func():
        for i in range(N):
            y[i] = x[i]

    for i in range(N):
        x[i] = i * 2

    func()

    for i in range(N):
        assert y[i] == i * 2


@test_utils.test()
def test_3d():
    N = 2
    M = 2

    x = qd.field(qd.f32, shape=(N, M))
    y = qd.field(qd.f32, shape=(N, M))

    @qd.kernel
    def func():
        for I in qd.grouped(x):
            y[I] = x[I]

    for i in range(N):
        for j in range(M):
            x[i, j] = i * 10 + j

    func()

    for i in range(N):
        for j in range(M):
            assert y[i, j] == i * 10 + j


@test_utils.test()
def test_matrix():
    N = 16

    x = qd.Matrix.field(2, 2, dtype=qd.f32, shape=(N,), layout=qd.Layout.AOS)

    @qd.kernel
    def func():
        for i in range(N):
            x[i][1, 1] = x[i][0, 0]

    for i in range(N):
        x[i][0, 0] = i + 3

    func()

    for i in range(N):
        assert x[i][1, 1] == i + 3


@test_utils.test()
def test_alloc_in_kernel():
    return  # build bots may not have this much memory to tests...
    x = qd.field(qd.f32)

    qd.root.pointer(qd.i, 8192).dense(qd.i, 1024 * 1024).place(x)

    @qd.kernel
    def touch():
        for i in range(4096):
            x[i * 1024 * 1024] = 1

    touch()
