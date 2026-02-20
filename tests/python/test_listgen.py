from random import randrange

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_listgen():
    x = qd.field(qd.i32)
    n = 1024

    qd.root.dense(qd.ij, 4).dense(qd.ij, 4).dense(qd.ij, 4).dense(qd.ij, 4).dense(qd.ij, 4).place(x)

    @qd.kernel
    def fill(c: qd.i32):
        for i, j in x:
            x[i, j] = i * 10 + j + c

    for c in range(2):
        print("Testing c=%d" % c)
        fill(c)
        # read it out once to avoid launching too many operator[] kernels
        xnp = x.to_numpy()
        for i in range(n):
            for j in range(n):
                assert xnp[i, j] == i * 10 + j + c

        # Randomly check 1000 items to ensure [] work as well
        for _ in range(1000):
            i, j = randrange(n), randrange(n)
            assert x[i, j] == i * 10 + j + c


@test_utils.test()
def test_nested_3d():
    x = qd.field(qd.i32)
    n = 128

    qd.root.dense(qd.ijk, 4).dense(qd.ijk, 4).dense(qd.ijk, 4).dense(qd.ijk, 2).place(x)

    @qd.kernel
    def fill():
        for i, j, k in x:
            x[i, j, k] = (i * n + j) * n + k

    fill()
    # read it out once to avoid launching too many operator[] kernels
    xnp = x.to_numpy()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                assert xnp[i, j, k] == (i * n + j) * n + k

    # Randomly check 1000 items to ensure [] work as well
    for _ in range(1000):
        i, j, k = randrange(n), randrange(n), randrange(n)
        assert x[i, j, k] == (i * n + j) * n + k
