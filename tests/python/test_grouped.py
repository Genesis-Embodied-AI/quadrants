import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_vector_index():
    val = qd.field(qd.i32)

    n = 4
    m = 7
    p = 11

    qd.root.dense(qd.i, n).dense(qd.j, m).dense(qd.k, p).place(val)

    @qd.kernel
    def test():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = qd.Vector([i, j, k])
                    val[I] = i + j * 2 + k * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == i + j * 2 + k * 3


@test_utils.test()
def test_grouped():
    val = qd.field(qd.i32)

    n = 4
    m = 8
    p = 16

    qd.root.dense(qd.i, n).dense(qd.j, m).dense(qd.k, p).place(val)

    @qd.kernel
    def test():
        for I in qd.grouped(val):
            val[I] = I[0] + I[1] * 2 + I[2] * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == i + j * 2 + k * 3


@test_utils.test()
def test_grouped_ndrange():
    val = qd.field(qd.i32)

    n = 4
    m = 8

    qd.root.dense(qd.ij, (n, m)).place(val)

    x0 = 2
    y0 = 3
    x1 = 1
    y1 = 6

    @qd.kernel
    def test():
        for I in qd.grouped(qd.ndrange((x0, y0), (x1, y1))):
            val[I] = I[0] + I[1] * 2

    test()

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i + j * 2 if x0 <= i < y0 and x1 <= j < y1 else 0)


@test_utils.test()
def test_static_grouped_ndrange():
    val = qd.field(qd.i32)

    n = 4
    m = 8

    qd.root.dense(qd.ij, (n, m)).place(val)

    x0 = 2
    y0 = 3
    x1 = 1
    y1 = 6

    @qd.kernel
    def test():
        for I in qd.static(qd.grouped(qd.ndrange((x0, y0), (x1, y1)))):
            val[I] = I[0] + I[1] * 2

    test()

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i + j * 2 if x0 <= i < y0 and x1 <= j < y1 else 0)


@test_utils.test()
def test_grouped_ndrange_starred():
    val = qd.field(qd.i32)

    n = 4
    m = 8
    p = 16
    dim = 3

    qd.root.dense(qd.ijk, (n, m, p)).place(val)

    @qd.kernel
    def test():
        for I in qd.grouped(qd.ndrange(*(((0, n),) * dim))):
            val[I] = I[0] + I[1] * 2 + I[2] * 3

    test()

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert val[i, j, k] == (i + j * 2 + k * 3 if j < n and k < n else 0)


@test_utils.test()
def test_grouped_ndrange_0d():
    val = qd.field(qd.i32, shape=())

    @qd.kernel
    def test():
        for I in qd.grouped(qd.ndrange()):
            val[I] = 42

    test()

    assert val[None] == 42


@test_utils.test()
def test_static_grouped_ndrange_0d():
    val = qd.field(qd.i32, shape=())

    @qd.kernel
    def test():
        for I in qd.static(qd.grouped(qd.ndrange())):
            val[I] = 42

    test()

    assert val[None] == 42


@test_utils.test()
def test_static_grouped_func():
    K = 3
    dim = 2

    v = qd.Vector.field(K, dtype=qd.i32, shape=((K,) * dim))

    def stencil_range():
        return qd.ndrange(*((K,) * (dim + 1)))

    @qd.kernel
    def p2g():
        for I in qd.static(qd.grouped(stencil_range())):
            v[I[0], I[1]][I[2]] = I[0] + I[1] * 3 + I[2] * 10

    p2g()

    for i in range(K):
        for j in range(K):
            for k in range(K):
                assert v[i, j][k] == i + j * 3 + k * 10
