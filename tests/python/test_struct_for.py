import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_singleton():
    x = qd.field(qd.i32, shape=())

    @qd.kernel
    def fill():
        for I in qd.grouped(x):
            x[I] = 3

    fill()

    assert x[None] == 3


@test_utils.test()
def test_singleton2():
    x = qd.field(qd.i32)

    qd.root.place(x)

    @qd.kernel
    def fill():
        for I in qd.grouped(x):
            x[I] = 3

    fill()

    assert x[None] == 3


@test_utils.test()
def test_linear():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.dense(qd.i, n).place(y)

    @qd.kernel
    def fill():
        for i in x:
            x[i] = i
            y[i] = i * 2

    fill()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i * 2


@test_utils.test()
def test_nested():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.i, n // 4).dense(qd.i, 4).place(x)
    qd.root.dense(qd.i, n).place(y)

    @qd.kernel
    def fill():
        for i in x:
            x[i] = i
            y[i] = i * 2

    fill()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i * 2


@test_utils.test()
def test_nested2():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 2048

    qd.root.dense(qd.i, n // 512).dense(qd.i, 16).dense(qd.i, 8).dense(qd.i, 4).place(x)
    qd.root.dense(qd.i, n).place(y)

    @qd.kernel
    def fill():
        for i in x:
            x[i] = i
            y[i] = i * 2

    fill()

    for i in range(n):
        assert x[i] == i
        assert y[i] == i * 2


@test_utils.test()
def test_2d():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n, m = 32, 16

    qd.root.dense(qd.ij, n).place(x, y)

    @qd.kernel
    def fill():
        for i, j in x:
            x[i, j] = i + j * 2

    fill()

    for i in range(n):
        for j in range(m):
            assert x[i, j] == i + j * 2


@test_utils.test()
def test_2d_non_POT():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32, shape=())

    n, m = 13, 17

    qd.root.dense(qd.ij, (n, m)).place(x)

    @qd.kernel
    def fill():
        for i, j in x:
            y[None] += i + j * j

    fill()

    tot = 0
    for i in range(n):
        for j in range(m):
            tot += i + j * j
    assert y[None] == tot


@test_utils.test()
def test_nested_2d():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 32

    qd.root.dense(qd.ij, n // 4).dense(qd.ij, 4).place(x, y)

    @qd.kernel
    def fill():
        for i, j in x:
            x[i, j] = i + j * 2

    fill()

    for i in range(n):
        for j in range(n):
            assert x[i, j] == i + j * 2


@test_utils.test()
def test_nested_2d_more_nests():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 64

    qd.root.dense(qd.ij, n // 16).dense(qd.ij, 2).dense(qd.ij, 4).dense(qd.ij, 2).place(x, y)

    @qd.kernel
    def fill():
        for i, j in x:
            x[i, j] = i + j * 2

    fill()

    for i in range(n):
        for j in range(n):
            assert x[i, j] == i + j * 2


@test_utils.test()
def test_linear_k():
    x = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.k, n).place(x)

    @qd.kernel
    def fill():
        for i in x:
            x[i] = i

    fill()

    for i in range(n):
        assert x[i] == i


@test_utils.test(require=qd.extension.sparse)
def test_struct_for_branching():
    # Related issue: https://github.com/taichi-dev/quadrants/issues/704
    x = qd.field(dtype=qd.i32)
    y = qd.field(dtype=qd.i32)
    qd.root.pointer(qd.ij, 128 // 4).dense(qd.ij, 4).place(x, y)

    @qd.kernel
    def func1():
        for i, j in x:
            if x[i, j] & 2 == 2:
                y[i, j] = 1

    @qd.kernel
    def func2():
        for i, j in x:
            if x[i, j] == 2 or x[i, j] == 4:
                y[i, j] = 1

    @qd.kernel
    def func3():
        for i, j in x:
            if x[i, j] & 2 == 2 or x[i, j] & 4 == 4:
                y[i, j] = 1

    func1()
    func2()
    func3()


@test_utils.test(require=qd.extension.sparse)
def test_struct_for_pointer_block():
    n = 16
    block_size = 8

    f = qd.field(dtype=qd.f32)

    block = qd.root.pointer(qd.ijk, n // block_size)
    block.dense(qd.ijk, block_size).place(f)

    f[0, 2, 3] = 1

    @qd.kernel
    def count() -> int:
        tot = 0
        for I in qd.grouped(block):
            tot += 1
        return tot

    assert count() == 1


@test_utils.test(require=qd.extension.quant)
def test_struct_for_quant():
    n = 8

    qi13 = qd.types.quant.int(13, True)
    x = qd.field(dtype=qi13)

    bitpack = qd.BitpackedFields(max_num_bits=32)
    bitpack.place(x)
    qd.root.dense(qd.i, n).place(bitpack)

    @qd.kernel
    def count() -> int:
        tot = 0
        for i in x:
            tot += i
        return tot

    assert count() == 28


@test_utils.test(require=qd.extension.sparse)
def test_struct_for_continue():
    # Related issue: https://github.com/taichi-dev/quadrants/issues/3272
    x = qd.field(dtype=qd.i32)
    n = 4
    qd.root.pointer(qd.i, n).dense(qd.i, n).place(x)

    @qd.kernel
    def init():
        for i in range(n):
            x[i * n + i] = 1

    @qd.kernel
    def struct_for_continue() -> qd.i32:
        cnt = 0
        for i in x:
            if x[i]:
                continue
            cnt += 1
        return cnt

    @qd.kernel
    def range_for_continue() -> qd.i32:
        cnt = 0
        for i in range(n * n):
            if x[i]:
                continue
            cnt += 1
        return cnt

    init()
    assert struct_for_continue() == n * (n - 1)
    assert range_for_continue() == n * (n - 1)
