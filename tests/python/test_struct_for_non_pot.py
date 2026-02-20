import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_1d():
    x = qd.field(qd.i32)
    sum = qd.field(qd.i32)

    n = 100

    qd.root.dense(qd.k, n).place(x)
    qd.root.place(sum)

    @qd.kernel
    def accumulate():
        for i in x:
            qd.atomic_add(sum[None], i)

    accumulate()

    for i in range(n):
        assert sum[None] == 4950


@test_utils.test()
def test_2d():
    x = qd.field(qd.i32)
    sum = qd.field(qd.i32)

    n = 100
    m = 19

    qd.root.dense(qd.k, n).dense(qd.i, m).place(x)
    qd.root.place(sum)

    @qd.kernel
    def accumulate():
        for i, k in x:
            qd.atomic_add(sum[None], i + k * 2)

    gt = 0
    for k in range(n):
        for i in range(m):
            gt += i + k * 2

    accumulate()

    for i in range(n):
        assert sum[None] == gt


@test_utils.test(require=qd.extension.sparse)
def test_2d_pointer():
    block_size, leaf_size = 3, 8
    x = qd.field(qd.i32)
    block = qd.root.pointer(qd.ij, (block_size, block_size))
    block.dense(qd.ij, (leaf_size, leaf_size)).place(x)

    @qd.kernel
    def activate():
        x[7, 7] = 1

    activate()

    @qd.kernel
    def test() -> qd.i32:
        res = 0
        for I in qd.grouped(x):
            res += I[0] + I[1] * 2
        return res

    ans = 0
    for i in range(leaf_size):
        for j in range(leaf_size):
            ans += i + j * 2

    assert ans == test()
