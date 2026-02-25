import quadrants as qd

from tests import test_utils


@test_utils.test(
    require=qd.extension.sparse,
    exclude=[qd.vulkan, qd.metal],
)
def test_dynamic():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32, shape=())

    n = 128

    qd.root.dynamic(qd.i, n).place(x)

    @qd.kernel
    def count():
        for i in x:
            y[None] += 1

    x[n // 3] = 1

    count()

    assert y[None] == n // 3 + 1


@test_utils.test(
    require=qd.extension.sparse,
    exclude=[qd.vulkan, qd.metal],
)
def test_dense_dynamic():
    n = 128

    x = qd.field(qd.i32)

    qd.root.dense(qd.i, n).dynamic(qd.j, n, 128).place(x)

    @qd.kernel
    def append():
        for i in range(n):
            for j in range(i):
                qd.append(x.parent(), i, j * 2)

    append()

    for i in range(n):
        for j in range(i):
            assert x[i, j] == j * 2
