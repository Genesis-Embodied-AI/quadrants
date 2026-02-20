import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse)
def test_pointer():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 16

    ptr = qd.root.pointer(qd.i, n)
    ptr.dense(qd.i, n).place(x)
    qd.root.place(s)

    s[None] = 0

    @qd.kernel
    def activate():
        qd.activate(ptr, qd.rescale_index(x, ptr, [1]))
        qd.activate(ptr, qd.rescale_index(x, ptr, [32]))

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    activate()
    func()
    assert s[None] == 32


@test_utils.test(require=qd.extension.sparse)
def test_non_dfs_snode_order():
    x = qd.field(dtype=qd.i32)
    y = qd.field(dtype=qd.i32)

    grid1 = qd.root.dense(qd.i, 1)
    grid2 = qd.root.dense(qd.i, 1)
    ptr = grid1.pointer(qd.i, 1)
    ptr.place(x)
    grid2.place(y)
    """
    This SNode tree has node ids that do not follow DFS order:
    S0root
      S1dense
        S3pointer
          S4place<i32>
      S2dense
        S5place<i32>
    """

    @qd.kernel
    def foo():
        qd.activate(ptr, [0])

    foo()  # Just make sure it doesn't crash
    qd.sync()
