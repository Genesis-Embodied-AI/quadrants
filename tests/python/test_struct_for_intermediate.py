import quadrants as qd

from tests import test_utils


def _test_nested():
    x = qd.field(qd.i32)

    p, q = 3, 7
    n, m = 2, 4

    qd.root.dense(qd.ij, (p, q)).dense(qd.ij, (n, m)).place(x)

    @qd.kernel
    def iterate():
        for i, j in x.parent():
            x[i, j] += 1

    iterate()
    for i in range(p):
        for j in range(q):
            assert x[i * n, j * m] == 1, (i, j)


@test_utils.test(require=qd.extension.sparse, demote_dense_struct_fors=False)
def test_nested():
    _test_nested()


@test_utils.test(demote_dense_struct_fors=True)
def test_nested_demote():
    _test_nested()
