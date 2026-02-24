import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_packed_size():
    x = qd.field(qd.i32)
    qd.root.dense(qd.l, 3).dense(qd.ijk, 129).place(x)
    assert x.shape == (129, 129, 129, 3)
    assert x.snode.parent().parent()._cell_size_bytes == 4 * 129**3
