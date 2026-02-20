import quadrants as qd

from tests import test_utils


@qd.kernel
def some_kernel(_: qd.template()): ...


@test_utils.test(cpu_max_num_threads=1)
def test_get_snode_tree_id():
    s = qd.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 0

    s = qd.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 1

    s = qd.field(int, shape=())
    some_kernel(s)
    assert s.snode._snode_tree_id == 2
