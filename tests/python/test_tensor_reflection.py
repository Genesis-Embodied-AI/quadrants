import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


@test_utils.test()
def test_POT():
    val = qd.field(qd.i32)

    n = 4
    m = 8
    p = 16

    qd.root.dense(qd.i, n).dense(qd.j, m).dense(qd.k, p).place(val)

    assert val.shape == (n, m, p)
    assert val.dtype == qd.i32


@test_utils.test()
def test_non_POT():
    val = qd.field(qd.i32)

    n = 3
    m = 7
    p = 11

    blk1 = qd.root.dense(qd.i, n)
    blk2 = blk1.dense(qd.j, m)
    blk3 = blk2.dense(qd.k, p)
    blk3.place(val)

    assert val.shape == (n, m, p)
    assert val.dtype == qd.i32


@test_utils.test()
def test_unordered():
    val = qd.field(qd.i32)

    n = 3
    m = 7
    p = 11

    blk1 = qd.root.dense(qd.k, n)
    blk2 = blk1.dense(qd.i, m)
    blk3 = blk2.dense(qd.j, p)
    blk3.place(val)

    assert val.dtype == qd.i32
    assert val.shape == (m, p, n)
    assert val.snode.parent(0) == val.snode
    assert val.snode.parent() == blk3
    assert val.snode.parent(1) == blk3
    assert val.snode.parent(2) == blk2
    assert val.snode.parent(3) == blk1
    assert val.snode.parent(4) == qd.root

    assert val.snode in blk3._get_children()
    assert blk3 in blk2._get_children()
    assert blk2 in blk1._get_children()
    impl.get_runtime().materialize_root_fb(False)
    assert blk1 in qd.FieldsBuilder._finalized_roots()[0]._get_children()

    expected_str = f"qd.root => dense {[n]} => dense {[m, n]}" f" => dense {[m, p, n]} => place {[m, p, n]}"
    assert str(val.snode) == expected_str


@test_utils.test()
def test_unordered_matrix():
    val = qd.Matrix.field(3, 2, qd.i32)

    n = 3
    m = 7
    p = 11

    blk1 = qd.root.dense(qd.k, n)
    blk2 = blk1.dense(qd.i, m)
    blk3 = blk2.dense(qd.j, p)
    blk3.place(val)

    assert val.shape == (m, p, n)
    assert val.dtype == qd.i32
    assert val.snode.parent(0) == val.snode
    assert val.snode.parent() == blk3
    assert val.snode.parent(1) == blk3
    assert val.snode.parent(2) == blk2
    assert val.snode.parent(3) == blk1
    assert val.snode.parent(4) == qd.root
    assert val.snode._path_from_root() == [qd.root, blk1, blk2, blk3, val.snode]


@test_utils.test()
def test_parent_exceeded():
    val = qd.field(qd.f32)

    m = 7
    n = 3

    blk1 = qd.root.dense(qd.i, m)
    blk2 = blk1.dense(qd.j, n)
    blk2.place(val)

    assert val.snode.parent() == blk2
    assert val.snode.parent(2) == blk1
    assert val.snode.parent(3) == qd.root
    assert val.snode.parent(4) == None
    assert val.snode.parent(42) == None

    assert qd.root.parent() == None
