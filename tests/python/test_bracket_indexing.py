import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_bracket_indexing_field():
    a = qd.field(qd.i32, ())

    @qd.kernel
    def k1():
        a[()] += 1

    k1()
    assert a[()] == 1


@test_utils.test()
def test_bracket_indexing_ndarray():
    a = qd.ndarray(qd.i32, ())

    @qd.kernel
    def k1(a: qd.types.NDArray[qd.i32, 0]):
        a[()] += 1

    k1(a)
    assert a[()] == 1
