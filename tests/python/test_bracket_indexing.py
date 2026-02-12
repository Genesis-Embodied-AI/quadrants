import quadrants as ti

from tests import test_utils


@test_utils.test()
def test_bracket_indexing_field():
    a = ti.field(ti.i32, ())

    @ti.kernel
    def k1():
        a[()] += 1

    k1()
    assert a[()] == 1


@test_utils.test()
def test_bracket_indexing_ndarray():
    a = ti.ndarray(ti.i32, ())

    @ti.kernel
    def k1(a: ti.types.NDArray[ti.i32, 0]):
        a[()] += 1

    k1(a)
    assert a[()] == 1
