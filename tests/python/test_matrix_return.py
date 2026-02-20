import quadrants as qd

from tests import test_utils


def _test_vector_return():
    @qd.kernel
    def func() -> qd.types.vector(3, qd.i32):
        return qd.Vector([1, 2, 3])

    assert (func() == qd.Vector([1, 2, 3])).all()


@test_utils.test()
def test_vector_return():
    _test_vector_return()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_vector_return_real_matrix():
    _test_vector_return()


def _test_matrix_return():
    @qd.kernel
    def func() -> qd.types.matrix(2, 3, qd.i16):
        return qd.Matrix([[1, 2, 3], [4, 5, 6]])

    assert (func() == qd.Matrix([[1, 2, 3], [4, 5, 6]])).all()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.metal])
def test_matrix_return():
    _test_matrix_return()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_matrix_return_real_matrix():
    _test_matrix_return()


def _test_matrix_return_limit():
    @qd.kernel
    def func() -> qd.types.matrix(3, 10, qd.i32):
        return qd.Matrix(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            ]
        )

    assert (
        func()
        == qd.Matrix(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            ]
        )
    ).all()


@test_utils.test()
def test_matrix_return_limit():
    _test_matrix_return_limit()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_matrix_return_limit_real_matrix():
    _test_matrix_return_limit()
