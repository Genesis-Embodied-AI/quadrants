import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def _test_matrix_slice_read_python_scope():
    v1 = qd.Vector([1, 2, 3, 4, 5, 6])[2::3]
    assert (v1 == qd.Vector([3, 6])).all()
    m = qd.Matrix([[2, 3], [4, 5]])[:1, 1:]
    assert (m == qd.Matrix([[3]])).all()
    v2 = qd.Matrix([[1, 2], [3, 4]])[:, 1]
    assert (v2 == qd.Vector([2, 4])).all()


@test_utils.test()
def test_matrix_slice_read():
    b = 6

    @qd.kernel
    def foo1() -> qd.types.vector(3, dtype=qd.i32):
        c = qd.Vector([0, 1, 2, 3, 4, 5, 6])
        return c[:b:2]

    @qd.kernel
    def foo2() -> qd.types.matrix(2, 3, dtype=qd.i32):
        a = qd.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        return a[1::, :]

    v = foo1()
    assert (v == qd.Vector([0, 2, 4])).all()
    m = foo2()
    assert (m == qd.Matrix([[4, 5, 6], [7, 8, 9]])).all()


@test_utils.test()
def test_matrix_slice_invalid():
    @qd.kernel
    def foo1(i: qd.i32):
        a = qd.Vector([0, 1, 2, 3, 4, 5, 6])
        b = a[i::2]

    @qd.kernel
    def foo2():
        i = 2
        a = qd.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = a[:i:, :i]

    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Quadrants does not support variables in slice now",
    ):
        foo1(1)
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Quadrants does not support variables in slice now",
    ):
        foo2()


@test_utils.test()
def test_matrix_slice_with_variable():
    @qd.kernel
    def test_one_row_slice(index: qd.i32) -> qd.types.vector(2, dtype=qd.i32):
        m = qd.Matrix([[1, 2, 3], [4, 5, 6]])
        return m[:, index]

    @qd.kernel
    def test_one_col_slice(index: qd.i32) -> qd.types.vector(3, dtype=qd.i32):
        m = qd.Matrix([[1, 2, 3], [4, 5, 6]])
        return m[index, :]

    r1 = test_one_row_slice(1)
    assert (r1 == qd.Vector([2, 5])).all()
    c1 = test_one_col_slice(1)
    assert (c1 == qd.Vector([4, 5, 6])).all()


@test_utils.test()
def test_matrix_slice_write():
    @qd.kernel
    def assign_col() -> qd.types.matrix(3, 4, qd.i32):
        mat = qd.Matrix([[0, 0, 0, 0] for _ in range(3)])
        col = qd.Vector([1, 2, 3])
        mat[:, 0] = col
        return mat

    @qd.kernel
    def assign_partial_row() -> qd.types.matrix(3, 4, qd.i32):
        mat = qd.Matrix([[0, 0, 0, 0] for _ in range(3)])
        mat[1, 1:3] = qd.Vector([1, 2])
        return mat

    @qd.kernel
    def augassign_rows() -> qd.types.matrix(3, 4, qd.i32):
        mat = qd.Matrix([[1, 1, 1, 1] for _ in range(3)])
        rows = qd.Matrix([[1, 2, 3, 4] for _ in range(2)])
        mat[:2, :] += rows
        return mat

    assert (assign_col() == qd.Matrix([[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]])).all()
    assert (assign_partial_row() == qd.Matrix([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]])).all()
    assert (augassign_rows() == qd.Matrix([[2, 3, 4, 5], [2, 3, 4, 5], [1, 1, 1, 1]])).all()


@test_utils.test()
def test_matrix_slice_write_dynamic_index():
    @qd.kernel
    def foo(i: qd.i32) -> qd.types.matrix(3, 4, qd.i32):
        mat = qd.Matrix([[0, 0, 0, 0] for _ in range(3)])
        mat[i, :] = qd.Vector([1, 2, 3, 4])
        return mat

    for i in range(3):
        assert (foo(i) == qd.Matrix([[1, 2, 3, 4] if j == i else [0, 0, 0, 0] for j in range(3)])).all()
