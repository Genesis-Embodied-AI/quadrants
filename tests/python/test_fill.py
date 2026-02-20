import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


@test_utils.test()
def test_fill_scalar_field():
    n = 4
    m = 7
    val = qd.field(qd.i32, shape=(n, m))

    val.fill(2)
    for i in range(n):
        for j in range(m):
            assert val[i, j] == 2

    @qd.kernel
    def fill_in_kernel(v: qd.i32):
        val.fill(v)

    fill_in_kernel(3)
    for i in range(n):
        for j in range(m):
            assert val[i, j] == 3


@test_utils.test()
def test_fill_matrix_field_with_scalar():
    n = 4
    m = 7
    val = qd.Matrix.field(2, 3, qd.i32, shape=(n, m))

    val.fill(2)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == 2).all()

    @qd.kernel
    def fill_in_kernel(v: qd.i32):
        val.fill(v)

    fill_in_kernel(3)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == 3).all()


@test_utils.test()
def test_fill_matrix_field_with_matrix():
    n = 4
    m = 7
    val = qd.Matrix.field(2, 3, qd.i32, shape=(n, m))

    mat = qd.Matrix([[0, 1, 2], [2, 3, 4]])
    val.fill(mat)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == mat).all()

    @qd.kernel
    def fill_in_kernel(v: qd.types.matrix(2, 3, qd.i32)):
        val.fill(v)

    mat = qd.Matrix([[4, 5, 6], [6, 7, 8]])
    fill_in_kernel(mat)
    for i in range(n):
        for j in range(m):
            assert (val[i, j] == mat).all()


@test_utils.test()
def test_fill_vector_field_recompile():
    a = qd.Vector.field(2, qd.i32, shape=3)
    for i in range(2):
        a.fill(qd.Vector([0, 0]))
    assert impl.get_runtime().get_num_compiled_functions() == 1
