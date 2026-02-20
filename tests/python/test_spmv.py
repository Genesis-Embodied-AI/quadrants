import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (qd.f32, "col_major"),
        (qd.f32, "row_major"),
        (qd.f64, "col_major"),
        (qd.f64, "row_major"),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_sparse_matrix_vector_multiplication1(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    b = qd.field(qd.f32, shape=n)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder(), b: qd.template()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()
    x = A @ b
    for i in range(n):
        assert x[i] == 8 * i


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (qd.f32, "col_major"),
        (qd.f32, "row_major"),
        (qd.f64, "col_major"),
        (qd.f64, "row_major"),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_sparse_matrix_vector_multiplication2(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    b = qd.field(qd.f32, shape=n)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder(), b: qd.template()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i - j

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()

    x = A @ b
    import numpy as np

    res = np.array([-28, -20, -12, -4, 4, 12, 20, 28])
    for i in range(n):
        assert x[i] == res[i]


@pytest.mark.parametrize(
    "dtype, storage_format",
    [
        (qd.f32, "col_major"),
        (qd.f32, "row_major"),
        (qd.f64, "col_major"),
        (qd.f64, "row_major"),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_sparse_matrix_vector_multiplication3(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    b = qd.field(qd.f32, shape=n)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder(), b: qd.template()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j

        for i in range(n):
            b[i] = 1.0

    fill(Abuilder, b)
    A = Abuilder.build()

    x = A @ b
    import numpy as np

    res = np.array([28, 36, 44, 52, 60, 68, 76, 84])
    for i in range(n):
        assert x[i] == res[i]
