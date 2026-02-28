import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

"""
A_psd used in the tests is a random positive definite matrix with a given number of rows and columns:
A_psd = A * A^T
Reference: https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices
2023.5.31 qbao: It's observed that the matrix generated above is semi-definite, and it fails about 5% of the tests.
Therefore, A_psd is modified from A * A^T to A * A^T + np.eye(n) to improve stability.
"""


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=qd.x64, print_full_traceback=False)
def test_sparse_LLT_solver(dtype, solver_type, ordering):
    np_dtype = qd.lang.util.to_numpy_type(dtype)
    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np_dtype)
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype)
    b = qd.field(dtype=dtype, shape=n)

    @qd.kernel
    def fill(
        Abuilder: qd.types.sparse_matrix_builder(),
        InputArray: qd.types.ndarray(),
        b: qd.template(),
    ):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = qd.linalg.SparseSolver(dtype=dtype, solver_type=solver_type, ordering=ordering)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


@pytest.mark.parametrize("dtype", [qd.f32])
@pytest.mark.parametrize("solver_type", ["LLT", "LDLT", "LU"])
@pytest.mark.parametrize("ordering", ["AMD", "COLAMD"])
@test_utils.test(arch=qd.cpu)
def test_sparse_solver_ndarray_vector(dtype, solver_type, ordering):
    np_dtype = qd.lang.util.to_numpy_type(dtype)
    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np_dtype)
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=dtype)
    b = qd.ndarray(qd.f32, shape=n)

    @qd.kernel
    def fill(
        Abuilder: qd.types.sparse_matrix_builder(),
        InputArray: qd.types.ndarray(),
        b: qd.types.ndarray(),
    ):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += InputArray[i, j]
        for i in range(n):
            b[i] = i + 1

    fill(Abuilder, A_psd, b)
    A = Abuilder.build()
    solver = qd.linalg.SparseSolver(dtype=dtype, solver_type=solver_type, ordering=ordering)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


