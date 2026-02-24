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


@test_utils.test(arch=qd.cuda)
def test_gpu_sparse_solver():
    from scipy.sparse import coo_matrix

    @qd.kernel
    def init_b(b: qd.types.ndarray(), nrows: qd.i32):
        for i in range(nrows):
            b[i] = 1.0 + i / nrows

    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np.float32)

    A_raw_coo = coo_matrix(A_psd)
    nrows, ncols = A_raw_coo.shape
    nnz = A_raw_coo.nnz

    A_csr = A_raw_coo.tocsr()
    b = qd.ndarray(shape=nrows, dtype=qd.f32)
    init_b(b, nrows)

    # solve Ax = b using cusolver
    A_coo = A_csr.tocoo()
    A_builder = qd.linalg.SparseMatrixBuilder(num_rows=nrows, num_cols=ncols, dtype=qd.f32, max_num_triplets=nnz)

    @qd.kernel
    def fill(
        A_builder: qd.types.sparse_matrix_builder(),
        row_coo: qd.types.ndarray(),
        col_coo: qd.types.ndarray(),
        val_coo: qd.types.ndarray(),
    ):
        for i in range(nnz):
            A_builder[row_coo[i], col_coo[i]] += val_coo[i]

    fill(A_builder, A_coo.row, A_coo.col, A_coo.data)
    A_qd = A_builder.build()
    x_qd = qd.ndarray(shape=ncols, dtype=qd.float32)

    # solve Ax=b using numpy
    b_np = b.to_numpy()
    x_np = np.linalg.solve(A_psd, b_np)

    # solve Ax=b using cusolver refectorization
    solver = qd.linalg.SparseSolver(dtype=qd.f32)
    solver.analyze_pattern(A_qd)
    solver.factorize(A_qd)
    x_qd = solver.solve(b)
    qd.sync()
    assert np.allclose(x_qd.to_numpy(), x_np, rtol=5.0e-3)

    # solve Ax = b using compute function
    solver = qd.linalg.SparseSolver(dtype=qd.f32)
    solver.compute(A_qd)
    x_cqd = solver.solve(b)
    qd.sync()
    assert np.allclose(x_cqd.to_numpy(), x_np, rtol=5.0e-3)


@pytest.mark.parametrize("dtype", [qd.f32])
@pytest.mark.parametrize("solver_type", ["LLT", "LU"])
@test_utils.test(arch=qd.cuda)
def test_gpu_sparse_solver2(dtype, solver_type):
    np_dtype = qd.lang.util.to_numpy_type(dtype)
    n = 10
    A = np.random.rand(n, n)
    A_psd = (np.dot(A, A.transpose()) + np.eye(n)).astype(np_dtype)
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=dtype)
    b = qd.ndarray(dtype, shape=n)

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
    solver = qd.linalg.SparseSolver(dtype=dtype, solver_type=solver_type)
    solver.analyze_pattern(A)
    solver.factorize(A)
    x = solver.solve(b)

    res = np.linalg.solve(A_psd, b.to_numpy())
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)
