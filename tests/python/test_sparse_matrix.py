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
def test_sparse_matrix_builder_deprecated_anno(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


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
def test_sparse_matrix_builder(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        for j in range(n):
            assert A[i, j] == i + j


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
def test_build_sparse_matrix_frome_ndarray(dtype, storage_format):
    n = 8
    triplets = qd.Vector.ndarray(n=3, dtype=qd.f32, shape=n)
    A = qd.linalg.SparseMatrix(n=10, m=10, dtype=qd.f32, storage_format=storage_format)

    @qd.kernel
    def fill(triplets: qd.types.ndarray()):
        for i in range(n):
            triplet = qd.Vector([i, i, i], dt=qd.f32)
            triplets[i] = triplet

    fill(triplets)
    A.build_from_ndarray(triplets)

    for i in range(n):
        assert A[i, i] == i


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
def test_sparse_matrix_shape(dtype, storage_format):
    n, m = 8, 9
    Abuilder = qd.linalg.SparseMatrixBuilder(n, m, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i, j in qd.ndrange(n, m):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    assert A.shape == (n, m)


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
def test_sparse_matrix_element_access(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    for i in range(n):
        assert A[i, i] == i


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
def test_sparse_matrix_element_modify(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i in range(n):
            Abuilder[i, i] += i

    fill(Abuilder)
    A = Abuilder.build()
    A[0, 0] = 1024.0
    assert A[0, 0] == 1024.0


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
def test_sparse_matrix_addition(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(
        Abuilder: qd.types.sparse_matrix_builder(),
        Bbuilder: qd.types.sparse_matrix_builder(),
    ):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A + B
    for i in range(n):
        for j in range(n):
            assert C[i, j] == 2 * i


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
def test_sparse_matrix_subtraction(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(
        Abuilder: qd.types.sparse_matrix_builder(),
        Bbuilder: qd.types.sparse_matrix_builder(),
    ):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A - B
    for i in range(n):
        for j in range(n):
            assert C[i, j] == 2 * j


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
def test_sparse_matrix_scalar_multiplication(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A * 3.0
    for i in range(n):
        for j in range(n):
            assert B[i, j] == 3 * (i + j)


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
def test_sparse_matrix_transpose(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    A = Abuilder.build()
    B = A.transpose()
    for i in range(n):
        for j in range(n):
            assert B[i, j] == A[j, i]


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
def test_sparse_matrix_elementwise_multiplication(dtype, storage_format):
    n = 8
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(
        Abuilder: qd.types.sparse_matrix_builder(),
        Bbuilder: qd.types.sparse_matrix_builder(),
    ):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A * B
    for i in range(n):
        for j in range(n):
            assert C[i, j] == (i + j) * (i - j)


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
def test_sparse_matrix_multiplication(dtype, storage_format):
    n = 2
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(
        Abuilder: qd.types.sparse_matrix_builder(),
        Bbuilder: qd.types.sparse_matrix_builder(),
    ):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j
            Bbuilder[i, j] += i - j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A @ B
    assert C[0, 0] == 1.0
    assert C[0, 1] == 0.0
    assert C[1, 0] == 2.0
    assert C[1, 1] == -1.0


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
@pytest.mark.flaky(retries=5)
def test_sparse_matrix_nonsymmetric_multiplication(dtype, storage_format):
    n, k, m = 2, 3, 4
    Abuilder = qd.linalg.SparseMatrixBuilder(n, k, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    Bbuilder = qd.linalg.SparseMatrixBuilder(k, m, max_num_triplets=100, dtype=dtype, storage_format=storage_format)

    @qd.kernel
    def fill(
        Abuilder: qd.types.sparse_matrix_builder(),
        Bbuilder: qd.types.sparse_matrix_builder(),
    ):
        for i, j in qd.ndrange(n, k):
            Abuilder[i, j] += i + j
        for i, j in qd.ndrange(k, m):
            Bbuilder[i, j] -= i + j

    fill(Abuilder, Bbuilder)
    A = Abuilder.build()
    B = Bbuilder.build()
    C = A @ B
    GT = [[-5, -8, -11, -14], [-8, -14, -20, -26]]
    for i in range(n):
        for j in range(m):
            assert C[i, j] == GT[i][j]


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
def test_sparse_matrix_ndarray_vector_multiplication(dtype, storage_format):
    n = 2
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100, dtype=dtype, storage_format=storage_format)
    x = qd.ndarray(dtype, n)

    @qd.kernel
    def fill(Abuilder: qd.types.sparse_matrix_builder()):
        for i, j in qd.ndrange(n, n):
            Abuilder[i, j] += i + j

    fill(Abuilder)
    x.fill(1.0)
    A = Abuilder.build()
    res = A @ x
    res_n = res.to_numpy()
    assert res_n[0] == 1.0
    assert res_n[1] == 3.0


@test_utils.test(arch=qd.cuda)
def test_gpu_sparse_matrix():
    import numpy as np

    num_triplets, num_rows, num_cols = 9, 4, 4
    np_idx_dtype, np_val_dtype = np.int32, np.float32
    coo_row = np.asarray([0, 0, 0, 1, 2, 2, 2, 3, 3], dtype=np_idx_dtype)
    coo_col = np.asarray([0, 2, 3, 1, 0, 2, 3, 1, 3], dtype=np_idx_dtype)
    coo_val = np.asarray([i + 1.0 for i in range(num_triplets)], dtype=np_val_dtype)
    h_X = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np_val_dtype)
    h_Y = np.asarray([19.0, 8.0, 51.0, 52.0], dtype=np_val_dtype)

    ti_dtype = qd.f32
    X = qd.ndarray(shape=num_cols, dtype=ti_dtype)
    Y = qd.ndarray(shape=num_rows, dtype=ti_dtype)

    X.from_numpy(h_X)
    Y.fill(0.0)

    A_builder = qd.linalg.SparseMatrixBuilder(num_rows=4, num_cols=4, dtype=ti_dtype, max_num_triplets=50)

    @qd.kernel
    def fill(
        A: qd.types.sparse_matrix_builder(),
        coo_row: qd.types.ndarray(),
        coo_col: qd.types.ndarray(),
        coo_val: qd.types.ndarray(),
    ):
        for i in range(num_triplets):
            A[coo_row[i], coo_col[i]] += coo_val[i]

    fill(A_builder, coo_row, coo_col, coo_val)
    A = A_builder.build()

    # Compute Y = A @ X
    Y = A @ X
    for i in range(4):
        assert Y[i] == h_Y[i]


@pytest.mark.parametrize("N", [5])
@test_utils.test(arch=qd.cuda)
def test_gpu_sparse_matrix_ops(N):
    import numpy as np
    from numpy.random import default_rng
    from scipy import stats
    from scipy.sparse import random

    @qd.kernel
    def fill(
        A: qd.types.sparse_matrix_builder(),
        coo_row: qd.types.ndarray(),
        coo_col: qd.types.ndarray(),
        coo_val: qd.types.ndarray(),
        nnz: qd.i32,
    ):
        for i in range(nnz):
            A[coo_row[i], coo_col[i]] += coo_val[i]

    seed = 2
    np.random.seed(seed)
    rng = default_rng(seed)
    rvs = stats.poisson(3, loc=1).rvs
    np_dtype = np.float32
    val_dt = qd.float32

    n_rows = N
    n_cols = N - 1

    S1 = random(n_rows, n_cols, density=0.5, random_state=rng, data_rvs=rvs).astype(np_dtype).tocoo()
    S2 = random(n_rows, n_cols, density=0.5, random_state=rng, data_rvs=rvs).astype(np_dtype).tocoo()

    nnz_A = S1.nnz
    nnz_B = S2.nnz

    A_builder = qd.linalg.SparseMatrixBuilder(num_rows=n_rows, num_cols=n_cols, dtype=val_dt, max_num_triplets=nnz_A)
    B_builder = qd.linalg.SparseMatrixBuilder(num_rows=n_rows, num_cols=n_cols, dtype=val_dt, max_num_triplets=nnz_B)
    fill(A_builder, S1.row, S1.col, S1.data, nnz_A)
    fill(B_builder, S2.row, S2.col, S2.data, nnz_B)
    A = A_builder.build()
    B = B_builder.build()

    def verify(scipy_spm, quadrants_spm):
        scipy_spm = scipy_spm.tocoo()
        for i, j, v in zip(scipy_spm.row, scipy_spm.col, scipy_spm.data):
            assert v == test_utils.approx(quadrants_spm[i, j], rel=1e-5)

    C = A + B
    S3 = S1 + S2
    verify(S3, C)

    D = C - A
    S4 = S3 - S1
    verify(S4, D)

    E = A * 2.5
    S5 = S1 * 2.5
    verify(S5, E)

    F = A * 2.5
    S6 = S1 * 2.5
    verify(S6, F)

    G = A.transpose()
    S7 = S1.T
    verify(S7, G)

    H = A @ B.transpose()
    S8 = S1 @ S2.T
    verify(S8, H)
