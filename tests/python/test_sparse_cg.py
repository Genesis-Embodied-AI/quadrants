import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.flaky(retries=5)
# (issue filed to fix this at https://linear.app/genesis-ai-company/issue/CMP-21/fix-failing-test-cg-test-in-windows)
@pytest.mark.parametrize("ti_dtype", [qd.f32, qd.f64])
@test_utils.test(arch=[qd.cpu])
def test_cg(ti_dtype):
    n = 10
    A = np.random.rand(n, n)
    A_psd = np.dot(A, A.transpose())
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=ti_dtype)
    b = qd.ndarray(dtype=ti_dtype, shape=n)
    x0 = qd.ndarray(dtype=ti_dtype, shape=n)

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
    A = Abuilder.build(dtype=ti_dtype)
    cg = qd.linalg.SparseCG(A, b, x0, max_iter=50, atol=1e-6)
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)


@pytest.mark.parametrize("ti_dtype", [qd.f32])
@test_utils.test(arch=[qd.cuda])
def test_cg_cuda(ti_dtype):
    n = 10
    A = np.random.rand(n, n)
    A_psd = np.dot(A, A.transpose())
    Abuilder = qd.linalg.SparseMatrixBuilder(n, n, max_num_triplets=300, dtype=ti_dtype)
    b = qd.ndarray(dtype=ti_dtype, shape=n)
    x0 = qd.ndarray(dtype=ti_dtype, shape=n)

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
    A = Abuilder.build(dtype=ti_dtype)
    cg = qd.linalg.SparseCG(A, b, x0, max_iter=50, atol=1e-6)
    x, exit_code = cg.solve()
    res = np.linalg.solve(A_psd, b.to_numpy())
    assert exit_code == True
    for i in range(n):
        assert x[i] == test_utils.approx(res[i], rel=1.0)
