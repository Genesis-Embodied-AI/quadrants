import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.data64, fast_math=False)
def test_precision():
    u = qd.field(qd.f64, shape=())
    v = qd.field(qd.f64, shape=())
    w = qd.field(qd.f64, shape=())

    @qd.kernel
    def forward():
        v[None] = qd.sqrt(qd.cast(u[None] + 3.25, qd.f64))
        w[None] = qd.cast(u[None] + 7, qd.f64) / qd.cast(u[None] + 3, qd.f64)

    forward()
    assert v[None] ** 2 == test_utils.approx(3.25, abs=1e-12)
    assert w[None] * 3 == test_utils.approx(7, abs=1e-12)


def mat_equal(A, B, tol=1e-6):
    return np.max(np.abs(A - B)) < tol


def _test_svd(dt, n):
    print(
        f"arch={qd.lang.impl.current_cfg().arch} default_fp={qd.lang.impl.current_cfg().default_fp} fast_math={qd.lang.impl.current_cfg().fast_math} dim={n}"
    )
    A = qd.Matrix.field(n, n, dtype=dt, shape=())
    A_reconstructed = qd.Matrix.field(n, n, dtype=dt, shape=())
    U = qd.Matrix.field(n, n, dtype=dt, shape=())
    UtU = qd.Matrix.field(n, n, dtype=dt, shape=())
    sigma = qd.Matrix.field(n, n, dtype=dt, shape=())
    V = qd.Matrix.field(n, n, dtype=dt, shape=())
    VtV = qd.Matrix.field(n, n, dtype=dt, shape=())

    @qd.kernel
    def run():
        U[None], sigma[None], V[None] = qd.svd(A[None], dt)
        UtU[None] = U[None].transpose() @ U[None]
        VtV[None] = V[None].transpose() @ V[None]
        A_reconstructed[None] = U[None] @ sigma[None] @ V[None].transpose()

    if n == 3:
        A[None] = [[1, 1, 3], [9, -3, 2], [-3, 4, 2]]
    else:
        A[None] = [[1, 1], [2, 3]]

    run()

    tol = 1e-5 if dt == qd.f32 else 1e-12

    assert mat_equal(UtU.to_numpy(), np.eye(n), tol=tol)
    assert mat_equal(VtV.to_numpy(), np.eye(n), tol=tol)
    assert mat_equal(A_reconstructed.to_numpy(), A.to_numpy(), tol=tol)
    for i in range(n):
        for j in range(n):
            if i != j:
                assert sigma[None][i, j] == test_utils.approx(0)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_svd_f32(dim):
    _test_svd(qd.f32, dim)


@pytest.mark.parametrize("dim", [2, 3])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_svd_f64(dim):
    _test_svd(qd.f64, dim)


@test_utils.test()
def test_transpose_no_loop():
    A = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())
    U = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())
    sigma = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())
    V = qd.Matrix.field(3, 3, dtype=qd.f32, shape=())

    @qd.kernel
    def run():
        U[None], sigma[None], V[None] = qd.svd(A[None])

    run()
    # As long as it passes compilation we are good
