import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def _solve_vector_equal(v1, v2, tol):
    if np.linalg.norm(v1) == 0.0:
        assert np.linalg.norm(v2) == 0.0
    else:
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        np.testing.assert_allclose(v1, v2, atol=tol, rtol=tol)


def _test_solve_2x2(dt, a00):
    A = qd.Matrix.field(2, 2, dtype=dt, shape=())
    b = qd.Vector.field(2, dtype=dt, shape=())
    x = qd.Vector.field(2, dtype=dt, shape=())

    @qd.kernel
    def solve_2x2():
        A[None] = qd.Matrix([[a00, 1.0], [1.0, 1.001]])
        b[None] = qd.Vector([3.0, 15.0])
        x[None] = qd.solve(A[None], b[None])

    solve_2x2()

    tol = 1e-5 if dt == qd.f32 else 1e-12
    dtype = np.float32 if dt == qd.f32 else np.float64
    x_np = np.linalg.solve(A[None].to_numpy().astype(dtype), b[None].to_numpy().astype(dtype))
    x_ti = x[None].to_numpy().astype(dtype)

    idx_np = np.argsort(x_np)
    idx_ti = np.argsort(x_ti)
    np.testing.assert_allclose(x_np[idx_np], x_ti[idx_ti], atol=tol, rtol=tol)
    _solve_vector_equal(x_ti, x_np, tol)


def _test_solve_3x3(dt, a00):
    A = qd.Matrix.field(3, 3, dtype=dt, shape=())
    b = qd.Vector.field(3, dtype=dt, shape=())
    x = qd.Vector.field(3, dtype=dt, shape=())

    @qd.kernel
    def solve_3x3():
        A[None] = qd.Matrix([[a00, 2.0, -4.0], [2.0, 3.0, 3.0], [5.0, -3, 1.0]])
        b[None] = qd.Vector([3.0, 15.0, 14.0])
        x[None] = qd.solve(A[None], b[None])

    solve_3x3()

    tol = 1e-5 if dt == qd.f32 else 1e-12
    dtype = np.float32 if dt == qd.f32 else np.float64
    x_np = np.linalg.solve(A[None].to_numpy().astype(dtype), b[None].to_numpy().astype(dtype))
    x_ti = x[None].to_numpy().astype(dtype)

    idx_np = np.argsort(x_np)
    idx_ti = np.argsort(x_ti)
    np.testing.assert_allclose(x_np[idx_np], x_ti[idx_ti], atol=tol, rtol=tol)
    _solve_vector_equal(x_ti, x_np, tol)


@pytest.mark.parametrize("a00", [float(i) for i in range(10)])
@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_solve_2x2_f32(a00):
    _test_solve_2x2(qd.f32, a00)


@pytest.mark.parametrize("a00", [float(i) for i in range(10)])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_solve_2x2_f64(a00):
    _test_solve_2x2(qd.f64, a00)


@pytest.mark.parametrize("a00", [float(i) for i in range(10)])
@test_utils.test(default_fp=qd.f32, fast_math=False)
def test_solve_3x3_f32(a00):
    _test_solve_3x3(qd.f32, a00)


@pytest.mark.parametrize("a00", [float(i) for i in range(10)])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64, fast_math=False)
def test_solve_3x3_f64(a00):
    _test_solve_3x3(qd.f64, a00)
