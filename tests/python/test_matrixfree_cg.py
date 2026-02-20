import math

import pytest

import quadrants as qd
from quadrants.linalg import LinearOperator, MatrixFreeCG

from tests import test_utils

vk_on_mac = (qd.vulkan, "Darwin")


@pytest.mark.parametrize("ti_dtype", [qd.f32, qd.f64])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], exclude=[vk_on_mac])
def test_matrixfree_cg(ti_dtype):
    GRID = 32
    Ax = qd.field(dtype=ti_dtype, shape=(GRID, GRID))
    x = qd.field(dtype=ti_dtype, shape=(GRID, GRID))
    b = qd.field(dtype=ti_dtype, shape=(GRID, GRID))

    @qd.kernel
    def init():
        for i, j in qd.ndrange(GRID, GRID):
            xl = i / (GRID - 1)
            yl = j / (GRID - 1)
            b[i, j] = qd.sin(2 * math.pi * xl) * qd.sin(2 * math.pi * yl)
            x[i, j] = 0.0

    @qd.kernel
    def compute_Ax(v: qd.template(), mv: qd.template()):
        for i, j in v:
            l = v[i - 1, j] if i - 1 >= 0 else 0.0
            r = v[i + 1, j] if i + 1 <= GRID - 1 else 0.0
            t = v[i, j + 1] if j + 1 <= GRID - 1 else 0.0
            b = v[i, j - 1] if j - 1 >= 0 else 0.0
            # Avoid ill-conditioned matrix A
            mv[i, j] = 20 * v[i, j] - l - r - t - b

    @qd.kernel
    def check_solution(sol: qd.template(), ans: qd.template(), tol: ti_dtype) -> bool:
        exit_code = True
        for i, j in qd.ndrange(GRID, GRID):
            if qd.abs(ans[i, j] - sol[i, j]) < tol:
                pass
            else:
                exit_code = False
        return exit_code

    A = LinearOperator(compute_Ax)
    init()
    MatrixFreeCG(A, b, x, maxiter=10 * GRID * GRID, tol=1e-18, quiet=True)
    compute_Ax(x, Ax)
    # `tol` can't be < 1e-6 for qd.f32 because of accumulating round-off error;
    # see https://en.wikipedia.org/wiki/Conjugate_gradient_method#cite_note-6
    # for more details.
    result = check_solution(Ax, b, tol=1e-6)
    assert result
