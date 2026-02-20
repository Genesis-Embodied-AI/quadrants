import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.adstack)
def test_loop_grad():
    x = qd.field(qd.f32)

    n = 16
    m = 8

    qd.root.dense(qd.ij, (n, m)).place(x)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for k in range(n):
            for i in range(m - 1):
                x[k, i + 1] = x[k, i] * 2

    for k in range(n):
        x[k, 0] = k
    func()

    for k in range(n):
        x.grad[k, m - 1] = 1
    func.grad()

    for k in range(n):
        # The grad of fields on left-hand sides of assignments (GlobalStoreStmt) need to be reset to zero after the corresponding adjoint assignments.
        # Therefore, only the grad of the element with index 0 at second dimension is preserved here.
        assert x[k, 0] == 2**0 * k
        assert x.grad[k, 0] == 2 ** (m - 1 - 0)


@test_utils.test(require=qd.extension.adstack)
@pytest.mark.skip(reason="not yet supported")
def test_loop_grad_complex():
    x = qd.field(qd.f32)

    n = 16
    m = 8

    qd.root.dense(qd.ij, (n, m)).place(x)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for k in range(n):
            t = k * k
            tt = t * 2
            for i in range(m - 1):
                x[k, i + 1] = x[k, i] * 2 + tt

    for k in range(n):
        x[k, 0] = k
    func()

    for k in range(n):
        x.grad[k, m - 1] = 1
    func.grad()

    for k in range(n):
        for i in range(m):
            assert x[k, i] == i**2 + 2 * k**2
            assert x.grad[k, i] == 2 ** (m - 1 - i)
