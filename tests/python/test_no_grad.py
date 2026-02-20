import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_no_grad():
    x = qd.field(qd.f32)
    loss = qd.field(qd.f32)

    N = 1

    # no gradients allocated for x
    qd.root.dense(qd.i, N).place(x)
    qd.root.place(loss, loss.grad)

    @qd.kernel
    def func():
        for i in range(N):
            qd.atomic_add(loss[None], x[i] ** 2)

    with qd.ad.Tape(loss):
        func()


@test_utils.test(print_full_traceback=False)
def test_raise_no_gradient():
    y = qd.field(shape=(), name="y", dtype=qd.f32, needs_grad=True)
    x = qd.field(shape=(), name="x", dtype=qd.f32)
    z = np.array([1.0])

    @qd.kernel
    def func(x: qd.template()):
        y[None] = x.grad[None] * x.grad[None]
        z[0] = x.grad[None]

    x[None] = 5.0
    with pytest.raises(
        qd.QuadrantsCompilationError,
        match="Gradient x.grad has not been placed, check whether `needs_grad=True`",
    ):
        func(x)
