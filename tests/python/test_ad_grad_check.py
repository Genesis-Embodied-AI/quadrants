import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(default_fp=qd.f64, exclude=[qd.vulkan, qd.metal])
def test_general():
    x1 = qd.field(dtype=float, shape=(2, 2), needs_grad=True)
    y1 = qd.field(dtype=float, shape=(), needs_grad=True)

    x1.from_numpy(np.array([[1, 2], [3, 4]]))

    @qd.kernel
    def compute_y1():
        for i, j in qd.ndrange(2, 2):
            y1[None] += qd.cos(x1[i, j])

    x2 = qd.Vector.field(n=3, dtype=float, shape=(2, 2), needs_grad=True)
    y2 = qd.field(dtype=float, shape=(), needs_grad=True)
    x2[0, 0] = qd.Vector([1, 2, 3])
    x2[0, 1] = qd.Vector([4, 5, 6])
    x2[1, 0] = qd.Vector([7, 8, 9])
    x2[1, 1] = qd.Vector([10, 11, 12])

    @qd.kernel
    def compute_y2():
        y2[None] += x2[0, 0][0] + x2[1, 0][1] + x2[1, 1][2]

    with qd.ad.Tape(y1, grad_check=[x1]):
        compute_y1()

    with qd.ad.Tape(y2, grad_check=[x2]):
        compute_y2()


def grad_test(tifunc):
    print(f"arch={qd.lang.impl.current_cfg().arch} default_fp={qd.lang.impl.current_cfg().default_fp}")
    x = qd.field(qd.lang.impl.current_cfg().default_fp)
    y = qd.field(qd.lang.impl.current_cfg().default_fp)

    qd.root.place(x, x.grad, y, y.grad)

    @qd.kernel
    def func():
        for i in qd.grouped(x):
            y[i] = tifunc(x[i])

    x[None] = 0.234

    with qd.ad.Tape(loss=y, grad_check=[x]):
        func()


@pytest.mark.parametrize(
    "tifunc",
    [
        lambda x: x,
        lambda x: qd.abs(-x),
        lambda x: -x,
        lambda x: x * x,
        lambda x: x**2,
        lambda x: x * x * x,
        lambda x: x * x * x * x,
        lambda x: 0.4 * x * x - 3,
        lambda x: (x - 3) * (x - 1),
        lambda x: (x - 3) * (x - 1) + x * x,
        lambda x: qd.tanh(x),
        lambda x: qd.sin(x),
        lambda x: qd.cos(x),
        lambda x: qd.acos(x),
        lambda x: qd.asin(x),
        lambda x: 1 / x,
        lambda x: (x + 1) / (x - 1),
        lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)),
        lambda x: qd.sqrt(x),
        lambda x: qd.exp(x),
        lambda x: qd.log(x),
        lambda x: qd.min(x, 0),
        lambda x: qd.min(x, 1),
        lambda x: qd.min(0, x),
        lambda x: qd.min(1, x),
        lambda x: qd.max(x, 0),
        lambda x: qd.max(x, 1),
        lambda x: qd.max(0, x),
        lambda x: qd.max(1, x),
        lambda x: qd.atan2(0.4, x),
        lambda x: qd.atan2(x, 0.4),
        lambda x: 0.4**x,
        lambda x: x**0.4,
    ],
)
@test_utils.test(default_fp=qd.f64, exclude=[qd.vulkan, qd.metal])
def test_basics(tifunc):
    grad_test(tifunc)
