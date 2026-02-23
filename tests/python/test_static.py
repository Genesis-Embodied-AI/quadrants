import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("val", [0, 1])
@test_utils.test(qd.cpu)
def test_static_if(val):
    x = qd.field(qd.i32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def static():
        if qd.static(val > 0.5):
            x[0] = 1
        else:
            x[0] = 0

    static()
    assert x[0] == val


@test_utils.test(qd.cpu, print_full_traceback=False)
def test_static_if_error():
    x = qd.field(qd.i32)

    qd.root.dense(qd.i, 1).place(x)

    @qd.kernel
    def static(val: float):
        if qd.static(val > 0.5):
            x[0] = 1
        else:
            x[0] = 0

    with pytest.raises(qd.QuadrantsCompilationError, match="must be compile-time constants"):
        static(42)


@test_utils.test()
def test_static_ndrange():
    n = 3
    x = qd.Matrix.field(n, n, dtype=qd.f32, shape=(n, n))

    @qd.kernel
    def fill():
        w = [0, 1, 2]
        for i, j in qd.static(qd.ndrange(3, 3)):
            x[i, j][i, j] = w[i] + w[j] * 2

    fill()
    for i in range(3):
        for j in range(3):
            assert x[i, j][i, j] == i + j * 2


@test_utils.test(qd.cpu)
def test_static_break():
    x = qd.field(qd.i32, 5)

    @qd.kernel
    def func():
        for i in qd.static(range(5)):
            x[i] = 1
            if qd.static(i == 2):
                break

    func()

    assert np.allclose(x.to_numpy(), np.array([1, 1, 1, 0, 0]))


@test_utils.test(qd.cpu)
def test_static_continue():
    x = qd.field(qd.i32, 5)

    @qd.kernel
    def func():
        for i in qd.static(range(5)):
            if qd.static(i == 2):
                continue
            x[i] = 1

    func()

    assert np.allclose(x.to_numpy(), np.array([1, 1, 0, 1, 1]))
