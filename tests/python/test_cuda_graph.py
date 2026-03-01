import os

os.environ["QD_CUDA_GRAPH"] = "1"

import numpy as np

import quadrants as qd
from tests import test_utils


@test_utils.test(arch=[qd.cuda])
def test_cuda_graph_two_loops():
    """A kernel with two top-level for loops should be fused into a CUDA graph."""
    n = 1024
    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def two_loops(x: qd.types.ndarray(qd.f32, ndim=1),
                  y: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    two_loops(x, y)
    two_loops(x, y)
    two_loops(x, y)

    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert np.allclose(x_np, 3.0), f"Expected 3.0, got {x_np[:5]}"
    assert np.allclose(y_np, 6.0), f"Expected 6.0, got {y_np[:5]}"


@test_utils.test(arch=[qd.cuda])
def test_cuda_graph_three_loops():
    """A kernel with three top-level for loops."""
    n = 512
    a = qd.ndarray(qd.f32, shape=(n,))
    b = qd.ndarray(qd.f32, shape=(n,))
    c = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def three_loops(a: qd.types.ndarray(qd.f32, ndim=1),
                    b: qd.types.ndarray(qd.f32, ndim=1),
                    c: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(a.shape[0]):
            a[i] = a[i] + 1.0
        for i in range(b.shape[0]):
            b[i] = b[i] + 10.0
        for i in range(c.shape[0]):
            c[i] = a[i] + b[i]

    three_loops(a, b, c)

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    c_np = c.to_numpy()
    assert np.allclose(a_np, 1.0)
    assert np.allclose(b_np, 10.0)
    assert np.allclose(c_np, 11.0)

    three_loops(a, b, c)

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    c_np = c.to_numpy()
    assert np.allclose(a_np, 2.0)
    assert np.allclose(b_np, 20.0)
    assert np.allclose(c_np, 22.0)


@test_utils.test(arch=[qd.cuda])
def test_cuda_graph_single_loop_no_graph():
    """A kernel with a single for loop should NOT use the graph path."""
    n = 256
    x = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def single_loop(x: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 5.0

    single_loop(x)
    single_loop(x)

    x_np = x.to_numpy()
    assert np.allclose(x_np, 10.0)
