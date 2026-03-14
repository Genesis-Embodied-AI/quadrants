import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _cuda_graph_cache_size():
    return impl.get_runtime().prog.get_cuda_graph_cache_size()


def _cuda_graph_used():
    return impl.get_runtime().prog.get_cuda_graph_cache_used_on_last_call()


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_counter():
    """Test graph_do_while with a counter that decrements each iteration."""
    N = 64

    @qd.kernel(graph_do_while="counter")
    def graph_loop(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    graph_loop(x, counter)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(10, dtype=np.int32))

    graph_loop(x, counter)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 10, dtype=np.int32))


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_boolean_done():
    """Test graph_do_while with a boolean 'continue' flag (non-zero = keep going)."""
    N = 64

    @qd.kernel(graph_do_while="keep_going")
    def increment_until_threshold(
        x: qd.types.ndarray(qd.i32, ndim=1),
        threshold: qd.i32,
        keep_going: qd.types.ndarray(qd.i32, ndim=0),
    ):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            if x[0] >= threshold:
                keep_going[()] = 0

    x = qd.ndarray(qd.i32, shape=(N,))
    keep_going = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_threshold(x, 7, keep_going)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_threshold(x, 12, keep_going)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 12, dtype=np.int32))


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_multiple_loops():
    """Test graph_do_while with multiple top-level loops in the kernel body."""
    N = 32

    @qd.kernel(graph_do_while="counter")
    def multi_loop(
        x: qd.types.ndarray(qd.f32, ndim=1),
        y: qd.types.ndarray(qd.f32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
    ):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0
        for i in range(1):
            counter[()] = counter[()] - 1

    x = qd.ndarray(qd.f32, shape=(N,))
    y = qd.ndarray(qd.f32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.float32))
    y.from_numpy(np.zeros(N, dtype=np.float32))
    counter.from_numpy(np.array(10, dtype=np.int32))

    multi_loop(x, y, counter)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 10.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 20.0))

    x.from_numpy(np.zeros(N, dtype=np.float32))
    y.from_numpy(np.zeros(N, dtype=np.float32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    multi_loop(x, y, counter)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 5.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 10.0))


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_changed_condition_ndarray_raises():
    """Passing a different ndarray for the condition parameter should raise."""

    @qd.kernel(graph_do_while="c")
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            c[()] = c[()] - 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c1 = qd.ndarray(qd.i32, shape=())
    c1.from_numpy(np.array(1, dtype=np.int32))
    k(x, c1)

    c2 = qd.ndarray(qd.i32, shape=())
    c2.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(RuntimeError, match="condition ndarray changed"):
        k(x, c2)
