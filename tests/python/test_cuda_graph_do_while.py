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
            counter[None] = counter[None] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    graph_loop(x, counter)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_boolean_done():
    """Test graph_do_while with a boolean 'continue' flag (non-zero = keep going)."""
    N = 64
    threshold = 7

    @qd.kernel(graph_do_while="keep_going")
    def increment_until_threshold(x: qd.types.ndarray(qd.i32, ndim=1), keep_going: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            if x[0] >= threshold:
                keep_going[None] = 0

    x = qd.ndarray(qd.i32, shape=(N,))
    keep_going = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_threshold(x, keep_going)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, threshold, dtype=np.int32))


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
            counter[None] = counter[None] - 1

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


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_replay():
    """Test that graph_do_while works correctly on subsequent calls (graph replay)."""
    N = 16

    @qd.kernel(graph_do_while="counter")
    def inc(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[None] = counter[None] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    # First call: 3 iterations
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(3, dtype=np.int32))
    inc(x, counter)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))

    # Second call: 7 iterations (graph replay with new counter value)
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(7, dtype=np.int32))
    inc(x, counter)
    assert _cuda_graph_used()
    assert _cuda_graph_cache_size() == 1
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_replay_new_ndarray_raises():
    """Passing a different ndarray for the condition parameter should raise."""
    N = 16

    @qd.kernel(graph_do_while="counter")
    def inc(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[None] = counter[None] - 1

    x = qd.ndarray(qd.i32, shape=(N,))

    counter1 = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter1.from_numpy(np.array(3, dtype=np.int32))
    inc(x, counter1)
    assert _cuda_graph_used()

    counter2 = qd.ndarray(qd.i32, shape=())
    counter2.from_numpy(np.array(5, dtype=np.int32))
    with pytest.raises(RuntimeError, match="condition ndarray changed"):
        inc(x, counter2)
