import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_graph_while_counter_cross_backend():
    """graph_while with a counter: must work identically on CPU and CUDA."""
    N = 64
    ITERS = 5

    @qd.kernel(graph_while="counter")
    def increment_loop(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[None] = counter[None] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(ITERS, dtype=np.int32))

    increment_loop(x, counter)
    qd.sync()

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, ITERS, dtype=np.int32))


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_graph_while_boolean_reduction_cross_backend():
    """graph_while with per-thread conditions reduced into a single flag.

    Each element has a different threshold. The loop continues while ANY element
    hasn't reached its threshold. A reduction kernel (reset flag to 0, then
    any-not-done sets it to 1) combines per-element state into the scalar flag.
    """
    N = 32

    @qd.kernel(graph_while="keep_going")
    def increment_until_all_done(
        x: qd.types.ndarray(qd.i32, ndim=1),
        thresholds: qd.types.ndarray(qd.i32, ndim=1),
        keep_going: qd.types.ndarray(qd.i32, ndim=0),
    ):
        # Work: increment elements that haven't reached their threshold
        for i in range(x.shape[0]):
            if x[i] < thresholds[i]:
                x[i] = x[i] + 1

        # Reduction: reset flag, then OR-reduce per-element conditions
        for i in range(1):
            keep_going[None] = 0
        for i in range(x.shape[0]):
            if x[i] < thresholds[i]:
                keep_going[None] = 1

    x = qd.ndarray(qd.i32, shape=(N,))
    thresholds = qd.ndarray(qd.i32, shape=(N,))
    keep_going = qd.ndarray(qd.i32, shape=())

    # Thresholds vary: 1, 2, 3, ..., N. Loop must run N times (max threshold).
    thresh_np = np.arange(1, N + 1, dtype=np.int32)
    x.from_numpy(np.zeros(N, dtype=np.int32))
    thresholds.from_numpy(thresh_np)
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_all_done(x, thresholds, keep_going)
    qd.sync()

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), thresh_np)


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_graph_while_multi_loop_cross_backend():
    """graph_while with multiple top-level for loops in the body."""
    N = 16
    ITERS = 8

    @qd.kernel(graph_while="counter")
    def multi_loop(
        a: qd.types.ndarray(qd.f32, ndim=1),
        b: qd.types.ndarray(qd.f32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
    ):
        for i in range(a.shape[0]):
            a[i] = a[i] + 1.0
        for i in range(b.shape[0]):
            b[i] = b[i] + 3.0
        for i in range(1):
            counter[None] = counter[None] - 1

    a = qd.ndarray(qd.f32, shape=(N,))
    b = qd.ndarray(qd.f32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    a.from_numpy(np.zeros(N, dtype=np.float32))
    b.from_numpy(np.zeros(N, dtype=np.float32))
    counter.from_numpy(np.array(ITERS, dtype=np.int32))

    multi_loop(a, b, counter)
    qd.sync()

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(a.to_numpy(), np.full(N, float(ITERS)))
    np.testing.assert_allclose(b.to_numpy(), np.full(N, float(ITERS * 3)))


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_graph_while_replay_cross_backend():
    """graph_while replay: second call with different counter value."""
    N = 16

    @qd.kernel(graph_while="counter")
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
    qd.sync()
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))
    assert counter.to_numpy() == 0

    # Second call: 7 iterations
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(7, dtype=np.int32))
    inc(x, counter)
    qd.sync()
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))
    assert counter.to_numpy() == 0


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_graph_while_single_iteration():
    """graph_while with counter=1 executes the body exactly once.

    Note: graph_while has do-while semantics (body executes at least once,
    matching CUDA conditional while node behavior). Counter must be >= 1.
    """
    N = 8

    @qd.kernel(graph_while="counter")
    def inc(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[None] = counter[None] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(1, dtype=np.int32))

    inc(x, counter)
    qd.sync()

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 1, dtype=np.int32))
