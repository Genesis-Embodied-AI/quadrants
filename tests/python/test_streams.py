"""Tests for GPU stream and event support."""

import numpy as np

import quadrants as qd
from quadrants.lang.stream import Event, Stream

from tests import test_utils


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_create_and_destroy_stream():
    s = qd.create_stream()
    assert isinstance(s, Stream)
    assert s.handle != 0
    s.destroy()
    assert s.handle == 0


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_create_and_destroy_event():
    e = qd.create_event()
    assert isinstance(e, Event)
    assert e.handle != 0
    e.destroy()
    assert e.handle == 0


@test_utils.test()
def test_kernel_on_stream():
    N = 1024
    x = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def fill():
        for i in range(N):
            x[i] = 42.0

    s = qd.create_stream()
    fill(qd_stream=s)
    s.synchronize()
    assert np.allclose(x.to_numpy(), 42.0)
    s.destroy()


@test_utils.test()
def test_two_streams():
    N = 1024
    a = qd.field(qd.f32, shape=(N,))
    b = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def fill_a():
        for i in range(N):
            a[i] = 1.0

    @qd.kernel
    def fill_b():
        for i in range(N):
            b[i] = 2.0

    s1 = qd.create_stream()
    s2 = qd.create_stream()
    fill_a(qd_stream=s1)
    fill_b(qd_stream=s2)
    s1.synchronize()
    s2.synchronize()
    assert np.allclose(a.to_numpy(), 1.0)
    assert np.allclose(b.to_numpy(), 2.0)
    s1.destroy()
    s2.destroy()


@test_utils.test()
def test_event_synchronization():
    N = 1024
    x = qd.field(qd.f32, shape=(N,))
    y = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def fill_x():
        for i in range(N):
            x[i] = 10.0

    @qd.kernel
    def copy_x_to_y():
        for i in range(N):
            y[i] = x[i]

    s1 = qd.create_stream()
    fill_x(qd_stream=s1)

    e = qd.create_event()
    e.record(s1)

    # Default stream waits for s1 to finish fill_x
    e.wait()
    copy_x_to_y()
    qd.sync()

    assert np.allclose(y.to_numpy(), 10.0)

    e.destroy()
    s1.destroy()


@test_utils.test()
def test_event_wait_on_stream():
    N = 1024
    x = qd.field(qd.f32, shape=(N,))
    y = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def fill_x():
        for i in range(N):
            x[i] = 5.0

    @qd.kernel
    def copy_x_to_y():
        for i in range(N):
            y[i] = x[i]

    s1 = qd.create_stream()
    s2 = qd.create_stream()

    fill_x(qd_stream=s1)

    e = qd.create_event()
    e.record(s1)

    # s2 waits for s1's event before running
    e.wait(qd_stream=s2)
    copy_x_to_y(qd_stream=s2)
    s2.synchronize()

    assert np.allclose(y.to_numpy(), 5.0)

    e.destroy()
    s1.destroy()
    s2.destroy()


@test_utils.test()
def test_default_stream_kernel():
    N = 1024
    x = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def fill():
        for i in range(N):
            x[i] = 7.0

    fill()
    qd.sync()
    assert np.allclose(x.to_numpy(), 7.0)


@test_utils.test(arch=[qd.cpu])
def test_stream_noop_on_cpu():
    """Streams should be no-ops on CPU without errors."""
    N = 64
    x = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def fill():
        for i in range(N):
            x[i] = 3.0

    s = qd.create_stream()
    assert s.handle == 0
    fill(qd_stream=s)
    qd.sync()
    assert np.allclose(x.to_numpy(), 3.0)

    e = qd.create_event()
    assert e.handle == 0
    e.record(s)
    e.wait()
    s.destroy()
    e.destroy()


@test_utils.test()
def test_concurrent_streams_with_events():
    """Two slow kernels on separate streams run concurrently (~1s on GPU),
    serial fallback on CPU/Metal."""
    SPIN_ITERS = 40_000_000

    @qd.kernel
    def slow_fill(
        a: qd.types.ndarray(dtype=qd.f32, ndim=1),
        lcg_state: qd.types.ndarray(dtype=qd.i32, ndim=1),
        index: qd.i32,
        value: qd.f32,
    ):
        qd.loop_config(block_dim=1)
        for _ in range(1):
            x = lcg_state[index]
            for _j in range(SPIN_ITERS):
                x = (1664525 * x + 1013904223) % 2147483647
            lcg_state[index] = x
            a[index] = value

    @qd.kernel
    def add_first_two(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        qd.loop_config(block_dim=1)
        for _ in range(1):
            a[2] = a[0] + a[1]

    import time

    # Warm up JIT
    a_warmup = qd.ndarray(qd.f32, shape=(3,))
    lcg_warmup = qd.ndarray(qd.i32, shape=(3,))
    slow_fill(a_warmup, lcg_warmup, 0, 0.0)
    add_first_two(a_warmup)
    qd.sync()

    # Serial baseline
    a = qd.ndarray(qd.f32, shape=(3,))
    lcg = qd.ndarray(qd.i32, shape=(3,))
    qd.sync()
    t0 = time.perf_counter()
    slow_fill(a, lcg, 0, 5.0)
    slow_fill(a, lcg, 1, 7.0)
    add_first_two(a)
    qd.sync()
    serial_time = time.perf_counter() - t0
    assert np.isclose(a.to_numpy()[2], 12.0)

    # Streams
    a = qd.ndarray(qd.f32, shape=(3,))
    lcg = qd.ndarray(qd.i32, shape=(3,))
    s1 = qd.create_stream()
    s2 = qd.create_stream()
    e1 = qd.create_event()
    e2 = qd.create_event()
    qd.sync()
    t0 = time.perf_counter()
    slow_fill(a, lcg, 0, 5.0, qd_stream=s1)
    slow_fill(a, lcg, 1, 7.0, qd_stream=s2)
    e1.record(s1)
    e2.record(s2)
    e1.wait()
    e2.wait()
    add_first_two(a)
    qd.sync()
    stream_time = time.perf_counter() - t0
    assert np.isclose(a.to_numpy()[2], 12.0)

    speedup = serial_time / stream_time
    if qd.lang.impl.current_cfg().arch in (qd.cuda, qd.amdgpu):
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"
    else:
        assert speedup > 0.75, f"Expected >=0.75x (serial fallback), got {speedup:.2f}x"

    s1.destroy()
    s2.destroy()
    e1.destroy()
    e2.destroy()


@test_utils.test()
def test_stream_parallel_basic():
    """Each with qd.stream_parallel() block runs on its own stream (serial fallback on CPU/Metal)."""
    N = 1024
    a = qd.field(qd.f32, shape=(N,))
    b = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def fill_parallel():
        with qd.stream_parallel():
            for i in range(N):
                a[i] = 1.0
        with qd.stream_parallel():
            for j in range(N):
                b[j] = 2.0

    fill_parallel()
    qd.sync()
    assert np.allclose(a.to_numpy(), 1.0)
    assert np.allclose(b.to_numpy(), 2.0)


@test_utils.test()
def test_stream_parallel_multiple_loops_per_stream():
    """Multiple for loops inside one stream_parallel block share a stream (serial fallback on CPU/Metal)."""
    N = 1024
    a = qd.field(qd.f32, shape=(N,))
    b = qd.field(qd.f32, shape=(N,))
    c = qd.field(qd.f32, shape=(N,))

    @qd.kernel
    def parallel_phase():
        with qd.stream_parallel():
            for i in range(N):
                a[i] = 1.0
            for i in range(N):
                a[i] = a[i] + 1.0
        with qd.stream_parallel():
            for j in range(N):
                b[j] = 10.0

    @qd.kernel
    def combine():
        for i in range(N):
            c[i] = a[i] + b[i]

    parallel_phase()
    combine()
    qd.sync()
    assert np.allclose(a.to_numpy(), 2.0)
    assert np.allclose(b.to_numpy(), 10.0)
    assert np.allclose(c.to_numpy(), 12.0)


@test_utils.test()
def test_stream_parallel_timing():
    """stream_parallel achieves speedup on GPU, serial fallback elsewhere."""
    SPIN_ITERS = 40_000_000

    a = qd.field(qd.i32, shape=(2,))
    b = qd.field(qd.i32, shape=(2,))

    @qd.kernel
    def serial_spin():
        for _ in range(1):
            x = a[0]
            for _j in range(SPIN_ITERS):
                x = (1664525 * x + 1013904223) % 2147483647
            a[0] = x
        for _ in range(1):
            x = a[1]
            for _j in range(SPIN_ITERS):
                x = (1664525 * x + 1013904223) % 2147483647
            a[1] = x

    @qd.kernel
    def parallel_spin():
        with qd.stream_parallel():
            for _ in range(1):
                x = b[0]
                for _j in range(SPIN_ITERS):
                    x = (1664525 * x + 1013904223) % 2147483647
                b[0] = x
        with qd.stream_parallel():
            for _ in range(1):
                x = b[1]
                for _j in range(SPIN_ITERS):
                    x = (1664525 * x + 1013904223) % 2147483647
                b[1] = x

    import time

    # Warm up
    serial_spin()
    parallel_spin()
    qd.sync()

    qd.sync()
    t0 = time.perf_counter()
    serial_spin()
    qd.sync()
    serial_time = time.perf_counter() - t0

    qd.sync()
    t0 = time.perf_counter()
    parallel_spin()
    qd.sync()
    stream_time = time.perf_counter() - t0

    speedup = serial_time / stream_time
    if qd.lang.impl.current_cfg().arch in (qd.cuda, qd.amdgpu):
        assert speedup > 1.5, (
            f"Expected >1.5x speedup, got {speedup:.2f}x " f"(serial={serial_time:.3f}s, stream={stream_time:.3f}s)"
        )
    else:
        assert speedup > 0.75, (
            f"Expected >=0.75x (serial fallback), got {speedup:.2f}x "
            f"(serial={serial_time:.3f}s, stream={stream_time:.3f}s)"
        )


@test_utils.test()
def test_stream_parallel_rejects_mixed_top_level():
    """Mixing stream_parallel and non-stream_parallel at top level is an error."""
    import pytest  # noqa: I001

    from quadrants.lang.exception import QuadrantsSyntaxError

    N = 64
    a = qd.field(qd.f32, shape=(N,))

    with pytest.raises(QuadrantsSyntaxError, match="all top-level statements"):

        @qd.kernel
        def bad_kernel():
            with qd.stream_parallel():
                for i in range(N):
                    a[i] = 1.0
            for i in range(N):
                a[i] = 2.0

        bad_kernel()


@test_utils.test()
def test_stream_with_ndarray():
    N = 1024

    @qd.kernel
    def fill(arr: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(N):
            arr[i] = 99.0

    arr = qd.ndarray(qd.f32, shape=(N,))
    s = qd.create_stream()
    fill(arr, qd_stream=s)
    s.synchronize()
    assert np.allclose(arr.to_numpy(), 99.0)
    s.destroy()
