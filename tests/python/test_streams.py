"""Tests for GPU stream and event support."""

import numpy as np

import quadrants as qd
from quadrants.lang.stream import Stream, Event

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


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
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


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
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


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
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


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
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


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
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


@test_utils.test(arch=[qd.cuda, qd.amdgpu])
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
