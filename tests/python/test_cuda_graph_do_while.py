import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _cuda_graph_cache_size():
    return impl.get_runtime().prog.get_cuda_graph_cache_size()


def _cuda_graph_used():
    return impl.get_runtime().prog.get_cuda_graph_cache_used_on_last_call()


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _skip_nested_if_gfx():
    arch = impl.current_cfg().arch
    if arch in (qd.metal, qd.vulkan):
        pytest.skip("Nested graph_do_while not supported on gfx backend")


def _xfail_if_cuda_without_hopper():
    if _on_cuda() and qd.lang.impl.get_cuda_compute_capability() < 90:
        pytest.xfail("graph_do_while requires SM 9.0+ (Hopper)")


@test_utils.test()
def test_graph_do_while_counter():
    """Test graph_do_while with a counter that decrements each iteration."""
    _xfail_if_cuda_without_hopper()
    N = 64

    @qd.kernel(cuda_graph=True)
    def graph_loop(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(counter):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for i in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    graph_loop(x, counter)
    if _on_cuda():
        assert _cuda_graph_used()
        assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(10, dtype=np.int32))

    graph_loop(x, counter)
    if _on_cuda():
        assert _cuda_graph_used()
        assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 10, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_boolean_done():
    """Test graph_do_while with a boolean 'continue' flag (non-zero = keep going)."""
    _xfail_if_cuda_without_hopper()
    N = 64

    @qd.kernel(cuda_graph=True)
    def increment_until_threshold(
        x: qd.types.ndarray(qd.i32, ndim=1),
        threshold: qd.i32,
        keep_going: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(keep_going):
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
    if _on_cuda():
        assert _cuda_graph_used()
        assert _cuda_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_threshold(x, 12, keep_going)
    if _on_cuda():
        assert _cuda_graph_used()
        assert _cuda_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 12, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_multiple_loops():
    """Test graph_do_while with multiple top-level loops in the kernel body."""
    _xfail_if_cuda_without_hopper()
    N = 32

    @qd.kernel(cuda_graph=True)
    def multi_loop(
        x: qd.types.ndarray(qd.f32, ndim=1),
        y: qd.types.ndarray(qd.f32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
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
    if _on_cuda():
        assert _cuda_graph_used()
        assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 10.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 20.0))

    x.from_numpy(np.zeros(N, dtype=np.float32))
    y.from_numpy(np.zeros(N, dtype=np.float32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    multi_loop(x, y, counter)
    if _on_cuda():
        assert _cuda_graph_used()
        assert _cuda_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 5.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 10.0))


@test_utils.test()
def test_graph_do_while_nested_2_levels():
    """Test nested graph_do_while: outer iterates 3x, inner iterates 2x per outer."""
    _skip_nested_if_gfx()
    _xfail_if_cuda_without_hopper()
    N = 16

    @qd.kernel(cuda_graph=True)
    def nested_loop(
        x: qd.types.ndarray(qd.f32, ndim=1),
        outer_c: qd.types.ndarray(qd.i32, ndim=0),
        inner_c: qd.types.ndarray(qd.i32, ndim=0),
        inner_iters: qd.i32,
    ):
        while qd.graph_do_while(outer_c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1.0
            while qd.graph_do_while(inner_c):
                for j in range(x.shape[0]):
                    x[j] = x[j] + 10.0
                for k in range(1):
                    inner_c[()] = inner_c[()] - 1
            for m in range(1):
                inner_c[()] = inner_iters
                outer_c[()] = outer_c[()] - 1

    x = qd.ndarray(qd.f32, shape=(N,))
    outer_c = qd.ndarray(qd.i32, shape=())
    inner_c = qd.ndarray(qd.i32, shape=())

    outer_iters = 3
    inner_iters = 2
    x.from_numpy(np.zeros(N, dtype=np.float32))
    outer_c.from_numpy(np.array(outer_iters, dtype=np.int32))
    inner_c.from_numpy(np.array(inner_iters, dtype=np.int32))

    nested_loop(x, outer_c, inner_c, inner_iters)

    if _on_cuda():
        assert _cuda_graph_used()
        assert _cuda_graph_cache_size() == 1

    assert outer_c.to_numpy() == 0
    expected_x = outer_iters * (1.0 + inner_iters * 10.0)
    np.testing.assert_allclose(x.to_numpy(), np.full(N, expected_x))


@test_utils.test()
def test_graph_do_while_nested_3_levels():
    """Test 3 levels of nesting."""
    _skip_nested_if_gfx()
    _xfail_if_cuda_without_hopper()

    @qd.kernel(cuda_graph=True)
    def triple_nested(
        x: qd.types.ndarray(qd.i32, ndim=1),
        a: qd.types.ndarray(qd.i32, ndim=0),
        b: qd.types.ndarray(qd.i32, ndim=0),
        c: qd.types.ndarray(qd.i32, ndim=0),
        b_reset: qd.i32,
        c_reset: qd.i32,
    ):
        while qd.graph_do_while(a):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            while qd.graph_do_while(b):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 10
                while qd.graph_do_while(c):
                    for i in range(x.shape[0]):
                        x[i] = x[i] + 100
                    for i in range(1):
                        c[()] = c[()] - 1
                for i in range(1):
                    c[()] = c_reset
                    b[()] = b[()] - 1
            for i in range(1):
                b[()] = b_reset
                a[()] = a[()] - 1

    N = 8
    x = qd.ndarray(qd.i32, shape=(N,))
    a = qd.ndarray(qd.i32, shape=())
    b = qd.ndarray(qd.i32, shape=())
    c = qd.ndarray(qd.i32, shape=())

    a_iters, b_iters, c_iters = 2, 3, 2
    x.from_numpy(np.zeros(N, dtype=np.int32))
    a.from_numpy(np.array(a_iters, dtype=np.int32))
    b.from_numpy(np.array(b_iters, dtype=np.int32))
    c.from_numpy(np.array(c_iters, dtype=np.int32))

    triple_nested(x, a, b, c, b_iters, c_iters)

    if _on_cuda():
        assert _cuda_graph_used()

    assert a.to_numpy() == 0
    expected = a_iters * (1 + b_iters * (10 + c_iters * 100))
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, expected, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_not_allowed_without_cuda_graph():
    """Using graph_do_while without cuda_graph=True should raise."""
    from quadrants.lang.exception import QuadrantsSyntaxError

    @qd.kernel
    def bad_kernel(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    c.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(QuadrantsSyntaxError):
        bad_kernel(x, c)


@test_utils.test()
def test_graph_do_while_nonexistent_arg():
    """Using a variable not in kernel params should raise."""
    from quadrants.lang.exception import QuadrantsSyntaxError

    @qd.kernel(cuda_graph=True)
    def bad_kernel2(x: qd.types.ndarray(qd.i32, ndim=1)):
        while qd.graph_do_while(nonexistent):  # noqa: F821
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    with pytest.raises(QuadrantsSyntaxError):
        bad_kernel2(x)


@test_utils.test(arch=[qd.cuda])
def test_graph_do_while_changed_condition_ndarray_raises():
    """Passing a different ndarray for the condition parameter should raise."""
    _xfail_if_cuda_without_hopper()

    @qd.kernel(cuda_graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c):
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
