import importlib
import importlib.util
import sys

from pathlib import Path

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _gpu_graph_cache_size():
    return impl.get_runtime().prog.get_gpu_graph_cache_size()


def _gpu_graph_used():
    return impl.get_runtime().prog.get_gpu_graph_cache_used_on_last_call()


def _gpu_graph_total_builds():
    return impl.get_runtime().prog.get_gpu_graph_total_builds()


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _xfail_if_cuda_without_hopper():
    if _on_cuda() and qd.lang.impl.get_cuda_compute_capability() < 90:
        pytest.xfail("graph_do_while requires SM 9.0+ (Hopper)")


@test_utils.test()
def test_graph_do_while_counter():
    """Test graph_do_while with a counter that decrements each iteration."""
    _xfail_if_cuda_without_hopper()
    N = 64

    @qd.kernel(gpu_graph=True)
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
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(10, dtype=np.int32))

    graph_loop(x, counter)
    if _on_cuda():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 10, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_boolean_done():
    """Test graph_do_while with a boolean 'continue' flag (non-zero = keep going)."""
    _xfail_if_cuda_without_hopper()
    N = 64

    @qd.kernel(gpu_graph=True)
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
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_threshold(x, 12, keep_going)
    if _on_cuda():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 12, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_multiple_loops():
    """Test graph_do_while with multiple top-level loops in the kernel body."""
    _xfail_if_cuda_without_hopper()
    N = 32

    @qd.kernel(gpu_graph=True)
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
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 10.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 20.0))

    x.from_numpy(np.zeros(N, dtype=np.float32))
    y.from_numpy(np.zeros(N, dtype=np.float32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    multi_loop(x, y, counter)
    if _on_cuda():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 5.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 10.0))


@test_utils.test()
def test_graph_do_while_swap_counter_ndarray():
    """Swapping the counter ndarray between calls should work correctly.

    Creates one counter c1, runs the kernel with counter=3, verifies x is all
    3s. Then creates a new ndarray c2 (different device pointer), runs the same
    kernel with counter=7, verifies x is all 7s. Confirms cache size stays 1 --
    the graph wasn't rebuilt, it just updated the indirection slot with c2's
    pointer.
    """
    _xfail_if_cuda_without_hopper()
    N = 32

    @qd.kernel(gpu_graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for i in range(1):
                c[()] = c[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    c1 = qd.ndarray(qd.i32, shape=())

    x.from_numpy(np.zeros(N, dtype=np.int32))
    c1.from_numpy(np.array(3, dtype=np.int32))
    k(x, c1)
    if _on_cuda():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1
    assert c1.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))

    c2 = qd.ndarray(qd.i32, shape=())
    assert c1.arr.device_allocation_ptr() != c2.arr.device_allocation_ptr()
    x.from_numpy(np.zeros(N, dtype=np.int32))
    c2.from_numpy(np.array(7, dtype=np.int32))
    k(x, c2)
    if _on_cuda():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1
        assert _gpu_graph_total_builds() == 1
    assert c2.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_alternate_counter_ndarrays():
    """Alternating between two counter ndarrays should work correctly.

    Creates c1 and c2 upfront, then alternates between them for 3 rounds (6
    kernel calls). Each call uses a different iteration count (count and
    count+10). Confirms the slot update works back and forth, not just as a
    one-time swap. Cache size is checked once at the end -- still 1.
    """
    _xfail_if_cuda_without_hopper()
    N = 16

    @qd.kernel(gpu_graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for i in range(1):
                c[()] = c[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    c1 = qd.ndarray(qd.i32, shape=())
    c2 = qd.ndarray(qd.i32, shape=())
    assert c1.arr.device_allocation_ptr() != c2.arr.device_allocation_ptr()

    for iteration in range(3):
        count = iteration + 2
        x.from_numpy(np.zeros(N, dtype=np.int32))
        c1.from_numpy(np.array(count, dtype=np.int32))
        k(x, c1)
        if _on_cuda():
            assert _gpu_graph_used()
        assert c1.to_numpy() == 0
        np.testing.assert_array_equal(x.to_numpy(), np.full(N, count, dtype=np.int32))

        x.from_numpy(np.zeros(N, dtype=np.int32))
        c2.from_numpy(np.array(count + 10, dtype=np.int32))
        k(x, c2)
        if _on_cuda():
            assert _gpu_graph_used()
        assert c2.to_numpy() == 0
        np.testing.assert_array_equal(x.to_numpy(), np.full(N, count + 10, dtype=np.int32))

    if _on_cuda():
        assert _gpu_graph_cache_size() == 1
        assert _gpu_graph_total_builds() == 1


@test_utils.test()
def test_graph_do_while_without_gpu_graph_raises():
    """Using qd.graph_do_while without gpu_graph=True should raise."""

    @qd.kernel
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    c.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(qd.QuadrantsSyntaxError, match="requires @qd.kernel\\(gpu_graph=True\\)"):
        k(x, c)


@test_utils.test()
def test_graph_do_while_nonexistent_arg_raises():
    """Using a variable name that isn't a kernel parameter should raise."""

    @qd.kernel(gpu_graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(nonexistent):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    c.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(qd.QuadrantsSyntaxError, match="does not match any parameter"):
        k(x, c)


def _import_kernel(filepath: Path, module_name: str, kernel_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, kernel_name)


_FASTCACHE_KERNEL_SRC = """\
import quadrants as qd

@qd.kernel(gpu_graph=True, fastcache=True)
def k(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[()] = counter[()] - 1
"""


@test_utils.test()
def test_graph_do_while_fastcache_restores_arg(tmp_path):
    """After fastcache restore, graph_do_while_arg should be set on the Kernel."""
    N = 16
    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    mod_name = "_test_fastcache_do_while_mod"

    filepath = tmp_path / "k.py"
    filepath.write_text(_FASTCACHE_KERNEL_SRC)

    # First import: compiles and populates the fastcache
    k1 = _import_kernel(filepath, mod_name, "k")
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(5, dtype=np.int32))
    k1(x, counter)

    primal1 = k1._primal
    assert primal1.graph_do_while_arg == "counter"
    assert primal1.src_ll_cache_observations.cache_stored

    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5))
    assert counter.to_numpy() == 0

    # Second import: loads from fastcache (same filepath = same cache key).
    # graph_do_while_arg must be restored from the cached metadata.
    k2 = _import_kernel(filepath, mod_name, "k")
    primal2 = k2._primal

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(3, dtype=np.int32))
    k2(x, counter)

    assert primal2.src_ll_cache_observations.cache_loaded, "fastcache should have been loaded"
    assert primal2.graph_do_while_arg == "counter", (
        "graph_do_while_arg should be restored from fastcache"
    )

    np.testing.assert_array_equal(
        x.to_numpy(), np.full(N, 3),
        err_msg="graph_do_while counter not restored from fastcache",
    )
    assert counter.to_numpy() == 0

    if mod_name in sys.modules:
        del sys.modules[mod_name]
