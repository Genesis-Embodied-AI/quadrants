import os
import pathlib
import subprocess
import sys

import numpy as np
import pydantic
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils

TEST_RAN = "test ran"
RET_SUCCESS = 42


def _gpu_graph_cache_size():
    return impl.get_runtime().prog.get_gpu_graph_cache_size()


def _gpu_graph_used():
    return impl.get_runtime().prog.get_gpu_graph_cache_used_on_last_call()


def _gpu_graph_total_builds():
    return impl.get_runtime().prog.get_gpu_graph_total_builds()


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _is_gpu_graph_do_while_natively_supported():
    return _on_cuda() and qd.lang.impl.get_cuda_compute_capability() >= 90


@test_utils.test()
def test_graph_do_while_counter():
    """Test graph_do_while with a counter that decrements each iteration."""
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
    if _is_gpu_graph_do_while_natively_supported():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(10, dtype=np.int32))

    graph_loop(x, counter)
    if _is_gpu_graph_do_while_natively_supported():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 10, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_boolean_done():
    """Test graph_do_while with a boolean 'continue' flag (non-zero = keep going)."""
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
    if _is_gpu_graph_do_while_natively_supported():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_threshold(x, 12, keep_going)
    if _is_gpu_graph_do_while_natively_supported():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 12, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_multiple_loops():
    """Test graph_do_while with multiple top-level loops in the kernel body."""
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
    if _is_gpu_graph_do_while_natively_supported():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 10.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 20.0))

    x.from_numpy(np.zeros(N, dtype=np.float32))
    y.from_numpy(np.zeros(N, dtype=np.float32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    multi_loop(x, y, counter)
    if _is_gpu_graph_do_while_natively_supported():
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
    if _is_gpu_graph_do_while_natively_supported():
        assert _gpu_graph_used()
        assert _gpu_graph_cache_size() == 1
    assert c1.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))

    c2 = qd.ndarray(qd.i32, shape=())
    assert c1.arr.device_allocation_ptr() != c2.arr.device_allocation_ptr()
    x.from_numpy(np.zeros(N, dtype=np.int32))
    c2.from_numpy(np.array(7, dtype=np.int32))
    k(x, c2)
    if _is_gpu_graph_do_while_natively_supported():
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
        if _is_gpu_graph_do_while_natively_supported():
            assert _gpu_graph_used()
        assert c1.to_numpy() == 0
        np.testing.assert_array_equal(x.to_numpy(), np.full(N, count, dtype=np.int32))

        x.from_numpy(np.zeros(N, dtype=np.int32))
        c2.from_numpy(np.array(count + 10, dtype=np.int32))
        k(x, c2)
        if _is_gpu_graph_do_while_natively_supported():
            assert _gpu_graph_used()
        assert c2.to_numpy() == 0
        np.testing.assert_array_equal(x.to_numpy(), np.full(N, count + 10, dtype=np.int32))

    if _is_gpu_graph_do_while_natively_supported():
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


@qd.kernel(gpu_graph=True, fastcache=True)
def _fastcache_do_while_kernel(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)):
    while qd.graph_do_while(counter):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(1):
            counter[()] = counter[()] - 1


class _FastcacheDoWhileArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    iterations: int
    expect_loaded_from_fastcache: bool


def _fastcache_do_while_child(args: list[str]) -> None:
    args_obj = _FastcacheDoWhileArgs.model_validate_json(args[0])
    qd.init(
        arch=getattr(qd, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=True,
    )

    N = 16
    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(args_obj.iterations, dtype=np.int32))

    _fastcache_do_while_kernel(x, counter)

    assert (
        _fastcache_do_while_kernel._primal.graph_do_while_arg == "counter"
    ), f"graph_do_while_arg should be 'counter', got {_fastcache_do_while_kernel._primal.graph_do_while_arg!r}"
    assert (
        _fastcache_do_while_kernel._primal.src_ll_cache_observations.cache_loaded
        == args_obj.expect_loaded_from_fastcache
    )
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, args_obj.iterations))
    assert counter.to_numpy() == 0

    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@test_utils.test()
def test_graph_do_while_fastcache_restores_arg(tmp_path: pathlib.Path):
    """After fastcache restore in a fresh process, graph_do_while_arg must be set."""
    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch.name
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    for iterations, expect_loaded in [(5, False), (3, True)]:
        args_obj = _FastcacheDoWhileArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path / "cache"),
            iterations=iterations,
            expect_loaded_from_fastcache=expect_loaded,
        )
        args_json = args_obj.model_dump_json()
        cmd_line = [sys.executable, __file__, _fastcache_do_while_child.__name__, args_json]
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
        if proc.returncode != RET_SUCCESS:
            print(" ".join(cmd_line))
            print(proc.stdout)
            print("-" * 100)
            print(proc.stderr)
        assert TEST_RAN in proc.stdout
        assert proc.returncode == RET_SUCCESS


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
