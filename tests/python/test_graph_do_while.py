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


def _graph_cache_size():
    return impl.get_runtime().prog.get_graph_cache_size()


def _graph_used():
    return impl.get_runtime().prog.get_graph_cache_used_on_last_call()


def _graph_total_builds():
    return impl.get_runtime().prog.get_graph_total_builds()


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _is_graph_do_while_natively_supported():
    return _on_cuda() and qd.lang.impl.get_cuda_compute_capability() >= 90


@test_utils.test()
def test_graph_do_while_counter():
    """Test graph_do_while with a counter that decrements each iteration."""
    N = 64

    @qd.kernel(graph=True)
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
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(10, dtype=np.int32))

    graph_loop(x, counter)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 10, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_boolean_done():
    """Test graph_do_while with a boolean 'continue' flag (non-zero = keep going)."""
    N = 64

    @qd.kernel(graph=True)
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
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 7, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    keep_going.from_numpy(np.array(1, dtype=np.int32))

    increment_until_threshold(x, 12, keep_going)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert keep_going.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 12, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_multiple_loops():
    """Test graph_do_while with multiple top-level loops in the kernel body."""
    N = 32

    @qd.kernel(graph=True)
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
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert counter.to_numpy() == 0
    np.testing.assert_allclose(x.to_numpy(), np.full(N, 10.0))
    np.testing.assert_allclose(y.to_numpy(), np.full(N, 20.0))

    x.from_numpy(np.zeros(N, dtype=np.float32))
    y.from_numpy(np.zeros(N, dtype=np.float32))
    counter.from_numpy(np.array(5, dtype=np.int32))

    multi_loop(x, y, counter)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

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

    @qd.kernel(graph=True)
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
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1
    assert c1.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))

    c2 = qd.ndarray(qd.i32, shape=())
    assert c1.arr.device_allocation_ptr() != c2.arr.device_allocation_ptr()
    x.from_numpy(np.zeros(N, dtype=np.int32))
    c2.from_numpy(np.array(7, dtype=np.int32))
    k(x, c2)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1
        assert _graph_total_builds() == 1
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

    @qd.kernel(graph=True)
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
        if _is_graph_do_while_natively_supported():
            assert _graph_used()
        assert c1.to_numpy() == 0
        np.testing.assert_array_equal(x.to_numpy(), np.full(N, count, dtype=np.int32))

        x.from_numpy(np.zeros(N, dtype=np.int32))
        c2.from_numpy(np.array(count + 10, dtype=np.int32))
        k(x, c2)
        if _is_graph_do_while_natively_supported():
            assert _graph_used()
        assert c2.to_numpy() == 0
        np.testing.assert_array_equal(x.to_numpy(), np.full(N, count + 10, dtype=np.int32))

    if _is_graph_do_while_natively_supported():
        assert _graph_cache_size() == 1
        assert _graph_total_builds() == 1


@test_utils.test()
def test_graph_do_while_without_graph_raises():
    """Using qd.graph_do_while without graph=True should raise."""

    @qd.kernel
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    c.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(qd.QuadrantsSyntaxError, match="requires @qd.kernel\\(graph=True\\)"):
        k(x, c)


@test_utils.test()
def test_graph_do_while_nonexistent_arg_raises():
    """Using a variable name that isn't a kernel parameter should raise."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(nonexistent):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    c.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(qd.QuadrantsSyntaxError, match="does not match any parameter"):
        k(x, c)


@test_utils.test()
def test_graph_do_while_unresolvable_member_attr_raises():
    """A graph_do_while condition that does not resolve to an ndarray kernel argument (here a bogus
    attribute on an ndarray parameter) must error at the loop site with the standard message."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c.not_an_attr):  # type: ignore[union-attr]
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="does not match any parameter"):
        k(x, c)


@test_utils.test()
def test_graph_do_while_data_oriented_member_counter():
    """graph_do_while accepts a @qd.data_oriented member ndarray (self.counter) as the loop condition.

    The member flattens to a real ndarray kernel argument, so its flat C++ arg-id is resolved at
    AST-build time exactly like a bare parameter -- it just reaches the arg via self. Mirrors
    test_graph_do_while_counter, but both the data (self.x) and the counter live on self.
    """
    N = 64

    @qd.data_oriented
    class Stepper:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.counter = qd.ndarray(qd.i32, shape=())

        @qd.kernel(graph=True)
        def step(self):
            while qd.graph_do_while(self.counter):
                for i in range(N):
                    self.x[i] = self.x[i] + 1
                for i in range(1):
                    self.counter[()] = self.counter[()] - 1

    s = Stepper()
    s.x.from_numpy(np.zeros(N, dtype=np.int32))
    s.counter.from_numpy(np.array(5, dtype=np.int32))

    s.step()
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1
    assert s.counter.to_numpy() == 0
    np.testing.assert_array_equal(s.x.to_numpy(), np.full(N, 5, dtype=np.int32))

    # Second launch with a different start value reuses the cached graph and still terminates.
    s.x.from_numpy(np.zeros(N, dtype=np.int32))
    s.counter.from_numpy(np.array(10, dtype=np.int32))
    s.step()
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1
    assert s.counter.to_numpy() == 0
    np.testing.assert_array_equal(s.x.to_numpy(), np.full(N, 10, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_data_oriented_member_metadata():
    """The gdw level table records the dotted label (self.counter) for a member-ndarray condition, and
    the legacy graph_do_while_arg alias mirrors the outermost level's label."""
    N = 8

    @qd.data_oriented
    class Stepper:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.counter = qd.ndarray(qd.i32, shape=())

        @qd.kernel(graph=True)
        def step(self):
            while qd.graph_do_while(self.counter):
                for i in range(N):
                    self.x[i] = self.x[i] + 1
                for i in range(1):
                    self.counter[()] = self.counter[()] - 1

    s = Stepper()
    s.x.from_numpy(np.zeros(N, dtype=np.int32))
    s.counter.from_numpy(np.array(3, dtype=np.int32))
    s.step()

    # `Stepper.step` is the @qd.data_oriented indirection wrapper; functools.update_wrapper copies the
    # inner @qd.kernel callable's __dict__ (including `_primal`) onto it, so the Kernel that records the
    # gdw level table is reachable directly as `Stepper.step._primal`.
    primal = Stepper.step._primal
    assert primal.graph_do_while_arg == "self.counter"
    assert [lvl.cond_arg_name for lvl in primal.graph_do_while_levels] == ["self.counter"]
    assert s.counter.to_numpy() == 0


@test_utils.test()
def test_graph_do_while_data_oriented_nested_members():
    """Two nested graph_do_while loops driven by two distinct @qd.data_oriented member ndarrays. Guards
    against arg-id mix-ups: each member must resolve to its own flat arg-id (mirrors
    test_graph_do_while_nested_two_levels with self.outer / self.inner)."""
    N = 32
    OUTER, INNER = 3, 4

    @qd.data_oriented
    class Stepper:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.outer = qd.ndarray(qd.i32, shape=())
            self.inner = qd.ndarray(qd.i32, shape=())

        @qd.kernel(graph=True)
        def step(self):
            while qd.graph_do_while(self.outer):
                for _ in range(1):
                    self.inner[()] = INNER
                while qd.graph_do_while(self.inner):
                    for i in range(N):
                        self.x[i] = self.x[i] + 1
                    for _ in range(1):
                        self.inner[()] = self.inner[()] - 1
                for _ in range(1):
                    self.outer[()] = self.outer[()] - 1

    s = Stepper()
    s.x.from_numpy(np.zeros(N, dtype=np.int32))
    s.outer.from_numpy(np.array(OUTER, dtype=np.int32))
    s.inner.from_numpy(np.array(0, dtype=np.int32))

    s.step()
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1
    assert s.outer.to_numpy() == 0
    np.testing.assert_array_equal(s.x.to_numpy(), np.full(N, OUTER * INNER, dtype=np.int32))

    primal = Stepper.step._primal
    assert [lvl.cond_arg_name for lvl in primal.graph_do_while_levels] == ["self.outer", "self.inner"]


@qd.kernel(graph=True, fastcache=True)
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


@test_utils.test()
def test_graph_do_while_nested_two_levels():
    """Two nested graph_do_while loops. The inner counter is reset at the start of each outer
    iteration; total work is outer_iters * inner_iters."""
    N = 32
    OUTER, INNER = 3, 4

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        outer: qd.types.ndarray(qd.i32, ndim=0),
        inner: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(outer):
            for _ in range(1):
                inner[()] = INNER
            while qd.graph_do_while(inner):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1
                for _ in range(1):
                    inner[()] = inner[()] - 1
            for _ in range(1):
                outer[()] = outer[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    outer = qd.ndarray(qd.i32, shape=())
    inner = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    outer.from_numpy(np.array(OUTER, dtype=np.int32))
    inner.from_numpy(np.array(0, dtype=np.int32))

    k(x, outer, inner)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert outer.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, OUTER * INNER, dtype=np.int32))

    # Re-run with different counts to confirm the indirection slots refresh (no rebuild).
    OUTER2, INNER2 = 5, 2
    x.from_numpy(np.zeros(N, dtype=np.int32))
    outer.from_numpy(np.array(OUTER2, dtype=np.int32))

    @qd.kernel(graph=True)
    def k2(
        x: qd.types.ndarray(qd.i32, ndim=1),
        outer: qd.types.ndarray(qd.i32, ndim=0),
        inner: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(outer):
            for _ in range(1):
                inner[()] = INNER2
            while qd.graph_do_while(inner):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1
                for _ in range(1):
                    inner[()] = inner[()] - 1
            for _ in range(1):
                outer[()] = outer[()] - 1

    k2(x, outer, inner)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, OUTER2 * INNER2, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_nested_three_levels():
    """Three nested graph_do_while loops; total work is the product of the three iteration counts."""
    N = 16
    A, B, C = 2, 3, 2

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        a: qd.types.ndarray(qd.i32, ndim=0),
        b: qd.types.ndarray(qd.i32, ndim=0),
        c: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(a):
            for _ in range(1):
                b[()] = B
            while qd.graph_do_while(b):
                for _ in range(1):
                    c[()] = C
                while qd.graph_do_while(c):
                    for i in range(x.shape[0]):
                        x[i] = x[i] + 1
                    for _ in range(1):
                        c[()] = c[()] - 1
                for _ in range(1):
                    b[()] = b[()] - 1
            for _ in range(1):
                a[()] = a[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    a = qd.ndarray(qd.i32, shape=())
    b = qd.ndarray(qd.i32, shape=())
    c = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    a.from_numpy(np.array(A, dtype=np.int32))
    b.from_numpy(np.array(0, dtype=np.int32))
    c.from_numpy(np.array(0, dtype=np.int32))

    k(x, a, b, c)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert a.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, A * B * C, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_siblings():
    """Two independent (sibling) graph_do_while loops at the kernel top level."""
    N = 24
    C1, C2 = 5, 3

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        y: qd.types.ndarray(qd.i32, ndim=1),
        c1: qd.types.ndarray(qd.i32, ndim=0),
        c2: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(c1):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for _ in range(1):
                c1[()] = c1[()] - 1
        while qd.graph_do_while(c2):
            for i in range(y.shape[0]):
                y[i] = y[i] + 2
            for _ in range(1):
                c2[()] = c2[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    y = qd.ndarray(qd.i32, shape=(N,))
    c1 = qd.ndarray(qd.i32, shape=())
    c2 = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    y.from_numpy(np.zeros(N, dtype=np.int32))
    c1.from_numpy(np.array(C1, dtype=np.int32))
    c2.from_numpy(np.array(C2, dtype=np.int32))

    k(x, y, c1, c2)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert c1.to_numpy() == 0
    assert c2.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, C1, dtype=np.int32))
    np.testing.assert_array_equal(y.to_numpy(), np.full(N, 2 * C2, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_mixed_with_top_level_for_loops():
    """Mix plain top-level for-loops (run once) with a graph_do_while loop. This is the headline
    case: a for-loop before and after the loop, both executed exactly once."""
    N = 20
    ITERS = 5

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 100
        while qd.graph_do_while(c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for _ in range(1):
                c[()] = c[()] - 1
        for i in range(x.shape[0]):
            x[i] = x[i] * 2

    x = qd.ndarray(qd.i32, shape=(N,))
    c = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    c.from_numpy(np.array(ITERS, dtype=np.int32))

    k(x, c)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert c.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, (100 + ITERS) * 2, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_canonical_seed_writeback_idiom():
    """The seed / iterate / writeback idiom for loop-carried state (see graph.md).

    The init and writeback live in *separate non-graph* kernels so they run exactly once, while the
    graph=True kernel contains only the do-while loop. State carries normally because nothing outside
    the loop body resets it. (On this branch the equivalent run-once top-level for-loops could also
    live inside the iterate kernel -- see test_graph_do_while_mixed_with_top_level_for_loops -- this
    test pins the separate-kernel split documented as the alternative.)
    """
    N = 8

    @qd.kernel  # no graph -- runs once per call
    def seed(q_iter: qd.types.ndarray(qd.i32, ndim=1), q: qd.types.ndarray(qd.i32, ndim=1)):
        for i in range(q.shape[0]):
            q_iter[i] = q[i]

    @qd.kernel(graph=True)
    def iterate(q_iter: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(c):
            for i in range(q_iter.shape[0]):
                q_iter[i] = q_iter[i] + 1
            for _ in range(1):
                c[()] = c[()] - 1

    @qd.kernel  # no graph -- runs once per call
    def writeback(q: qd.types.ndarray(qd.i32, ndim=1), q_iter: qd.types.ndarray(qd.i32, ndim=1)):
        for i in range(q.shape[0]):
            q[i] = q_iter[i]

    q = qd.ndarray(qd.i32, shape=(N,))
    q_iter = qd.ndarray(qd.i32, shape=(N,))
    c = qd.ndarray(qd.i32, shape=())
    q.from_numpy(np.full(N, 100, dtype=np.int32))
    c.from_numpy(np.array(4, dtype=np.int32))

    seed(q_iter, q)
    iterate(q_iter, c)
    writeback(q, q_iter)

    assert c.to_numpy() == 0
    np.testing.assert_array_equal(q.to_numpy(), np.full(N, 104, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_nested_mixed_with_for_loops():
    """For-loops interleaved with a nested graph_do_while at the outer level."""
    N = 16
    OUTER, INNER = 4, 3

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        outer: qd.types.ndarray(qd.i32, ndim=0),
        inner: qd.types.ndarray(qd.i32, ndim=0),
    ):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1000
        while qd.graph_do_while(outer):
            for _ in range(1):
                inner[()] = INNER
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
            while qd.graph_do_while(inner):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1
                for _ in range(1):
                    inner[()] = inner[()] - 1
            for _ in range(1):
                outer[()] = outer[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    outer = qd.ndarray(qd.i32, shape=())
    inner = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    outer.from_numpy(np.array(OUTER, dtype=np.int32))
    inner.from_numpy(np.array(0, dtype=np.int32))

    k(x, outer, inner)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert outer.to_numpy() == 0
    expected = 1000 + OUTER * 10 + OUTER * INNER
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, expected, dtype=np.int32))


@test_utils.test()
def test_graph_do_while_nested_dynamic_bounds():
    """A nested loop whose inner for-loop bound is read from device memory. The dynamic bound forces
    the offloader to emit a serial bound-computation task, which must be tagged at the inner level."""
    N = 32
    OUTER = 3

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        n: qd.types.ndarray(qd.i32, ndim=0),
        outer: qd.types.ndarray(qd.i32, ndim=0),
        inner: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(outer):
            for _ in range(1):
                inner[()] = 2
            while qd.graph_do_while(inner):
                for i in range(n[()]):
                    x[i] = x[i] + 1
                for _ in range(1):
                    inner[()] = inner[()] - 1
            for _ in range(1):
                outer[()] = outer[()] - 1

    half = N // 2
    x = qd.ndarray(qd.i32, shape=(N,))
    n = qd.ndarray(qd.i32, shape=())
    outer = qd.ndarray(qd.i32, shape=())
    inner = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    n.from_numpy(np.array(half, dtype=np.int32))
    outer.from_numpy(np.array(OUTER, dtype=np.int32))
    inner.from_numpy(np.array(0, dtype=np.int32))

    k(x, n, outer, inner)
    if _is_graph_do_while_natively_supported():
        assert _graph_used()
        assert _graph_cache_size() == 1

    assert outer.to_numpy() == 0
    expected = np.zeros(N, dtype=np.int32)
    expected[:half] = OUTER * 2
    np.testing.assert_array_equal(x.to_numpy(), expected)


@test_utils.test()
def test_graph_do_while_bare_statement_raises():
    """A bare (non-for) statement in a graph_do_while-using kernel body must raise."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        x[0] = 1
        while qd.graph_do_while(c):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            for _ in range(1):
                c[()] = c[()] - 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    c.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only for-loops"):
        k(x, c)


@test_utils.test()
def test_graph_do_while_inside_for_loop_raises():
    """graph_do_while nested inside a real for-loop must raise."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), c: qd.types.ndarray(qd.i32, ndim=0)):
        for _ in range(2):
            while qd.graph_do_while(c):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1
                for _ in range(1):
                    c[()] = c[()] - 1

    x = qd.ndarray(qd.i32, shape=(4,))
    c = qd.ndarray(qd.i32, shape=())
    c.from_numpy(np.array(1, dtype=np.int32))
    with pytest.raises(qd.QuadrantsSyntaxError, match="kernel top level"):
        k(x, c)


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
