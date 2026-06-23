"""Tests for qd.graph_parallel_context / qd.graph_parallel -- concurrent fork/join branches in graph kernels.

`with qd.graph_parallel_context():` opens a fork/join region whose `with qd.graph_parallel():` members are independent
sequences of work. On the CUDA graph path the branches become independent graph chains joined by a single
empty node, so the runtime schedules them on parallel streams; on other backends (CPU / AMDGPU / Vulkan /
Metal) they run serially but produce identical results.

The behavioural assertions (disjoint-array correctness) hold on every backend. The graph-structure
assertions (node counts: one kernel node per branch task + one empty join node) only apply on the CUDA
graph path, where the builder forks/joins; they are guarded by `_on_cuda()`.
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _platform_supports_graph():
    arch = impl.current_cfg().arch
    return arch == qd.cuda or arch == qd.amdgpu


def _graph_num_nodes():
    return impl.get_runtime().prog.get_graph_num_nodes_on_last_call()


def _num_offloaded_tasks():
    return impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call()


@test_utils.test()
def test_graph_parallel_is_no_op_outside_kernels():
    """At Python runtime (outside kernels) qd.graph_parallel_context / qd.graph_parallel must be usable no-op context
    managers, so helpers that are sometimes called from Python and sometimes from kernels still import
    and run. Mirrors qd.stream_parallel / qd.checkpoint."""
    sentinel = []
    with qd.graph_parallel_context():
        with qd.graph_parallel():
            sentinel.append("a")
        with qd.graph_parallel(name="b"):
            sentinel.append("b")
    assert sentinel == ["a", "b"]


@test_utils.test()
def test_graph_parallel_two_branches():
    """Two branches write disjoint arrays; a serial loop after the region reads both (so it depends on
    the join). Results must match the serial reference on every backend; on CUDA the graph has one node
    per task plus one empty join node."""
    n = 1024

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.f32, ndim=1),
        y: qd.types.ndarray(qd.f32, ndim=1),
        z: qd.types.ndarray(qd.f32, ndim=1),
    ):
        with qd.graph_parallel_context():
            with qd.graph_parallel(name="bx"):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            with qd.graph_parallel(name="by"):
                for i in range(y.shape[0]):
                    y[i] = y[i] + 2.0
        for i in range(z.shape[0]):
            z[i] = x[i] + y[i]

    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))
    z = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))
    z.from_numpy(np.zeros(n, dtype=np.float32))

    k(x, y, z)

    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        # One graph node per offloaded task (each dynamic-bound loop is a bound-compute serial + a
        # range_for, both in the branch) plus exactly one empty join node for the single region.
        assert _graph_num_nodes() == num_tasks + 1

    np.testing.assert_allclose(x.to_numpy(), 1.0)
    np.testing.assert_allclose(y.to_numpy(), 2.0)
    np.testing.assert_allclose(z.to_numpy(), 3.0)

    # Relaunch: same cached graph, same result.
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))
    k(x, y, z)
    np.testing.assert_allclose(z.to_numpy(), 3.0)


@test_utils.test()
def test_graph_parallel_three_branches():
    """Fan-out of three independent branches; one empty join node."""
    n = 256

    @qd.kernel(graph=True)
    def k(
        a: qd.types.ndarray(qd.f32, ndim=1),
        b: qd.types.ndarray(qd.f32, ndim=1),
        c: qd.types.ndarray(qd.f32, ndim=1),
    ):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(a.shape[0]):
                    a[i] = a[i] + 1.0
            with qd.graph_parallel():
                for i in range(b.shape[0]):
                    b[i] = b[i] + 2.0
            with qd.graph_parallel():
                for i in range(c.shape[0]):
                    c[i] = c[i] + 3.0

    a = qd.ndarray(qd.f32, shape=(n,))
    b = qd.ndarray(qd.f32, shape=(n,))
    c = qd.ndarray(qd.f32, shape=(n,))
    for arr in (a, b, c):
        arr.from_numpy(np.zeros(n, dtype=np.float32))

    k(a, b, c)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # three branches + one join

    np.testing.assert_allclose(a.to_numpy(), 1.0)
    np.testing.assert_allclose(b.to_numpy(), 2.0)
    np.testing.assert_allclose(c.to_numpy(), 3.0)


@test_utils.test()
def test_graph_parallel_multi_loop_branches():
    """Each branch contains several loops; they must chain in order inside the branch while the two
    branches run independently. Branch tasks = 4, plus one join node on CUDA."""
    n = 128

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
                for i in range(x.shape[0]):
                    x[i] = x[i] * 2.0
            with qd.graph_parallel():
                for i in range(y.shape[0]):
                    y[i] = y[i] + 3.0
                for i in range(y.shape[0]):
                    y[i] = y[i] * 4.0

    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))

    k(x, y)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # all branch tasks + one join

    np.testing.assert_allclose(x.to_numpy(), 2.0)  # (0+1)*2
    np.testing.assert_allclose(y.to_numpy(), 12.0)  # (0+3)*4


@test_utils.test()
def test_graph_parallel_single_branch_no_join():
    """A region with a single branch (e.g. an optional branch compiled out) needs no join: it degenerates
    to a plain chain, so the node count equals the number of branch tasks (no extra empty node)."""
    n = 256

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 5.0

    x = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))

    k(x)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks  # single branch -> plain chain, no join node

    np.testing.assert_allclose(x.to_numpy(), 5.0)


@test_utils.test()
def test_graph_parallel_optional_branch_static_if():
    """The qipc ENABLE_EE pattern: a branch wrapped in `if qd.static(...)`. When the flag is False the
    branch is compiled out (region has one branch -> no join); when True both branches run."""
    n = 128

    @qd.kernel(graph=True)
    def k_off(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            if qd.static(False):
                with qd.graph_parallel():
                    for i in range(y.shape[0]):
                        y[i] = y[i] + 1.0

    @qd.kernel(graph=True)
    def k_on(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            if qd.static(True):
                with qd.graph_parallel():
                    for i in range(y.shape[0]):
                        y[i] = y[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))
    k_off(x, y)
    if _on_cuda():
        assert _graph_num_nodes() == _num_offloaded_tasks()  # single branch -> no join
    np.testing.assert_allclose(x.to_numpy(), 1.0)
    np.testing.assert_allclose(y.to_numpy(), 0.0)  # EE branch compiled out

    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))
    k_on(x, y)
    if _on_cuda():
        assert _graph_num_nodes() == _num_offloaded_tasks() + 1  # two branches + join
    np.testing.assert_allclose(x.to_numpy(), 1.0)
    np.testing.assert_allclose(y.to_numpy(), 1.0)


@test_utils.test()
def test_graph_parallel_inside_graph_do_while():
    """A fork/join region inside a qd.graph_do_while loop body must be correct across iterations: each
    iteration runs both branches, then decrements the counter."""
    n = 64
    iters = 5

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        y: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.graph_parallel_context():
                with qd.graph_parallel():
                    for i in range(x.shape[0]):
                        x[i] = x[i] + 1
                with qd.graph_parallel():
                    for i in range(y.shape[0]):
                        y[i] = y[i] + 2
            for _ in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(n,))
    y = qd.ndarray(qd.i32, shape=(n,))
    counter = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(n, dtype=np.int32))
    y.from_numpy(np.zeros(n, dtype=np.int32))
    counter.from_numpy(np.array(iters, dtype=np.int32))

    k(x, y, counter)

    assert counter.to_numpy() == 0
    np.testing.assert_array_equal(x.to_numpy(), np.full(n, iters, dtype=np.int32))
    np.testing.assert_array_equal(y.to_numpy(), np.full(n, 2 * iters, dtype=np.int32))


@test_utils.test()
def test_graph_parallel_branch_outside_region_raises():
    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(
        qd.QuadrantsSyntaxError, match="qd.graph_parallel.. can only be used .* inside a qd.graph_parallel_context"
    ):
        k(x)


@test_utils.test()
def test_graph_parallel_requires_graph_kernel():
    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="requires @qd.kernel.graph=True"):
        k(x)


@test_utils.test()
def test_graph_parallel_non_branch_body_raises():
    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.graph_parallel"):
        k(x)


@test_utils.test()
def test_graph_parallel_nested_region_raises():
    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                with qd.graph_parallel_context():
                    with qd.graph_parallel():
                        for i in range(x.shape[0]):
                            x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError):
        k(x)
