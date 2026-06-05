"""Nested qd.kernel-as-subgraph: a graph=True parent calls another qd.kernel, embedded as a child-graph node."""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _platform_supports_graph():
    arch = impl.current_cfg().arch
    return arch == qd.cuda or arch == qd.amdgpu


def _graph_used():
    return impl.get_runtime().prog.get_graph_cache_used_on_last_call()


def _graph_num_nodes():
    return impl.get_runtime().prog.get_graph_num_nodes_on_last_call()


def _num_offloaded_tasks():
    return impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call()


@test_utils.test()
def test_subgraph_single_child():
    """A graph=True parent that just calls one child embeds the child as a single child-graph node."""
    platform_supports_graph = _platform_supports_graph()
    n = 64

    @qd.kernel
    def child_double(a: qd.types.NDArray[qd.i32, 1]):
        for i in a:
            a[i] = a[i] * 2

    @qd.kernel(graph=True)
    def parent(a: qd.types.NDArray[qd.i32, 1]):
        child_double(a)

    a = qd.ndarray(qd.i32, (n,))
    a.fill(3)
    parent(a)
    nodes, used = _graph_num_nodes(), _graph_used()
    assert np.all(a.to_numpy() == 6)
    assert used == platform_supports_graph
    if platform_supports_graph:
        # Exactly one node: the embedded child subgraph.
        assert nodes == 1, f"expected 1 child-graph node, got {nodes}"

    # Relaunch from cache: child arg buffer refreshes against the same (device-resident) array, 6 -> 12.
    parent(a)
    assert _graph_used() == platform_supports_graph
    assert np.all(a.to_numpy() == 12)


@test_utils.test()
def test_subgraph_interleaved_with_loops():
    """Child calls interleave with the parent's own top-level for-loops, in source order."""
    platform_supports_graph = _platform_supports_graph()
    n = 64

    @qd.kernel
    def child_add(a: qd.types.NDArray[qd.i32, 1]):
        for i in a:
            a[i] = a[i] + 100

    @qd.kernel(graph=True)
    def parent(a: qd.types.NDArray[qd.i32, 1]):
        for i in a:
            a[i] = a[i] + 1
        child_add(a)
        for i in a:
            a[i] = a[i] + 1

    a = qd.ndarray(qd.i32, (n,))
    a.fill(0)
    parent(a)
    nodes, used, ntasks = _graph_num_nodes(), _graph_used(), _num_offloaded_tasks()
    assert np.all(a.to_numpy() == 102)  # 0 + 1 + 100 + 1
    assert used == platform_supports_graph
    if platform_supports_graph:
        # One graph node per offloaded task: the parent's own range_for tasks plus the embedded child node.
        assert nodes == ntasks
        assert nodes >= 3


@test_utils.test()
def test_subgraph_two_children():
    """Two child calls in one parent each become their own child-graph node with independent arg buffers."""
    platform_supports_graph = _platform_supports_graph()
    n = 32

    @qd.kernel
    def child_double(a: qd.types.NDArray[qd.i32, 1]):
        for i in a:
            a[i] = a[i] * 2

    @qd.kernel
    def child_inc(b: qd.types.NDArray[qd.i32, 1]):
        for i in b:
            b[i] = b[i] + 1

    @qd.kernel(graph=True)
    def parent(a: qd.types.NDArray[qd.i32, 1], b: qd.types.NDArray[qd.i32, 1]):
        child_double(a)
        child_inc(b)

    a = qd.ndarray(qd.i32, (n,))
    b = qd.ndarray(qd.i32, (n,))
    a.fill(5)
    b.fill(5)
    parent(a, b)
    used = _graph_used()  # capture before to_numpy() (readback triggers non-graph launches that reset the flag)
    assert np.all(a.to_numpy() == 10)
    assert np.all(b.to_numpy() == 6)
    assert used == platform_supports_graph


@test_utils.test()
def test_subgraph_changed_args():
    """A cached parent graph embeds the child correctly when relaunched with a different array."""
    platform_supports_graph = _platform_supports_graph()
    n = 48

    @qd.kernel
    def child_double(a: qd.types.NDArray[qd.i32, 1]):
        for i in a:
            a[i] = a[i] * 2

    @qd.kernel(graph=True)
    def parent(a: qd.types.NDArray[qd.i32, 1]):
        child_double(a)

    a1 = qd.ndarray(qd.i32, (n,))
    a1.fill(2)
    parent(a1)
    assert np.all(a1.to_numpy() == 4)

    a2 = qd.ndarray(qd.i32, (n,))
    a2.fill(7)
    parent(a2)
    assert _graph_used() == platform_supports_graph
    assert np.all(a2.to_numpy() == 14)
    # First array is untouched by the second launch.
    assert np.all(a1.to_numpy() == 4)


@test_utils.test()
def test_subgraph_requires_graph_parent():
    """Calling a qd.kernel from a non-graph kernel is rejected with a clear error."""

    @qd.kernel
    def child(a: qd.types.NDArray[qd.i32, 1]):
        for i in a:
            a[i] = a[i] + 1

    @qd.kernel
    def parent(a: qd.types.NDArray[qd.i32, 1]):
        child(a)

    a = qd.ndarray(qd.i32, (4,))
    with pytest.raises(Exception):
        parent(a)
