"""Tests for qd.graph_parallel / qd.branch -- concurrent fork/join branches in graph kernels.

`with qd.graph_parallel():` opens a fork/join region whose `with qd.branch():` members are independent
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
    """At Python runtime (outside kernels) qd.graph_parallel / qd.branch must be usable no-op context
    managers, so helpers that are sometimes called from Python and sometimes from kernels still import
    and run. Mirrors qd.stream_parallel / qd.checkpoint."""
    sentinel = []
    with qd.graph_parallel():
        with qd.branch():
            sentinel.append("a")
        with qd.branch():
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
        with qd.graph_parallel():
            with qd.branch():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            with qd.branch():
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
        with qd.graph_parallel():
            with qd.branch():
                for i in range(a.shape[0]):
                    a[i] = a[i] + 1.0
            with qd.branch():
                for i in range(b.shape[0]):
                    b[i] = b[i] + 2.0
            with qd.branch():
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
        with qd.graph_parallel():
            with qd.branch():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
                for i in range(x.shape[0]):
                    x[i] = x[i] * 2.0
            with qd.branch():
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
        with qd.graph_parallel():
            with qd.branch():
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
        with qd.graph_parallel():
            with qd.branch():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            if qd.static(False):
                with qd.branch():
                    for i in range(y.shape[0]):
                        y[i] = y[i] + 1.0

    @qd.kernel(graph=True)
    def k_on(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            with qd.branch():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            if qd.static(True):
                with qd.branch():
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
            with qd.graph_parallel():
                with qd.branch():
                    for i in range(x.shape[0]):
                        x[i] = x[i] + 1
                with qd.branch():
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
        with qd.branch():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="qd.branch.. can only be used .* inside a qd.graph_parallel"):
        k(x)


@test_utils.test()
def test_graph_parallel_requires_graph_kernel():
    @qd.kernel
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            with qd.branch():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="requires @qd.kernel.graph=True"):
        k(x)


@test_utils.test()
def test_graph_parallel_non_branch_body_raises():
    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.branch"):
        k(x)


@test_utils.test()
def test_graph_parallel_nested_region_raises():
    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            with qd.branch():
                with qd.graph_parallel():
                    with qd.branch():
                        for i in range(x.shape[0]):
                            x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError):
        k(x)


# --- static-loop-generated branches (qd.static for-loop inside qd.graph_parallel) ---------------------
# The validator recurses into `for ... in qd.static(...)` loops, which unroll at trace time into literal
# `with qd.branch():` blocks (each with a fresh stream_parallel_group_id). This lets branches be generated
# from a compile-time sequence. Runtime for-loops stay rejected (see test_..._runtime_loop_raises).


@test_utils.test()
def test_graph_parallel_static_loop_two_branches():
    """`for b in qd.static(range(NB))` unrolls into NB literal branches, each writing a disjoint row."""
    nb = 2
    n = 256

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2)):
        with qd.graph_parallel():
            for b in qd.static(range(nb)):
                with qd.branch():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + (b + 1)

    x = qd.ndarray(qd.f32, shape=(nb, n))
    x.from_numpy(np.zeros((nb, n), dtype=np.float32))

    k(x)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # nb branches + one join

    out = x.to_numpy()
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 2.0)


@test_utils.test()
def test_graph_parallel_static_loop_over_funcs():
    """The motivating pattern: a @qd.data_oriented class iterates a static list of @qd.func members,
    one branch each (mirrors qipc's per-contact-type assembly funcs)."""
    n = 4

    @qd.data_oriented
    class Demo:
        def __init__(self):
            self.a = qd.field(qd.i32, shape=(n,))
            self.b = qd.field(qd.i32, shape=(n,))
            self.funcs = [self._fill_a, self._fill_b]

        @qd.func
        def _fill_a(self):
            for i in range(n):
                self.a[i] += 1

        @qd.func
        def _fill_b(self):
            for i in range(n):
                self.b[i] += 10

        @qd.kernel(graph=True)
        def step(self):
            with qd.graph_parallel():
                for i in qd.static(range(len(self.funcs))):
                    with qd.branch():
                        self.funcs[i]()

    d = Demo()
    d.a.from_numpy(np.zeros(n, dtype=np.int32))
    d.b.from_numpy(np.zeros(n, dtype=np.int32))
    d.step()
    np.testing.assert_array_equal(d.a.to_numpy(), np.ones(n, dtype=np.int32))
    np.testing.assert_array_equal(d.b.to_numpy(), np.full(n, 10, dtype=np.int32))


@test_utils.test()
def test_graph_parallel_static_loop_single_branch():
    """A static loop of one iteration is a single-branch region: a plain chain, no join node."""
    n = 256

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            for _b in qd.static(range(1)):
                with qd.branch():
                    for i in range(x.shape[0]):
                        x[i] = x[i] + 5.0

    x = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))

    k(x)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks  # single branch -> no join node

    np.testing.assert_allclose(x.to_numpy(), 5.0)


@test_utils.test()
def test_graph_parallel_static_loop_empty_range():
    """An empty static range produces zero branches: the region is a no-op (consistent with wrapping the
    only branch in `if qd.static(False)`). Serial work after it still runs."""
    n = 128

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            for _b in qd.static(range(0)):
                with qd.branch():
                    for i in range(x.shape[0]):
                        x[i] = x[i] + 1.0
        for i in range(x.shape[0]):
            x[i] = x[i] + 5.0

    x = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))

    k(x)
    np.testing.assert_allclose(x.to_numpy(), 5.0)  # region did nothing; only the serial +5 applied


@test_utils.test()
def test_graph_parallel_static_loop_nested():
    """Nested static loops fan out to N*M branches, each writing a disjoint row."""
    ni, nj = 2, 2
    nrows = ni * nj
    n = 64

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2)):
        with qd.graph_parallel():
            for i in qd.static(range(ni)):
                for j in qd.static(range(nj)):
                    with qd.branch():
                        for c in range(x.shape[1]):
                            x[i * nj + j, c] = x[i * nj + j, c] + (i * nj + j + 1)

    x = qd.ndarray(qd.f32, shape=(nrows, n))
    x.from_numpy(np.zeros((nrows, n), dtype=np.float32))

    k(x)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # nrows branches + one join

    out = x.to_numpy()
    for r in range(nrows):
        np.testing.assert_allclose(out[r], float(r + 1))


@test_utils.test()
def test_graph_parallel_static_loop_mixed_with_static_if():
    """A static branch loop and an `if qd.static(...)` optional branch coexist in one region."""
    nb = 2
    n = 64

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2), y: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            for b in qd.static(range(nb)):
                with qd.branch():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + (b + 1)
            if qd.static(True):
                with qd.branch():
                    for i in range(y.shape[0]):
                        y[i] = y[i] + 7.0

    x = qd.ndarray(qd.f32, shape=(nb, n))
    y = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros((nb, n), dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))

    k(x, y)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # nb + 1 branches + one join

    out = x.to_numpy()
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 2.0)
    np.testing.assert_allclose(y.to_numpy(), 7.0)


@test_utils.test()
def test_graph_parallel_runtime_loop_raises():
    """A *runtime* for-loop in a region body stays rejected: only `qd.static(...)` loops unroll to literal
    branches; a runtime range would nest the branch tagging inside a parallel range_for (malformed)."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2), nb: qd.i32):
        with qd.graph_parallel():
            for b in range(nb):
                with qd.branch():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + 1.0

    x = qd.ndarray(qd.f32, shape=(2, 16))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.branch"):
        k(x, 2)


@test_utils.test()
def test_branch_takes_no_arguments():
    """qd.branch() no longer accepts name= (the kwarg was parsed then dropped). Any argument raises."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel():
            with qd.branch(name="bx"):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="qd.branch.. takes no arguments"):
        k(x)


@test_utils.test()
def test_graph_parallel_static_loop_body_non_branch_raises():
    """A static loop body must still be branch-only: serial work inside the loop (outside any branch)
    would silently fall outside a branch, so it is rejected (the validator recurses into the loop body)."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2)):
        with qd.graph_parallel():
            for b in qd.static(range(2)):
                x[b, 0] = 1.0  # serial work outside any branch
                with qd.branch():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + 1.0

    x = qd.ndarray(qd.f32, shape=(2, 16))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.branch"):
        k(x)


@test_utils.test()
def test_graph_parallel_static_loop_runtime_inner_loop_raises():
    """Staticness is re-checked at every nesting level: a *runtime* loop nested inside a static loop and
    wrapping a branch is still rejected (only the static unroll yields independent branches)."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2), m: qd.i32):
        with qd.graph_parallel():
            for b in qd.static(range(2)):
                for _j in range(m):  # runtime loop around a branch -> rejected
                    with qd.branch():
                        for i in range(x.shape[1]):
                            x[b, i] = x[b, i] + 1.0

    x = qd.ndarray(qd.f32, shape=(2, 16))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.branch"):
        k(x, 2)


@test_utils.test()
def test_graph_parallel_static_loop_inside_graph_do_while():
    """A static branch loop composes with qd.graph_do_while: each iteration runs all unrolled branches,
    then decrements the counter."""
    nb = 2
    n = 64
    iters = 4

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=2), counter: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(counter):
            with qd.graph_parallel():
                for b in qd.static(range(nb)):
                    with qd.branch():
                        for i in range(x.shape[1]):
                            x[b, i] = x[b, i] + (b + 1)
            for _ in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(nb, n))
    counter = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros((nb, n), dtype=np.int32))
    counter.from_numpy(np.array(iters, dtype=np.int32))

    k(x, counter)

    assert counter.to_numpy() == 0
    out = x.to_numpy()
    np.testing.assert_array_equal(out[0], np.full(n, iters, dtype=np.int32))
    np.testing.assert_array_equal(out[1], np.full(n, 2 * iters, dtype=np.int32))


@test_utils.test()
def test_graph_parallel_static_loop_branch_body_in_gdw_rejects_bare_stmt():
    """Inside qd.graph_do_while, a bare assignment in a branch body re-executes every iteration (the E25
    footgun). The graph_do_while structure validator descends into static branch loops, so such a bare
    statement is rejected for static-loop branches just as it is for hand-written ones."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=2), counter: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(counter):
            with qd.graph_parallel():
                for b in qd.static(range(2)):
                    with qd.branch():
                        x[b, 0] = 1  # bare assignment in a branch body -> rejected inside graph_do_while
            for _ in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(2, 8))
    counter = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only for-loops"):
        k(x, counter)
