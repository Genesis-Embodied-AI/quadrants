"""Tests for qd.graph_parallel_context / qd.graph_parallel -- concurrent fork/join sections in graph
kernels.

`with qd.graph_parallel_context():` opens a fork/join region whose `with qd.graph_parallel():` members are
independent sequences of work. On the graph path the qd.graph_parallel sections become independent graph
chains joined by a single empty node, so the runtime schedules them on parallel streams; on other backends
(CPU / AMDGPU / Vulkan / Metal) they run serially but produce identical results.

The behavioral assertions (disjoint-array correctness) hold on every backend. The graph-structure
assertions (node counts: one kernel node per qd.graph_parallel section task + one empty join node) only
apply where the builder forks/joins (CUDA today), so they are guarded by `_on_cuda()`.
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
        with qd.graph_parallel():
            sentinel.append("b")
    assert sentinel == ["a", "b"]


@test_utils.test()
def test_graph_parallel_two_sections():
    """Two qd.graph_parallel sections write disjoint arrays; a serial loop after the region reads both (so
    it depends on the join). Results must match the serial reference on every backend; on CUDA the graph
    has one node per task plus one empty join node."""
    n = 1024

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.f32, ndim=1),
        y: qd.types.ndarray(qd.f32, ndim=1),
        z: qd.types.ndarray(qd.f32, ndim=1),
    ):
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0
            with qd.graph_parallel():
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
        # range_for, both in the qd.graph_parallel section) plus exactly one empty join node for the region.
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
def test_graph_parallel_back_to_back_regions_keep_join():
    """Two qd.graph_parallel_context() regions written back-to-back (no serial work between them) must each get
    their own fork/join. Region B reads what region A wrote, so if the two regions were merged into one fork/join
    (dropping A's join) B's sections would fork alongside -- and race -- A's. On CUDA we assert the graph has one
    empty join node per region (two total); on every backend the post-join values must be correct. Regression test
    for the back-to-back-region merge bug.

    Each region's two sections stay mutually disjoint (one touches only x, the other only y); the only data
    dependency is across the region boundary (B's x-section reads A's x-section output, B's y-section reads A's
    y-section output), which is exactly the edge the join protects."""
    n = 1024

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        # Region A: write x and y in two independent sections.
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = 1.0
            with qd.graph_parallel():
                for i in range(y.shape[0]):
                    y[i] = 2.0
        # Region B, immediately after A (no serial statement between the regions). Each section reads the region-A
        # section that wrote the same array, so B must wait for A's join. B's sections remain disjoint from each
        # other (x-only vs y-only).
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                for i in range(x.shape[0]):
                    x[i] = x[i] * 10.0
            with qd.graph_parallel():
                for i in range(y.shape[0]):
                    y[i] = y[i] * 10.0

    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))

    k(x, y)

    if _on_cuda():
        # One empty join node per region (two regions -> two joins). The merge bug this guards would build a single
        # fork/join across both regions, emitting only one join, i.e. num_tasks + 1.
        assert _graph_num_nodes() == _num_offloaded_tasks() + 2

    # Correct only if region A fully joined before region B ran: x = 1 * 10, y = 2 * 10.
    np.testing.assert_allclose(x.to_numpy(), 10.0)
    np.testing.assert_allclose(y.to_numpy(), 20.0)

    # Relaunch: same cached graph, same result.
    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))
    k(x, y)
    np.testing.assert_allclose(x.to_numpy(), 10.0)
    np.testing.assert_allclose(y.to_numpy(), 20.0)


@test_utils.test()
def test_graph_parallel_back_to_back_regions_serial_sections_keep_join():
    """Same back-to-back-region guard as above, but each qd.graph_parallel section's body is a bare serial store
    (a side-effecting statement, not a for-loop). Such statements are tagged by ASTBuilder::insert with the
    section's stream_parallel_group_id; the region id must be stamped there too, otherwise every serial-only
    section lands in region 0 and the graph builder merges the two contexts into one fork/join -- dropping A's
    join, so context B's sections fork alongside (and can race) context A's. Regression test for serial-task
    sections missing the graph_parallel_region_id stamp.

    Each context's two sections stay disjoint (one touches only the first array, the other only the second); the
    only dependency is across the context boundary (context B reads what context A wrote)."""

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.f32, ndim=1),
        y: qd.types.ndarray(qd.f32, ndim=1),
        a: qd.types.ndarray(qd.f32, ndim=1),
        b: qd.types.ndarray(qd.f32, ndim=1),
    ):
        # Context A: two serial-only sections writing disjoint scalars.
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                x[0] = 1.0
            with qd.graph_parallel():
                y[0] = 2.0
        # Context B immediately after A (no serial statement between them). Each section reads the context-A
        # section that wrote the array it consumes, so B must wait for A's join.
        with qd.graph_parallel_context():
            with qd.graph_parallel():
                a[0] = x[0] * 10.0
            with qd.graph_parallel():
                b[0] = y[0] * 10.0

    x = qd.ndarray(qd.f32, shape=(1,))
    y = qd.ndarray(qd.f32, shape=(1,))
    a = qd.ndarray(qd.f32, shape=(1,))
    b = qd.ndarray(qd.f32, shape=(1,))
    for arr in (x, y, a, b):
        arr.from_numpy(np.zeros(1, dtype=np.float32))

    k(x, y, a, b)

    if _on_cuda():
        # One empty join node per context (two contexts -> two joins). The merge bug this guards would build a
        # single fork/join across both contexts, emitting only one join, i.e. num_tasks + 1.
        assert _graph_num_nodes() == _num_offloaded_tasks() + 2

    # Correct only if context A fully joined before context B ran: a = x * 10 = 10, b = y * 10 = 20.
    np.testing.assert_allclose(a.to_numpy(), 10.0)
    np.testing.assert_allclose(b.to_numpy(), 20.0)

    # Relaunch: same cached graph, same result.
    for arr in (x, y, a, b):
        arr.from_numpy(np.zeros(1, dtype=np.float32))
    k(x, y, a, b)
    np.testing.assert_allclose(a.to_numpy(), 10.0)
    np.testing.assert_allclose(b.to_numpy(), 20.0)


@test_utils.test()
def test_graph_parallel_three_sections():
    """Fan-out of three independent qd.graph_parallel sections; one empty join node."""
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
        assert _graph_num_nodes() == num_tasks + 1  # three qd.graph_parallel sections + one join

    np.testing.assert_allclose(a.to_numpy(), 1.0)
    np.testing.assert_allclose(b.to_numpy(), 2.0)
    np.testing.assert_allclose(c.to_numpy(), 3.0)


@test_utils.test()
def test_graph_parallel_multi_loop_sections():
    """Each qd.graph_parallel section contains several loops; they must chain in order inside the
    qd.graph_parallel section while the two qd.graph_parallel sections run independently. qd.graph_parallel
    section tasks = 4, plus one join node on CUDA."""
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
        assert _graph_num_nodes() == num_tasks + 1  # all qd.graph_parallel section tasks + one join

    np.testing.assert_allclose(x.to_numpy(), 2.0)  # (0+1)*2
    np.testing.assert_allclose(y.to_numpy(), 12.0)  # (0+3)*4


@test_utils.test()
def test_graph_parallel_single_section_no_join():
    """A region with a single qd.graph_parallel section (e.g. an optional qd.graph_parallel section compiled
    out) needs no join: it degenerates to a plain chain, so the node count equals the number of
    qd.graph_parallel section tasks (no extra empty node)."""
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
        assert _graph_num_nodes() == num_tasks  # single qd.graph_parallel section -> plain chain, no join

    np.testing.assert_allclose(x.to_numpy(), 5.0)


@test_utils.test()
def test_graph_parallel_optional_section_static_if():
    """The qipc ENABLE_EE pattern: a qd.graph_parallel section wrapped in `if qd.static(...)`. When the flag
    is False the qd.graph_parallel section is compiled out (region has one qd.graph_parallel section -> no
    join); when True both qd.graph_parallel sections run."""
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
        assert _graph_num_nodes() == _num_offloaded_tasks()  # single qd.graph_parallel section -> no join
    np.testing.assert_allclose(x.to_numpy(), 1.0)
    np.testing.assert_allclose(y.to_numpy(), 0.0)  # EE qd.graph_parallel section compiled out

    x.from_numpy(np.zeros(n, dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))
    k_on(x, y)
    if _on_cuda():
        assert _graph_num_nodes() == _num_offloaded_tasks() + 1  # two qd.graph_parallel sections + join
    np.testing.assert_allclose(x.to_numpy(), 1.0)
    np.testing.assert_allclose(y.to_numpy(), 1.0)


@test_utils.test()
def test_graph_parallel_inside_graph_do_while():
    """A fork/join region inside a qd.graph_do_while loop body must be correct across iterations: each
    iteration runs both qd.graph_parallel sections, then decrements the counter."""
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
def test_graph_parallel_outside_context_raises():
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
def test_graph_parallel_context_non_graph_parallel_raises():
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


@test_utils.test()
def test_graph_parallel_static_loop_two_sections():
    """`for b in qd.static(range(NB))` unrolls into NB literal qd.graph_parallel sections, each writing a
    disjoint row."""
    nb = 2
    n = 256

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2)):
        with qd.graph_parallel_context():
            for b in qd.static(range(nb)):
                with qd.graph_parallel():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + (b + 1)

    x = qd.ndarray(qd.f32, shape=(nb, n))
    x.from_numpy(np.zeros((nb, n), dtype=np.float32))

    k(x)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # nb qd.graph_parallel sections + one join

    out = x.to_numpy()
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 2.0)


@test_utils.test()
def test_graph_parallel_static_loop_over_funcs():
    """The motivating pattern: a @qd.data_oriented class iterates a static list of @qd.func members, one
    qd.graph_parallel section each (mirrors qipc's per-contact-type assembly funcs)."""
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
            with qd.graph_parallel_context():
                for i in qd.static(range(len(self.funcs))):
                    with qd.graph_parallel():
                        self.funcs[i]()

    d = Demo()
    d.a.from_numpy(np.zeros(n, dtype=np.int32))
    d.b.from_numpy(np.zeros(n, dtype=np.int32))
    d.step()
    np.testing.assert_array_equal(d.a.to_numpy(), np.ones(n, dtype=np.int32))
    np.testing.assert_array_equal(d.b.to_numpy(), np.full(n, 10, dtype=np.int32))


@test_utils.test()
def test_graph_parallel_static_loop_single_section():
    """A static loop of one iteration is a single-section region: a plain chain, no join node."""
    n = 256

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            for _b in qd.static(range(1)):
                with qd.graph_parallel():
                    for i in range(x.shape[0]):
                        x[i] = x[i] + 5.0

    x = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros(n, dtype=np.float32))

    k(x)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks  # single section -> no join node

    np.testing.assert_allclose(x.to_numpy(), 5.0)


@test_utils.test()
def test_graph_parallel_static_loop_empty_range():
    """An empty static range produces zero qd.graph_parallel sections: the region is a no-op (consistent
    with wrapping the only section in `if qd.static(False)`). Serial work after it still runs."""
    n = 128

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            for _b in qd.static(range(0)):
                with qd.graph_parallel():
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
    """Nested static loops fan out to N*M qd.graph_parallel sections, each writing a disjoint row."""
    ni, nj = 2, 2
    nrows = ni * nj
    n = 64

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2)):
        with qd.graph_parallel_context():
            for i in qd.static(range(ni)):
                for j in qd.static(range(nj)):
                    with qd.graph_parallel():
                        for c in range(x.shape[1]):
                            x[i * nj + j, c] = x[i * nj + j, c] + (i * nj + j + 1)

    x = qd.ndarray(qd.f32, shape=(nrows, n))
    x.from_numpy(np.zeros((nrows, n), dtype=np.float32))

    k(x)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # nrows qd.graph_parallel sections + one join

    out = x.to_numpy()
    for r in range(nrows):
        np.testing.assert_allclose(out[r], float(r + 1))


@test_utils.test()
def test_graph_parallel_static_loop_mixed_with_static_if():
    """A static section loop and an `if qd.static(...)` optional qd.graph_parallel section coexist in one
    region."""
    nb = 2
    n = 64

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2), y: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            for b in qd.static(range(nb)):
                with qd.graph_parallel():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + (b + 1)
            if qd.static(True):
                with qd.graph_parallel():
                    for i in range(y.shape[0]):
                        y[i] = y[i] + 7.0

    x = qd.ndarray(qd.f32, shape=(nb, n))
    y = qd.ndarray(qd.f32, shape=(n,))
    x.from_numpy(np.zeros((nb, n), dtype=np.float32))
    y.from_numpy(np.zeros(n, dtype=np.float32))

    k(x, y)
    num_tasks = _num_offloaded_tasks()
    if _on_cuda():
        assert _graph_num_nodes() == num_tasks + 1  # nb + 1 qd.graph_parallel sections + one join

    out = x.to_numpy()
    np.testing.assert_allclose(out[0], 1.0)
    np.testing.assert_allclose(out[1], 2.0)
    np.testing.assert_allclose(y.to_numpy(), 7.0)


@test_utils.test()
def test_graph_parallel_runtime_loop_raises():
    """A *runtime* for-loop in a region body stays rejected: only `qd.static(...)` loops unroll to literal
    qd.graph_parallel sections; a runtime range would nest the section tagging inside a parallel range_for
    (malformed)."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2), nb: qd.i32):
        with qd.graph_parallel_context():
            for b in range(nb):
                with qd.graph_parallel():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + 1.0

    x = qd.ndarray(qd.f32, shape=(2, 16))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.graph_parallel"):
        k(x, 2)


@test_utils.test()
def test_graph_parallel_takes_no_arguments():
    """qd.graph_parallel() (the section) takes no arguments. Any argument raises."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=1)):
        with qd.graph_parallel_context():
            with qd.graph_parallel(name="bx"):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1.0

    x = qd.ndarray(qd.f32, shape=(16,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="qd.graph_parallel.. takes no arguments"):
        k(x)


@test_utils.test()
def test_graph_parallel_static_loop_body_non_section_raises():
    """A static loop body must still be section-only: serial work inside the loop (outside any
    qd.graph_parallel section) would silently fall outside a section, so it is rejected (the validator
    recurses into the loop body)."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2)):
        with qd.graph_parallel_context():
            for b in qd.static(range(2)):
                x[b, 0] = 1.0  # serial work outside any qd.graph_parallel section
                with qd.graph_parallel():
                    for i in range(x.shape[1]):
                        x[b, i] = x[b, i] + 1.0

    x = qd.ndarray(qd.f32, shape=(2, 16))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.graph_parallel"):
        k(x)


@test_utils.test()
def test_graph_parallel_static_loop_runtime_inner_loop_raises():
    """Staticness is re-checked at every nesting level: a *runtime* loop nested inside a static loop and
    wrapping a qd.graph_parallel section is still rejected (only the static unroll yields independent
    sections)."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.f32, ndim=2), m: qd.i32):
        with qd.graph_parallel_context():
            for b in qd.static(range(2)):
                for _j in range(m):  # runtime loop around a section -> rejected
                    with qd.graph_parallel():
                        for i in range(x.shape[1]):
                            x[b, i] = x[b, i] + 1.0

    x = qd.ndarray(qd.f32, shape=(2, 16))
    with pytest.raises(qd.QuadrantsSyntaxError, match="may contain only .with qd.graph_parallel"):
        k(x, 2)


@test_utils.test()
def test_graph_parallel_static_loop_inside_graph_do_while():
    """A static section loop composes with qd.graph_do_while: each iteration runs all unrolled
    qd.graph_parallel sections, then decrements the counter."""
    nb = 2
    n = 64
    iters = 4

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=2), counter: qd.types.ndarray(qd.i32, ndim=0)):
        while qd.graph_do_while(counter):
            with qd.graph_parallel_context():
                for b in qd.static(range(nb)):
                    with qd.graph_parallel():
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
