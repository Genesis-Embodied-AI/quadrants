import numpy as np

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _cuda_graph_cache_size():
    return impl.get_runtime().prog.get_cuda_graph_cache_size()


def _cuda_graph_used():
    return impl.get_runtime().prog.get_cuda_graph_cache_used_on_last_call()


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


@test_utils.test()
def test_cuda_graph_two_loops():
    """A kernel with two top-level for loops should be fused into a CUDA graph."""
    platform_supports_graph = _on_cuda()
    n = 1024
    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel(cuda_graph=True)
    def two_loops(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    assert _cuda_graph_cache_size() == 0
    two_loops(x, y)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _cuda_graph_used() == platform_supports_graph
    two_loops(x, y)
    assert _cuda_graph_used() == platform_supports_graph
    two_loops(x, y)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)

    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert np.allclose(x_np, 3.0), f"Expected 3.0, got {x_np[:5]}"
    assert np.allclose(y_np, 6.0), f"Expected 6.0, got {y_np[:5]}"


@test_utils.test()
def test_cuda_graph_three_loops():
    """A kernel with three top-level for loops."""
    platform_supports_graph = _on_cuda()
    n = 512
    a = qd.ndarray(qd.f32, shape=(n,))
    b = qd.ndarray(qd.f32, shape=(n,))
    c = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel(cuda_graph=True)
    def three_loops(
        a: qd.types.ndarray(qd.f32, ndim=1), b: qd.types.ndarray(qd.f32, ndim=1), c: qd.types.ndarray(qd.f32, ndim=1)
    ):
        for i in range(a.shape[0]):
            a[i] = a[i] + 1.0
        for i in range(b.shape[0]):
            b[i] = b[i] + 10.0
        for i in range(c.shape[0]):
            c[i] = a[i] + b[i]

    assert _cuda_graph_cache_size() == 0
    three_loops(a, b, c)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _cuda_graph_used() == platform_supports_graph

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    c_np = c.to_numpy()
    assert np.allclose(a_np, 1.0)
    assert np.allclose(b_np, 10.0)
    assert np.allclose(c_np, 11.0)

    three_loops(a, b, c)
    assert _cuda_graph_used() == platform_supports_graph
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    c_np = c.to_numpy()
    assert np.allclose(a_np, 2.0)
    assert np.allclose(b_np, 20.0)
    assert np.allclose(c_np, 22.0)


@test_utils.test()
def test_cuda_graph_single_loop_no_graph():
    """A kernel with a single for loop should NOT use the graph path,
    even with cuda_graph=True (falls back since < 2 tasks)."""
    n = 256
    x = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel(cuda_graph=True)
    def single_loop(x: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 5.0

    single_loop(x)
    assert not _cuda_graph_used()
    single_loop(x)
    assert not _cuda_graph_used()
    assert _cuda_graph_cache_size() == 0

    x_np = x.to_numpy()
    assert np.allclose(x_np, 10.0)


@test_utils.test()
def test_no_cuda_graph_annotation():
    """A kernel WITHOUT cuda_graph=True should never use the graph path."""
    n = 256
    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def two_loops(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    two_loops(x, y)
    assert not _cuda_graph_used()
    two_loops(x, y)
    assert not _cuda_graph_used()
    assert _cuda_graph_cache_size() == 0

    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert np.allclose(x_np, 2.0)
    assert np.allclose(y_np, 4.0)


@test_utils.test()
def test_cuda_graph_changed_args():
    """Graph should produce correct results when called with different ndarrays."""
    platform_supports_graph = _on_cuda()
    n = 256

    @qd.kernel(cuda_graph=True)
    def two_loops(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    x1 = qd.ndarray(qd.f32, shape=(n,))
    y1 = qd.ndarray(qd.f32, shape=(n,))
    assert _cuda_graph_cache_size() == 0
    two_loops(x1, y1)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _cuda_graph_used() == platform_supports_graph
    two_loops(x1, y1)
    assert _cuda_graph_used() == platform_supports_graph

    x1_np = x1.to_numpy()
    y1_np = y1.to_numpy()
    assert np.allclose(x1_np, 2.0), f"Expected 2.0, got {x1_np[:5]}"
    assert np.allclose(y1_np, 4.0), f"Expected 4.0, got {y1_np[:5]}"

    x2 = qd.ndarray(qd.f32, shape=(n,))
    y2 = qd.ndarray(qd.f32, shape=(n,))
    x2.from_numpy(np.full(n, 10.0, dtype=np.float32))
    y2.from_numpy(np.full(n, 20.0, dtype=np.float32))
    two_loops(x2, y2)
    assert _cuda_graph_used() == platform_supports_graph
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)

    x2_np = x2.to_numpy()
    y2_np = y2.to_numpy()
    assert np.allclose(x2_np, 11.0), f"Expected 11.0, got {x2_np[:5]}"
    assert np.allclose(y2_np, 22.0), f"Expected 22.0, got {y2_np[:5]}"

    x1_np = x1.to_numpy()
    y1_np = y1.to_numpy()
    assert np.allclose(x1_np, 2.0), f"x1 should be unchanged, got {x1_np[:5]}"
    assert np.allclose(y1_np, 4.0), f"y1 should be unchanged, got {y1_np[:5]}"


@test_utils.test()
def test_cuda_graph_different_sizes():
    """Graph must produce correct results when called with different-sized arrays.

    Catches stale grid dims: if the graph cached from the small call is
    replayed for the large call, elements beyond the original size stay zero.
    """
    platform_supports_graph = _on_cuda()

    @qd.kernel(cuda_graph=True)
    def add_one(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    x1 = qd.ndarray(qd.f32, shape=(256,))
    y1 = qd.ndarray(qd.f32, shape=(256,))
    assert _cuda_graph_cache_size() == 0
    add_one(x1, y1)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _cuda_graph_used() == platform_supports_graph

    x2 = qd.ndarray(qd.f32, shape=(1024,))
    y2 = qd.ndarray(qd.f32, shape=(1024,))
    add_one(x2, y2)
    assert _cuda_graph_used() == platform_supports_graph

    x2_np = x2.to_numpy()
    y2_np = y2.to_numpy()
    assert np.allclose(x2_np, 1.0), f"Expected all 1.0, got {x2_np[250:260]}"
    assert np.allclose(y2_np, 2.0), f"Expected all 2.0, got {y2_np[250:260]}"


@test_utils.test()
def test_cuda_graph_after_reset():
    """cuda_graph=True kernel must work correctly after qd.reset()."""
    platform_supports_graph = _on_cuda()

    @qd.kernel(cuda_graph=True)
    def add_one(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    n = 256
    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))
    add_one(x, y)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _cuda_graph_used() == platform_supports_graph
    add_one(x, y)
    assert _cuda_graph_used() == platform_supports_graph

    assert np.allclose(x.to_numpy(), 2.0)
    assert np.allclose(y.to_numpy(), 4.0)

    arch = impl.current_cfg().arch
    qd.reset()
    qd.init(arch=arch)

    x2 = qd.ndarray(qd.f32, shape=(n,))
    y2 = qd.ndarray(qd.f32, shape=(n,))
    assert _cuda_graph_cache_size() == 0
    add_one(x2, y2)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _cuda_graph_used() == platform_supports_graph

    assert np.allclose(x2.to_numpy(), 1.0)
    assert np.allclose(y2.to_numpy(), 2.0)


@test_utils.test()
def test_cuda_graph_annotation_cross_platform():
    """cuda_graph=True should be a harmless no-op on non-CUDA backends."""
    platform_supports_graph = _on_cuda()
    n = 256
    x = qd.ndarray(qd.f32, shape=(n,))
    y = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel(cuda_graph=True)
    def two_loops(x: qd.types.ndarray(qd.f32, ndim=1), y: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    assert _cuda_graph_cache_size() == 0
    two_loops(x, y)
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _cuda_graph_used() == platform_supports_graph
    two_loops(x, y)
    assert _cuda_graph_used() == platform_supports_graph
    assert _cuda_graph_cache_size() == (1 if platform_supports_graph else 0)

    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert np.allclose(x_np, 2.0), f"Expected 2.0, got {x_np[:5]}"
    assert np.allclose(y_np, 4.0), f"Expected 4.0, got {y_np[:5]}"
