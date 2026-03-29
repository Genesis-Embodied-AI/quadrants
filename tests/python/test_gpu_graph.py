import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _gpu_graph_cache_size():
    return impl.get_runtime().prog.get_gpu_graph_cache_size()


def _gpu_graph_used():
    return impl.get_runtime().prog.get_gpu_graph_cache_used_on_last_call()


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _num_offloaded_tasks():
    return impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call()


def _gpu_graph_num_nodes():
    return impl.get_runtime().prog.get_gpu_graph_num_nodes_on_last_call()


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_gpu_graph_two_loops(tensor_type):
    """A kernel with two top-level for loops should be fused into a CUDA graph."""
    platform_supports_graph = _on_cuda()
    n = 1024

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.kernel(gpu_graph=True)
    def two_loops(x: Annotation, y: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    x = tensor_type(qd.f32, (n,))
    y = tensor_type(qd.f32, (n,))

    assert _gpu_graph_cache_size() == 0
    two_loops(x, y)
    num_tasks = _num_offloaded_tasks()
    assert num_tasks >= 2
    expected_nodes = num_tasks if platform_supports_graph else 0
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph
    two_loops(x, y)
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_used() == platform_supports_graph
    two_loops(x, y)
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)

    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert np.allclose(x_np, 3.0), f"Expected 3.0, got {x_np[:5]}"
    assert np.allclose(y_np, 6.0), f"Expected 6.0, got {y_np[:5]}"


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_gpu_graph_three_loops(tensor_type):
    """A kernel with three top-level for loops."""
    platform_supports_graph = _on_cuda()
    n = 512

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.kernel(gpu_graph=True)
    def three_loops(a: Annotation, b: Annotation, c: Annotation):
        for i in range(a.shape[0]):
            a[i] = a[i] + 1.0
        for i in range(b.shape[0]):
            b[i] = b[i] + 10.0
        for i in range(c.shape[0]):
            c[i] = a[i] + b[i]

    a = tensor_type(qd.f32, (n,))
    b = tensor_type(qd.f32, (n,))
    c = tensor_type(qd.f32, (n,))

    assert _gpu_graph_cache_size() == 0
    three_loops(a, b, c)
    num_tasks = _num_offloaded_tasks()
    assert num_tasks >= 3
    expected_nodes = num_tasks if platform_supports_graph else 0
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    c_np = c.to_numpy()
    assert np.allclose(a_np, 1.0)
    assert np.allclose(b_np, 10.0)
    assert np.allclose(c_np, 11.0)

    three_loops(a, b, c)
    assert _gpu_graph_used() == platform_supports_graph
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)

    a_np = a.to_numpy()
    b_np = b.to_numpy()
    c_np = c.to_numpy()
    assert np.allclose(a_np, 2.0)
    assert np.allclose(b_np, 20.0)
    assert np.allclose(c_np, 22.0)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_gpu_graph_multi_func(tensor_type):
    """A kernel calling three funcs with 2, 4, and 3 top-level for loops."""
    platform_supports_graph = _on_cuda()
    n = 256

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.func
    def func_a(x: Annotation, y: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    @qd.func
    def func_b(a: Annotation, b: Annotation, c: Annotation, d: Annotation):
        for i in range(a.shape[0]):
            a[i] = a[i] + 3.0
        for i in range(b.shape[0]):
            b[i] = b[i] + 4.0
        for i in range(c.shape[0]):
            c[i] = c[i] + 5.0
        for i in range(d.shape[0]):
            d[i] = d[i] + 6.0

    @qd.func
    def func_c(x: Annotation, y: Annotation, z: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 7.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 8.0
        for i in range(z.shape[0]):
            z[i] = z[i] + 9.0

    @qd.kernel(gpu_graph=True)
    def multi_func(a: Annotation, b: Annotation, c: Annotation, d: Annotation, e: Annotation, f: Annotation):
        func_a(a, b)
        func_b(a, b, c, d)
        func_c(d, e, f)

    a = tensor_type(qd.f32, (n,))
    b = tensor_type(qd.f32, (n,))
    c = tensor_type(qd.f32, (n,))
    d = tensor_type(qd.f32, (n,))
    e = tensor_type(qd.f32, (n,))
    f = tensor_type(qd.f32, (n,))

    assert _gpu_graph_cache_size() == 0
    multi_func(a, b, c, d, e, f)
    num_tasks = _num_offloaded_tasks()
    assert num_tasks >= 9
    expected_nodes = num_tasks if platform_supports_graph else 0
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph

    # func_a: a += 1, b += 2
    # func_b: a += 3, b += 4, c += 5, d += 6
    # func_c: d += 7, e += 8, f += 9
    assert np.allclose(a.to_numpy(), 4.0)  # 1 + 3
    assert np.allclose(b.to_numpy(), 6.0)  # 2 + 4
    assert np.allclose(c.to_numpy(), 5.0)
    assert np.allclose(d.to_numpy(), 13.0)  # 6 + 7
    assert np.allclose(e.to_numpy(), 8.0)
    assert np.allclose(f.to_numpy(), 9.0)

    multi_func(a, b, c, d, e, f)
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_used() == platform_supports_graph

    assert np.allclose(a.to_numpy(), 8.0)
    assert np.allclose(b.to_numpy(), 12.0)
    assert np.allclose(c.to_numpy(), 10.0)
    assert np.allclose(d.to_numpy(), 26.0)
    assert np.allclose(e.to_numpy(), 16.0)
    assert np.allclose(f.to_numpy(), 18.0)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_no_gpu_graph_annotation(tensor_type):
    """A kernel WITHOUT gpu_graph=True should never use the graph path."""
    n = 256

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.kernel
    def two_loops(x: Annotation, y: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    x = tensor_type(qd.f32, (n,))
    y = tensor_type(qd.f32, (n,))

    two_loops(x, y)
    assert _num_offloaded_tasks() >= 2
    assert _gpu_graph_num_nodes() == 0
    assert not _gpu_graph_used()
    two_loops(x, y)
    assert not _gpu_graph_used()
    assert _gpu_graph_cache_size() == 0

    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert np.allclose(x_np, 2.0)
    assert np.allclose(y_np, 4.0)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_gpu_graph_changed_args(tensor_type):
    """Graph should produce correct results when called with different tensors."""
    platform_supports_graph = _on_cuda()
    n = 256

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.kernel(gpu_graph=True)
    def two_loops(x: Annotation, y: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    x1 = tensor_type(qd.f32, (n,))
    y1 = tensor_type(qd.f32, (n,))
    assert _gpu_graph_cache_size() == 0
    two_loops(x1, y1)
    num_tasks = _num_offloaded_tasks()
    assert num_tasks >= 2
    expected_nodes = num_tasks if platform_supports_graph else 0
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph
    two_loops(x1, y1)
    assert _gpu_graph_used() == platform_supports_graph

    x1_np = x1.to_numpy()
    y1_np = y1.to_numpy()
    assert np.allclose(x1_np, 2.0), f"Expected 2.0, got {x1_np[:5]}"
    assert np.allclose(y1_np, 4.0), f"Expected 4.0, got {y1_np[:5]}"

    x2 = tensor_type(qd.f32, (n,))
    y2 = tensor_type(qd.f32, (n,))
    x2.from_numpy(np.full(n, 10.0, dtype=np.float32))
    y2.from_numpy(np.full(n, 20.0, dtype=np.float32))
    two_loops(x2, y2)
    assert _gpu_graph_used() == platform_supports_graph
    # Fields are template args, so different field objects produce a second
    # compiled kernel and a second graph cache entry.
    if tensor_type == qd.field:
        assert _gpu_graph_cache_size() == (2 if platform_supports_graph else 0)
    else:
        assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)

    x2_np = x2.to_numpy()
    y2_np = y2.to_numpy()
    assert np.allclose(x2_np, 11.0), f"Expected 11.0, got {x2_np[:5]}"
    assert np.allclose(y2_np, 22.0), f"Expected 22.0, got {y2_np[:5]}"

    x1_np = x1.to_numpy()
    y1_np = y1.to_numpy()
    assert np.allclose(x1_np, 2.0), f"x1 should be unchanged, got {x1_np[:5]}"
    assert np.allclose(y1_np, 4.0), f"y1 should be unchanged, got {y1_np[:5]}"


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_gpu_graph_different_sizes(tensor_type):
    """Graph must produce correct results when called with different-sized arrays.

    Catches stale grid dims: if the graph cached from the small call is
    replayed for the large call, elements beyond the original size stay zero.

    For fields, different-sized fields are separate template specializations,
    so each gets its own graph cache entry.
    """
    platform_supports_graph = _on_cuda()

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.kernel(gpu_graph=True)
    def add_one(x: Annotation, y: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    x1 = tensor_type(qd.f32, (256,))
    y1 = tensor_type(qd.f32, (256,))
    assert _gpu_graph_cache_size() == 0
    add_one(x1, y1)
    num_tasks = _num_offloaded_tasks()
    assert num_tasks >= 2
    expected_nodes = num_tasks if platform_supports_graph else 0
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph

    x2 = tensor_type(qd.f32, (1024,))
    y2 = tensor_type(qd.f32, (1024,))
    add_one(x2, y2)
    assert _gpu_graph_used() == platform_supports_graph
    # Ndarrays reuse the same compiled kernel; fields produce a second
    # template specialization with its own graph cache entry.
    if tensor_type == qd.field:
        assert _gpu_graph_cache_size() == (2 if platform_supports_graph else 0)
    else:
        assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)

    x2_np = x2.to_numpy()
    y2_np = y2.to_numpy()
    assert np.allclose(x2_np, 1.0), f"Expected all 1.0, got {x2_np[250:260]}"
    assert np.allclose(y2_np, 2.0), f"Expected all 2.0, got {y2_np[250:260]}"


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_gpu_graph_after_reset(tensor_type):
    """gpu_graph=True kernel must work correctly after qd.reset()."""
    platform_supports_graph = _on_cuda()

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.kernel(gpu_graph=True)
    def add_one(x: Annotation, y: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    n = 256
    x = tensor_type(qd.f32, (n,))
    y = tensor_type(qd.f32, (n,))
    add_one(x, y)
    num_tasks = _num_offloaded_tasks()
    assert num_tasks >= 2
    expected_nodes = num_tasks if platform_supports_graph else 0
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph
    add_one(x, y)
    assert _gpu_graph_used() == platform_supports_graph

    assert np.allclose(x.to_numpy(), 2.0)
    assert np.allclose(y.to_numpy(), 4.0)

    arch = impl.current_cfg().arch
    qd.reset()
    qd.init(arch=arch)

    x2 = tensor_type(qd.f32, (n,))
    y2 = tensor_type(qd.f32, (n,))
    assert _gpu_graph_cache_size() == 0
    add_one(x2, y2)
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph

    assert np.allclose(x2.to_numpy(), 1.0)
    assert np.allclose(y2.to_numpy(), 2.0)


@pytest.mark.parametrize("tensor_type", [qd.ndarray, qd.field])
@test_utils.test()
def test_gpu_graph_annotation_cross_platform(tensor_type):
    """gpu_graph=True should be a harmless no-op on non-CUDA backends."""
    platform_supports_graph = _on_cuda()
    n = 256

    Annotation = qd.types.NDArray[qd.f32, 1] if tensor_type == qd.ndarray else qd.Template

    @qd.kernel(gpu_graph=True)
    def two_loops(x: Annotation, y: Annotation):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1.0
        for i in range(y.shape[0]):
            y[i] = y[i] + 2.0

    x = tensor_type(qd.f32, (n,))
    y = tensor_type(qd.f32, (n,))

    assert _gpu_graph_cache_size() == 0
    two_loops(x, y)
    num_tasks = _num_offloaded_tasks()
    assert num_tasks >= 2
    expected_nodes = num_tasks if platform_supports_graph else 0
    assert _gpu_graph_num_nodes() == expected_nodes
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)
    assert _gpu_graph_used() == platform_supports_graph
    two_loops(x, y)
    assert _gpu_graph_used() == platform_supports_graph
    assert _gpu_graph_cache_size() == (1 if platform_supports_graph else 0)

    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert np.allclose(x_np, 2.0), f"Expected 2.0, got {x_np[:5]}"
    assert np.allclose(y_np, 4.0), f"Expected 4.0, got {y_np[:5]}"
