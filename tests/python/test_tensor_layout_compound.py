"""Compound tensor layouts permute tensor axes while preserving trailing component axes."""

import pickle

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
DTYPES = [qd.types.vector(3, qd.f32), qd.types.matrix(2, 3, qd.f32)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,layout", [((2, 5), (1, 0)), ((2, 3, 5), (2, 0, 1))])
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_compound_layout_indexing_and_numpy(backend, dtype, shape, layout):
    a = qd.tensor(dtype, shape, backend=backend, layout=layout)
    element_shape = a.element_shape
    total_shape = (*shape, *element_shape)
    expected = np.arange(np.prod(total_shape), dtype=np.float32).reshape(total_shape)
    assert a.shape == shape
    assert a.layout == layout
    assert isinstance(a, qd.VectorTensor if len(element_shape) == 1 else qd.MatrixTensor)

    @qd.kernel
    def fill(x: qd.Tensor):
        for I in qd.grouped(x):
            base = 0
            for axis in qd.static(range(len(shape))):
                base = base * shape[axis] + I[axis]
            if qd.static(len(element_shape) == 1):
                for c in qd.static(range(element_shape[0])):
                    x[I][c] = base * element_shape[0] + c
            else:
                for r, c in qd.static(qd.ndrange(element_shape[0], element_shape[1])):
                    x[I][r, c] = (base * element_shape[0] + r) * element_shape[1] + c

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), expected)
    key = tuple(n - 1 for n in shape)
    np.testing.assert_array_equal(a[key].to_numpy(), expected[key])
    a[key] = (expected[key] + 100).tolist()
    expected[key] += 100
    np.testing.assert_array_equal(a.to_numpy(), expected)

    # Ingestion also accepts non-contiguous canonical arrays.
    source = expected[::-1]
    a.from_numpy(source)
    np.testing.assert_array_equal(a.to_numpy(), source)
    if backend is qd.Backend.NDARRAY:
        physical_shape = tuple(shape[axis] for axis in layout)
        with pytest.raises(ValueError, match="Mismatch shape"):
            a.from_numpy(np.zeros((*physical_shape, *element_shape), dtype=np.float32))

    # The same compiled entry point must distinguish natural and permuted storage.
    natural = qd.tensor(dtype, shape, backend=backend)
    fill(natural)
    fill(a)
    original = np.arange(np.prod(total_shape), dtype=np.float32).reshape(total_shape)
    np.testing.assert_array_equal(natural.to_numpy(), original)
    np.testing.assert_array_equal(a.to_numpy(), original)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", [None, (0, 1, 2), (2, 0, 1)])
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_compound_layout_torch_and_dlpack(backend, dtype, layout):
    torch = pytest.importorskip("torch")
    shape = (2, 3, 5)
    a = qd.tensor(dtype, shape, backend=backend, layout=layout)
    total_shape = (*shape, *a.element_shape)
    source = np.arange(np.prod(total_shape), dtype=np.float32).reshape(total_shape)
    a.from_numpy(source)
    np.testing.assert_array_equal(a.to_torch(device="cpu", copy=True).numpy(), source)

    # Slice a twice-sized last axis to exercise ingestion of a non-contiguous canonical tensor.
    incoming = torch.repeat_interleave(torch.from_numpy(source + 10), 2, dim=-1)[..., ::2]
    assert not incoming.is_contiguous()
    a.from_torch(incoming)
    np.testing.assert_array_equal(a.to_numpy(), source + 10)
    if backend is qd.Backend.NDARRAY and layout == (2, 0, 1):
        with pytest.raises(ValueError, match="Mismatch shape"):
            a.from_torch(torch.zeros((5, 2, 3, *a.element_shape)))

    qd.sync()
    view = torch.utils.dlpack.from_dlpack(a.to_dlpack())
    assert tuple(view.shape) == total_shape
    np.testing.assert_array_equal(view.cpu().numpy(), source + 10)
    permutation = (*layout, *range(3, len(total_shape))) if layout is not None else tuple(range(len(total_shape)))
    physical = np.empty(tuple(total_shape[axis] for axis in permutation), dtype=np.float32)
    reference = physical.transpose(np.argsort(permutation))
    assert view.stride() == tuple(s // reference.itemsize for s in reference.strides)

    alias = a.to_torch(copy=False)
    assert alias.data_ptr() == view.data_ptr()
    alias.add_(1)
    if alias.is_cuda:
        torch.cuda.synchronize()
    np.testing.assert_array_equal(a.to_numpy(), source + 11)

    if qd.lang.impl.current_cfg().arch == qd.cpu:
        numpy_view = a.to_numpy(copy=False)
        assert numpy_view.strides == reference.strides
        numpy_view[...] += 1
        np.testing.assert_array_equal(a.to_numpy(), source + 12)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_compound_layout_gradients(backend, dtype):
    shape = (2, 3, 5)
    layout = (2, 0, 1)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout, needs_grad=True)
    y = qd.tensor(dtype, shape, backend=backend, layout=layout, needs_grad=True)
    total_shape = (*shape, *x.element_shape)
    source = np.arange(np.prod(total_shape), dtype=np.float32).reshape(total_shape)
    x.from_numpy(source)

    @qd.kernel
    def square(a: qd.Tensor, b: qd.Tensor):
        for i, j, k in qd.ndrange(2, 3, 5):
            b[i, j, k] = a[i, j, k] * a[i, j, k]

    square(x, y)
    y.grad.fill(1)
    square.grad(x, y)
    assert x.grad.shape == shape
    assert x.grad.layout == layout
    np.testing.assert_array_equal(y.to_numpy(), source * source)
    np.testing.assert_array_equal(x.grad.to_numpy(), 2 * source)
    torch = pytest.importorskip("torch")
    qd.sync()
    grad_view = torch.utils.dlpack.from_dlpack(x.grad.to_dlpack())
    np.testing.assert_array_equal(grad_view.cpu().numpy(), 2 * source)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,layout", [((), ()), ((5,), (0,)), ((2, 3, 5), (2, 0, 1))])
@test_utils.test(arch=qd.cpu)
def test_compound_layout_pickle(backend, dtype, shape, layout):
    a = qd.tensor(dtype, shape, backend=backend, layout=layout)
    total_shape = (*shape, *a.element_shape)
    source = np.arange(np.prod(total_shape), dtype=np.float32).reshape(total_shape)
    a.from_numpy(source)
    restored = pickle.loads(pickle.dumps(a))
    assert type(restored) is type(a)
    assert restored.shape == a.shape
    assert restored.layout == a.layout
    np.testing.assert_array_equal(restored.to_numpy(), source)
