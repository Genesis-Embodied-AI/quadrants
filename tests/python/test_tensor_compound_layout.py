"""Shape-axis layouts on vector and matrix tensors preserve canonical access and interop."""

import copy
import pickle

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.field import _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
DTYPES = [qd.types.vector(3, qd.f32), qd.types.matrix(2, 3, qd.f32)]
LAYOUTS = [None, (0, 1, 2), (1, 0, 2), (2, 0, 1)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@test_utils.test()
def test_compound_layout_indexing_and_numpy(backend, dtype, layout):
    shape = (2, 4, 5)
    a = qd.tensor(dtype, shape, backend=backend, layout=layout)
    element_shape = (dtype.n,) if dtype.ndim == 1 else (dtype.n, dtype.m)
    expected = np.arange(np.prod(shape + element_shape), dtype=np.float32).reshape(shape + element_shape)
    assert a.shape == shape
    assert a.element_shape == element_shape
    assert a.layout == (None if layout in (None, (0, 1, 2)) else layout)

    @qd.kernel
    def fill(x: qd.Tensor):
        for i, j, k in qd.ndrange(2, 4, 5):
            for r in qd.static(range(dtype.n)):
                if qd.static(dtype.ndim == 1):
                    x[i, j, k][r] = ((i * 4 + j) * 5 + k) * dtype.n + r
                else:
                    for c in qd.static(range(dtype.m)):
                        x[i, j, k][r, c] = (((i * 4 + j) * 5 + k) * dtype.n + r) * dtype.m + c

    @qd.kernel
    def increment(x: qd.Tensor):
        for I in qd.grouped(x):
            x[I] += 1

    fill(a)
    np.testing.assert_array_equal(a.to_numpy(), expected)
    np.testing.assert_array_equal(a[1, 2, 3].to_numpy(), expected[1, 2, 3])
    increment(a)
    expected += 1
    np.testing.assert_array_equal(a.to_numpy(), expected)
    a[1, 2, 3] = expected[0, 0, 0].tolist()
    expected[1, 2, 3] = expected[0, 0, 0]
    np.testing.assert_array_equal(a.to_numpy(), expected)

    # Non-contiguous canonical input must not be mistaken for the physical allocation shape.
    source = expected[:, :, ::-1]
    a.from_numpy(source)
    np.testing.assert_array_equal(a.to_numpy(), source)
    copied = a.to_numpy()
    copied.fill(-1)
    np.testing.assert_array_equal(a.to_numpy(), source)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", [(1, 0, 2), (2, 0, 1)])
@test_utils.test()
def test_compound_layout_copy_and_pickle(backend, dtype, layout):
    a = qd.tensor(dtype, (2, 4, 5), backend=backend, layout=layout)
    b = qd.tensor(dtype, a.shape, backend=backend)
    expected = np.arange(np.prod(a.shape + a.element_shape), dtype=np.float32).reshape(a.shape + a.element_shape)
    a.from_numpy(expected)
    b.copy_from(a)
    np.testing.assert_array_equal(b.to_numpy(), expected)
    a.fill(0)
    a.copy_from(b)
    np.testing.assert_array_equal(a.to_numpy(), expected)
    for restored in (pickle.loads(pickle.dumps(a)), copy.deepcopy(a)):
        assert type(restored) is type(a)
        assert restored.shape == a.shape
        assert restored.element_shape == a.element_shape
        assert restored.layout == layout
        np.testing.assert_array_equal(restored.to_numpy(), expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", [(1, 0, 2), (2, 0, 1)])
@test_utils.test(exclude=[qd.vulkan])
def test_compound_layout_torch_and_dlpack(backend, dtype, layout):
    torch = pytest.importorskip("torch")
    arch = qd.lang.impl.current_cfg().arch
    if arch == qd.metal and backend == qd.Backend.FIELD and not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET:
        pytest.skip("Metal field DLPack export requires PyTorch newer than 2.9.1")
    a = qd.tensor(dtype, (2, 4, 5), backend=backend, layout=layout)
    expected = np.arange(np.prod(a.shape + a.element_shape), dtype=np.float32).reshape(a.shape + a.element_shape)
    a.from_numpy(expected)
    device = "cuda" if arch in (qd.cuda, qd.amdgpu) else "cpu"
    copied = a.to_torch(device=device)
    np.testing.assert_array_equal(copied.cpu().numpy(), expected)
    copied.add_(10)
    np.testing.assert_array_equal(a.to_numpy(), expected)
    # Keep the canonical shape but supply non-contiguous strides.
    source = copied.transpose(0, 1).contiguous().transpose(0, 1)
    a.from_torch(source)
    expected += 10
    np.testing.assert_array_equal(a.to_numpy(), expected)

    view = torch.utils.dlpack.from_dlpack(a.to_dlpack())
    np.testing.assert_array_equal(view.cpu().numpy(), expected)
    physical_axes = layout + tuple(range(3, expected.ndim))
    physical = np.ascontiguousarray(expected.transpose(physical_axes))
    canonical = physical.transpose(np.argsort(physical_axes))
    assert tuple(view.stride()) == tuple(s // canonical.itemsize for s in canonical.strides)
    view[1, 2, 3] = -7
    if view.device.type == "cuda":
        torch.cuda.synchronize()
    elif view.device.type == "mps":
        torch.mps.synchronize()
    expected[1, 2, 3] = -7
    np.testing.assert_array_equal(a.to_numpy(), expected)
    for copy_mode in (False, None):
        shared = a.to_torch(copy=copy_mode)
        assert shared.data_ptr() == view.data_ptr()
        np.testing.assert_array_equal(shared.cpu().numpy(), expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@test_utils.test(arch=qd.cpu)
def test_compound_layout_numpy_zero_copy(backend, dtype):
    pytest.importorskip("torch")
    a = qd.tensor(dtype, (2, 4, 5), backend=backend, layout=(2, 0, 1))
    a.fill(3)
    for copy_mode in (False, None):
        view = a.to_numpy(copy=copy_mode)
        assert view.shape == a.shape + a.element_shape
        view[1, 2, 3] = 7
        np.testing.assert_array_equal(a[1, 2, 3].to_numpy(), np.full(a.element_shape, 7))
        a.fill(3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@test_utils.test()
def test_compound_layout_grad(backend, dtype):
    a = qd.tensor(dtype, (2, 4, 5), backend=backend, layout=(2, 0, 1), needs_grad=True)
    b = qd.tensor(dtype, a.shape, backend=backend, layout=(1, 0, 2), needs_grad=True)
    assert a.grad.shape == a.shape
    assert a.grad.layout == a.layout

    @qd.kernel
    def double(x: qd.Tensor, y: qd.Tensor):
        for I in qd.grouped(x):
            y[I] = 2 * x[I]

    a.fill(3)
    a.grad.fill(0)
    b.grad.fill(1)
    double(a, b)
    double.grad(a, b)
    np.testing.assert_array_equal(a.grad.to_numpy(), np.full(a.shape + a.element_shape, 2))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,layout", [((), ()), ((4,), (0,)), ((2, 5), (1, 0))])
@test_utils.test()
def test_compound_layout_other_ranks(backend, dtype, shape, layout):
    a = qd.tensor(dtype, shape, backend=backend, layout=layout)
    a.fill(4)
    assert a.shape == shape
    np.testing.assert_array_equal(a.to_numpy(), np.full(shape + a.element_shape, 4))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", [(0,), (0, 0), (0, 2), (0, 1, 2)])
@test_utils.test()
def test_compound_layout_invalid(backend, dtype, layout):
    with pytest.raises(ValueError, match="layout"):
        qd.tensor(dtype, (2, 5), backend=backend, layout=layout)


@pytest.mark.parametrize("dtype", DTYPES)
@test_utils.test()
def test_compound_layout_ndarray_import_rejects_physical_shape(dtype):
    a = qd.tensor(dtype, (2, 5), layout=(1, 0))
    wrong_shape = np.zeros((5, 2) + a.element_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="Mismatch shape"):
        a.from_numpy(wrong_shape)
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="Mismatch shape"):
        a.from_torch(torch.from_numpy(wrong_shape))


@pytest.mark.parametrize("dtype", DTYPES)
@test_utils.test()
def test_compound_layout_kernel_specialization(dtype):
    @qd.kernel
    def fill(x: qd.Tensor):
        for i, j in qd.ndrange(2, 5):
            x[i, j] = x[i, j] * 0 + 10 * i + j

    expected = np.arange(2, dtype=np.float32)[:, None] * 10 + np.arange(5, dtype=np.float32)
    for layout in (None, (1, 0), None):
        a = qd.tensor(dtype, (2, 5), layout=layout)
        fill(a)
        expanded = expected.reshape((2, 5) + (1,) * len(a.element_shape))
        np.testing.assert_array_equal(a.to_numpy(), np.broadcast_to(expanded, (2, 5) + a.element_shape))
