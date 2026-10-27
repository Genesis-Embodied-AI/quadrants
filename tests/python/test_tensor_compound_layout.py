"""Shape-axis layouts on vector and matrix tensors preserve canonical access and interop."""

import copy
import pickle

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.field import _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET

from tests import test_utils


def new_numpy_tensor_containing_index_codes(shape, element_shape):
    """Build NumPy values encoding every shape and component coordinate as decimal digits."""
    values = np.empty(shape + element_shape, dtype=np.float32)
    # np.ndindex(values.shape) produces every index tuple in the array.
    # E.g. for shape (2, 3), it yields (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2).
    for index in np.ndindex(values.shape):
        values[index] = test_utils.encode_index(index)
    return values


@qd.kernel
def fill_index_codes(x: qd.Tensor, dtype: qd.template()):
    """Write each scalar component's index code through canonical kernel indices."""
    for I in qd.grouped(x):
        position_code = test_utils.encode_index(I)
        for row in qd.static(range(dtype.n)):
            if qd.static(dtype.ndim == 1):
                x[I][row] = test_utils.encode_index([position_code, row])
            else:
                for column in qd.static(range(dtype.m)):
                    x[I][row, column] = test_utils.encode_index([position_code, row, column])


def require_torch_dlpack(backend):
    """Require Torch and, for Metal fields, support for DLPack byte offsets."""
    torch = pytest.importorskip("torch")
    if (
        qd.lang.impl.current_cfg().arch == qd.metal
        and backend == qd.Backend.FIELD
        and not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET
    ):
        pytest.skip("Metal field DLPack export requires PyTorch newer than 2.9.1")
    return torch


def synchronize_torch(tensor):
    """Wait for a Torch write before reading the shared memory through Quadrants."""
    import torch

    if tensor.device.type == "cuda":
        torch.cuda.synchronize()
    elif tensor.device.type == "mps":
        torch.mps.synchronize()


# Construction and indexing


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), None, id="field-vector-default"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="field-matrix-swap-axes-0-1"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (0, 1, 2), id="ndarray-matrix-identity"),
    ],
)
@test_utils.test()
def test_compound_layout_metadata(backend, dtype, layout):
    """Allocation preserves canonical shape and component dimensions, and normalizes identity layouts."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    element_shape = (dtype.n,) if dtype.ndim == 1 else (dtype.n, dtype.m)
    expected_layout = None if layout in (None, (0, 1, 2)) else layout

    assert x.shape == shape
    assert x.element_shape == element_shape
    assert x.layout == expected_layout


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_kernel_writes(backend, dtype, layout):
    """Kernel indexing writes the correct value to every shape and component coordinate."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    expected = new_numpy_tensor_containing_index_codes(shape, x.element_shape)

    fill_index_codes(x, dtype)

    np.testing.assert_array_equal(x.to_numpy(), expected)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_kernel_increment(backend, dtype, layout):
    """Grouped kernel iteration updates every canonical component exactly once."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    @qd.kernel
    def increment(x: qd.Tensor):
        for I in qd.grouped(x):
            x[I] += 1

    increment(x)

    np.testing.assert_array_equal(x.to_numpy(), initial + 1)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_python_reads(backend, dtype, layout):
    """Python indexing returns the components at the requested canonical position."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    actual = x[1, 2, 3].to_numpy()

    np.testing.assert_array_equal(actual, initial[1, 2, 3])


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_python_writes(backend, dtype, layout):
    """Python assignment changes only the requested canonical element."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    expected = initial.copy()
    expected[1, 2, 3] = initial[0, 0, 0]

    x[1, 2, 3] = initial[0, 0, 0].tolist()

    np.testing.assert_array_equal(x.to_numpy(), expected)


# NumPy imports and exports


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_numpy_noncontiguous_import(backend, dtype, layout):
    """NumPy imports preserve canonical values even when the source has non-contiguous storage."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    source = initial[:, :, ::-1]
    assert not source.flags.c_contiguous

    x.from_numpy(source)

    np.testing.assert_array_equal(x.to_numpy(), source)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_numpy_export_is_independent(backend, dtype, layout):
    """A copied NumPy export has canonical values and can be changed without changing the tensor."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    copied = x.to_numpy(copy=True)
    np.testing.assert_array_equal(copied, initial)
    copied.fill(-1)

    np.testing.assert_array_equal(x.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype,copy_mode",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), False, id="field-vector-copy-false"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), None, id="field-matrix-copy-none"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), None, id="ndarray-vector-copy-none"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), False, id="ndarray-matrix-copy-false"),
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), None, id="field-vector-copy-none"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), False, id="field-matrix-copy-false"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), False, id="ndarray-vector-copy-false"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), None, id="ndarray-matrix-copy-none"),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_compound_layout_numpy_zero_copy(backend, dtype, copy_mode):
    """Writing a NumPy view updates the tensor at the same canonical position."""
    pytest.importorskip("torch")
    shape = (2, 4, 5)
    layout = (2, 0, 1)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    expected = initial.copy()
    expected[1, 2, 3] = -7

    view = x.to_numpy(copy=copy_mode)
    np.testing.assert_array_equal(view, initial)
    view[1, 2, 3] = -7

    np.testing.assert_array_equal(x.to_numpy(), expected)


# Copies and serialization


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_copy_to_default_layout(backend, dtype, layout):
    """Copying between different layouts preserves every canonical component value."""
    shape = (2, 4, 5)
    src = qd.tensor(dtype, shape, backend=backend, layout=layout)
    dst = qd.tensor(dtype, shape, backend=backend, layout=None)
    initial = new_numpy_tensor_containing_index_codes(shape, src.element_shape)
    src.from_numpy(initial)
    dst.fill(-1)

    dst.copy_from(src)

    np.testing.assert_array_equal(dst.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_copy_to_permuted_layout(backend, dtype, layout):
    """Copying between different layouts preserves every canonical component value."""
    shape = (2, 4, 5)
    src = qd.tensor(dtype, shape, backend=backend, layout=None)
    dst = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, src.element_shape)
    src.from_numpy(initial)
    dst.fill(-1)

    dst.copy_from(src)

    np.testing.assert_array_equal(dst.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_pickle(backend, dtype, layout):
    """Pickle preserves the tensor type, canonical values, component dimensions, and layout."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    restored = pickle.loads(pickle.dumps(x))

    assert type(restored) is type(x)
    assert restored.shape == shape
    assert restored.element_shape == x.element_shape
    assert restored.layout == layout
    np.testing.assert_array_equal(restored.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_deepcopy(backend, dtype, layout):
    """Deepcopy preserves the tensor type, canonical values, component dimensions, and layout."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    restored = copy.deepcopy(x)

    assert type(restored) is type(x)
    assert restored.shape == shape
    assert restored.element_shape == x.element_shape
    assert restored.layout == layout
    np.testing.assert_array_equal(restored.to_numpy(), initial)


# Torch imports and exports


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_torch_export_is_independent(backend, dtype, layout):
    """A copied Torch export has canonical values and can be changed without changing the tensor."""
    pytest.importorskip("torch")
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    arch = qd.lang.impl.current_cfg().arch
    device = "cuda" if arch in (qd.cuda, qd.amdgpu) else "cpu"

    copied = x.to_torch(device=device, copy=True)
    np.testing.assert_array_equal(copied.cpu().numpy(), initial)
    copied.add_(10)

    np.testing.assert_array_equal(x.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test()
def test_compound_layout_torch_noncontiguous_import(backend, dtype, layout):
    """Torch imports preserve canonical values from a non-contiguous source."""
    torch = pytest.importorskip("torch")
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    arch = qd.lang.impl.current_cfg().arch
    device = "cuda" if arch in (qd.cuda, qd.amdgpu) else "cpu"
    source = torch.tensor(initial, device=device).transpose(0, 1).contiguous().transpose(0, 1)
    assert not source.is_contiguous()

    x.from_torch(source)

    np.testing.assert_array_equal(x.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype,copy_mode",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), False, id="field-vector-copy-false"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), None, id="field-matrix-copy-none"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), None, id="ndarray-vector-copy-none"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), False, id="ndarray-matrix-copy-false"),
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), None, id="field-vector-copy-none"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), False, id="field-matrix-copy-false"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), False, id="ndarray-vector-copy-false"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), None, id="ndarray-matrix-copy-none"),
    ],
)
@test_utils.test(exclude=[qd.vulkan])
def test_compound_layout_torch_zero_copy(backend, dtype, copy_mode):
    """Torch sharing modes return the existing buffer and expose writes at canonical positions."""
    torch = require_torch_dlpack(backend)
    shape = (2, 4, 5)
    layout = (2, 0, 1)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    expected = initial.copy()
    expected[1, 2, 3] = -7
    dlpack_view = torch.utils.dlpack.from_dlpack(x.to_dlpack())

    shared = x.to_torch(copy=copy_mode)
    assert shared.data_ptr() == dlpack_view.data_ptr()
    np.testing.assert_array_equal(shared.cpu().numpy(), initial)
    shared[1, 2, 3] = -7
    synchronize_torch(shared)

    np.testing.assert_array_equal(x.to_numpy(), expected)


# DLPack views


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test(exclude=[qd.vulkan])
def test_compound_layout_dlpack_shape_and_strides(backend, dtype, layout):
    """DLPack describes canonical values over physical storage using the correct strides."""
    torch = require_torch_dlpack(backend)
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    physical_axes = layout + tuple(range(len(shape), initial.ndim))
    physical = np.ascontiguousarray(initial.transpose(physical_axes))
    canonical_view = physical.transpose(np.argsort(physical_axes))
    expected_strides = tuple(stride // canonical_view.itemsize for stride in canonical_view.strides)

    view = torch.utils.dlpack.from_dlpack(x.to_dlpack())

    assert tuple(view.shape) == initial.shape
    assert tuple(view.stride()) == expected_strides
    np.testing.assert_array_equal(view.cpu().numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), id="field-vector-swap-axes-0-1"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), id="field-matrix-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), id="ndarray-vector-cycle-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"),
    ],
)
@test_utils.test(exclude=[qd.vulkan])
def test_compound_layout_dlpack_writes(backend, dtype, layout):
    """A write through a DLPack view changes the same canonical element in the tensor."""
    torch = require_torch_dlpack(backend)
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    expected = initial.copy()
    expected[1, 2, 3] = -7

    view = torch.utils.dlpack.from_dlpack(x.to_dlpack())
    view[1, 2, 3] = -7
    synchronize_torch(view)

    np.testing.assert_array_equal(x.to_numpy(), expected)


# Gradients, rank edge cases, validation, and kernel reuse


@pytest.mark.parametrize(
    "backend,dtype",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), id="field-vector"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), id="field-matrix"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), id="ndarray-vector"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), id="ndarray-matrix"),
    ],
)
@test_utils.test()
def test_compound_layout_grad(backend, dtype):
    """Check that gradients retain the layout and compute correct values across different input and output layouts."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=(2, 0, 1), needs_grad=True)
    y = qd.tensor(dtype, x.shape, backend=backend, layout=(1, 0, 2), needs_grad=True)
    assert x.grad.shape == x.shape
    assert x.grad.layout == x.layout

    @qd.kernel
    def double(x: qd.Tensor, y: qd.Tensor):
        for I in qd.grouped(x):
            y[I] = 2 * x[I]

    x.fill(3)
    x.grad.fill(0)
    y.grad.fill(1)
    double(x, y)
    double.grad(x, y)
    np.testing.assert_array_equal(x.grad.to_numpy(), np.full(x.shape + x.element_shape, 2))


@pytest.mark.parametrize(
    "backend,dtype,shape,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (), (), id="field-vector-0d-identity"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (), (), id="ndarray-matrix-0d-identity"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (4,), (0,), id="ndarray-vector-1d-identity"),
        pytest.param(
            qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 5), (1, 0), id="field-matrix-2d-swap-axes-0-1"
        ),
    ],
)
@test_utils.test()
def test_compound_layout_other_ranks(backend, dtype, shape, layout):
    """Check allocation, filling, and NumPy export for tensors with zero, one, or two shape axes."""
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    x.fill(4)
    assert x.shape == shape
    np.testing.assert_array_equal(x.to_numpy(), np.full(shape + x.element_shape, 4))


@pytest.mark.parametrize(
    "backend,dtype,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (0,), id="too-few-axes"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (0, 0), id="repeated-axis"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (0, 2), id="axis-out-of-range"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (0, 1, 2), id="too-many-axes"),
    ],
)
@test_utils.test()
def test_compound_layout_invalid(backend, dtype, layout):
    """Check that short, long, repeated-axis, and out-of-range layout permutations are rejected."""
    with pytest.raises(ValueError, match="layout"):
        qd.tensor(dtype, (2, 5), backend=backend, layout=layout)


@pytest.mark.parametrize(
    "backend,dtype",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), id="field-vector"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), id="field-matrix"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), id="ndarray-vector"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), id="ndarray-matrix"),
    ],
)
@test_utils.test()
def test_compound_layout_kernel_specialization(backend, dtype):
    """Check kernel reuse across layouts using distinct codes for every tensor position and element component."""

    shape = (2, 5)
    element_shape = (dtype.n,) if dtype.ndim == 1 else (dtype.n, dtype.m)
    expected = new_numpy_tensor_containing_index_codes(shape, element_shape)

    for layout in (None, (1, 0), None):
        x = qd.tensor(dtype, shape, backend=backend, layout=layout)
        fill_index_codes(x, dtype)

        np.testing.assert_array_equal(x.to_numpy(), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(qd.types.vector(3, qd.f32), id="vector"),
        pytest.param(qd.types.matrix(2, 3, qd.f32), id="matrix"),
    ],
)
@test_utils.test()
def test_compound_layout_numpy_import_rejects_physical_shape(dtype):
    """NumPy imports require canonical shape rather than the physical storage shape."""
    shape = (2, 5)
    layout = (1, 0)
    x = qd.tensor(dtype, shape, backend=qd.Backend.NDARRAY, layout=layout)
    source = np.zeros((5, 2) + x.element_shape, dtype=np.float32)

    with pytest.raises(ValueError, match="Mismatch shape"):
        x.from_numpy(source)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(qd.types.vector(3, qd.f32), id="vector"),
        pytest.param(qd.types.matrix(2, 3, qd.f32), id="matrix"),
    ],
)
@test_utils.test()
def test_compound_layout_torch_import_rejects_physical_shape(dtype):
    """Torch imports require canonical shape rather than the physical storage shape."""
    shape = (2, 5)
    layout = (1, 0)
    x = qd.tensor(dtype, shape, backend=qd.Backend.NDARRAY, layout=layout)
    wrong_shape = np.zeros((5, 2) + x.element_shape, dtype=np.float32)
    torch = pytest.importorskip("torch")
    source = torch.from_numpy(wrong_shape)

    with pytest.raises(ValueError, match="Mismatch shape"):
        x.from_torch(source)
