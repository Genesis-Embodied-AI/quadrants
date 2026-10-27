"""Shape-axis layouts on vector and matrix tensors preserve canonical access and interop."""

import copy
import pickle

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.matrix import MatrixType

from tests import test_utils

# Construction and indexing


@pytest.mark.parametrize(
    "backend,dtype,shape,layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (2, 4, 5), None, id="field-vector-default"),
        pytest.param(
            qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 4, 5), (1, 0, 2), id="field-matrix-swap-axes-0-1"
        ),
        pytest.param(
            qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 4, 5), (2, 0, 1), id="ndarray-vector-cycle-axes"
        ),
        pytest.param(
            qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (2, 4, 5), (0, 1, 2), id="ndarray-matrix-identity"
        ),
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (), (), id="field-vector-0d-identity"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (), (), id="ndarray-matrix-0d-identity"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (4,), (0,), id="ndarray-vector-1d-identity"),
        pytest.param(
            qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 5), (1, 0), id="field-matrix-2d-swap-axes-0-1"
        ),
    ],
)
@test_utils.test()
def test_compound_layout_metadata(
    backend: qd.Backend, dtype: MatrixType, shape: tuple[int, ...], layout: tuple[int, ...] | None
) -> None:
    """Check shape, component dimensions, and normalized layouts across tensor ranks."""
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    element_shape = (dtype.n,) if dtype.ndim == 1 else (dtype.n, dtype.m)
    expected_layout = None if layout is None or layout == tuple(range(len(shape))) else layout

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
def test_compound_layout_kernel_writes(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
    """Kernel indexing writes the correct value to every shape and component coordinate."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    expected = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)

    test_utils.fill_index_codes(x, dtype)

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
def test_compound_layout_kernel_increment(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
    """Grouped kernel iteration updates every canonical component exactly once."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    @qd.kernel
    def increment(x: qd.Tensor) -> None:
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
def test_compound_layout_python_reads(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
    """Python indexing returns the components at the requested canonical position."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
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
def test_compound_layout_python_writes(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
    """Python assignment changes only the requested canonical element."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    replacement = np.full(x.element_shape, -1, dtype=np.float32)
    x[1, 2, 3] = replacement.tolist()

    expected = initial.copy()
    expected[1, 2, 3] = replacement
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
def test_compound_layout_numpy_noncontiguous_import(
    backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]
) -> None:
    """NumPy imports preserve canonical values even when the source has non-contiguous storage."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    # ::-1 reverses the third axis, creating a view with a negative stride without copying the data.
    noncontiguous_input = initial[:, :, ::-1]
    assert not noncontiguous_input.flags.c_contiguous

    x.from_numpy(noncontiguous_input)

    np.testing.assert_array_equal(x.to_numpy(), noncontiguous_input)


@pytest.mark.parametrize(
    "backend,dtype,shape,layout",
    [
        pytest.param(
            qd.Backend.FIELD, qd.types.vector(3, qd.f32), (2, 4, 5), (1, 0, 2), id="field-vector-swap-axes-0-1"
        ),
        pytest.param(
            qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 4, 5), (2, 0, 1), id="field-matrix-cycle-axes"
        ),
        pytest.param(
            qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 4, 5), (2, 0, 1), id="ndarray-vector-cycle-axes"
        ),
        pytest.param(
            qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (2, 4, 5), (1, 0, 2), id="ndarray-matrix-swap-axes-0-1"
        ),
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (), (), id="field-vector-0d-identity"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (), (), id="ndarray-matrix-0d-identity"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (4,), (0,), id="ndarray-vector-1d-identity"),
        pytest.param(
            qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 5), (1, 0), id="field-matrix-2d-swap-axes-0-1"
        ),
    ],
)
@test_utils.test()
def test_compound_layout_numpy_export_is_independent(
    backend: qd.Backend, dtype: MatrixType, shape: tuple[int, ...], layout: tuple[int, ...]
) -> None:
    """A copied NumPy export has canonical values and can be changed without changing the tensor."""
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)

    # Request a new NumPy array with its own storage. Its indices must have the same meaning as initial,
    # including when x stores its axes in a different physical order.
    copied = x.to_numpy(copy=True)
    np.testing.assert_array_equal(copied, initial)

    # Overwrite the export with a value absent from the nonnegative index codes. If it shares storage with x,
    # this write would also change x and the final comparison would fail.
    copied.fill(-1)

    # Read x again to verify that changing the exported array left every original value intact.
    np.testing.assert_array_equal(x.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), id="field-vector"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), id="field-matrix"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), id="ndarray-vector"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), id="ndarray-matrix"),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_compound_layout_numpy_zero_copy(backend: qd.Backend, dtype: MatrixType) -> None:
    """Writing a NumPy view updates the tensor at the same canonical position."""
    pytest.importorskip("torch")
    shape = (2, 4, 5)
    layout = (2, 0, 1)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    expected = initial.copy()
    expected[1, 2, 3] = -7

    np_view = x.to_numpy(copy=False)
    np.testing.assert_array_equal(np_view, initial)
    np_view[1, 2, 3] = -7

    np.testing.assert_array_equal(x.to_numpy(), expected)


# Copies and serialization


@pytest.mark.parametrize(
    "backend,dtype,src_layout,dst_layout",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), (1, 0, 2), None, id="field-vector-to-default"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), (2, 0, 1), None, id="field-matrix-to-default"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), (2, 0, 1), None, id="ndarray-vector-to-default"),
        pytest.param(
            qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), (1, 0, 2), None, id="ndarray-matrix-to-default"
        ),
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), None, (1, 0, 2), id="field-vector-to-permuted"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), None, (2, 0, 1), id="field-matrix-to-permuted"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), None, (2, 0, 1), id="ndarray-vector-to-permuted"),
        pytest.param(
            qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), None, (1, 0, 2), id="ndarray-matrix-to-permuted"
        ),
    ],
)
@test_utils.test()
def test_compound_layout_copy_between_layouts(
    backend: qd.Backend,
    dtype: MatrixType,
    src_layout: tuple[int, ...] | None,
    dst_layout: tuple[int, ...] | None,
) -> None:
    """Copying between different layouts preserves every canonical component value."""
    shape = (2, 4, 5)
    src = qd.tensor(dtype, shape, backend=backend, layout=src_layout)
    dst = qd.tensor(dtype, shape, backend=backend, layout=dst_layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, src.element_shape)
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
def test_compound_layout_pickle(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
    """Pickle preserves the tensor type, canonical values, component dimensions, and layout."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
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
def test_compound_layout_deepcopy(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
    """Deepcopy preserves the tensor type, canonical values, component dimensions, and layout."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
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
def test_compound_layout_torch_export_is_independent(
    backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]
) -> None:
    """A copied Torch export has canonical values and can be changed without changing the tensor."""
    pytest.importorskip("torch")
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
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
def test_compound_layout_torch_noncontiguous_import(
    backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]
) -> None:
    """Torch imports preserve canonical values from a non-contiguous source."""
    torch = pytest.importorskip("torch")
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    arch = qd.lang.impl.current_cfg().arch
    device = "cuda" if arch in (qd.cuda, qd.amdgpu) else "cpu"
    noncontiguous_input = torch.tensor(initial, device=device).transpose(0, 1).contiguous().transpose(0, 1)
    assert not noncontiguous_input.is_contiguous()

    x.from_torch(noncontiguous_input)

    np.testing.assert_array_equal(x.to_numpy(), initial)


@pytest.mark.parametrize(
    "backend,dtype",
    [
        pytest.param(qd.Backend.FIELD, qd.types.vector(3, qd.f32), id="field-vector"),
        pytest.param(qd.Backend.FIELD, qd.types.matrix(2, 3, qd.f32), id="field-matrix"),
        pytest.param(qd.Backend.NDARRAY, qd.types.vector(3, qd.f32), id="ndarray-vector"),
        pytest.param(qd.Backend.NDARRAY, qd.types.matrix(2, 3, qd.f32), id="ndarray-matrix"),
    ],
)
@test_utils.test(exclude=[qd.vulkan])
def test_compound_layout_torch_zero_copy(backend: qd.Backend, dtype: MatrixType) -> None:
    """Torch sharing modes return the existing buffer and expose writes at canonical positions."""
    test_utils.require_torch_dlpack(need_field_support=backend == qd.Backend.FIELD)
    shape = (2, 4, 5)
    layout = (2, 0, 1)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    expected = initial.copy()
    expected[1, 2, 3] = -7

    torch_view = x.to_torch(copy=False)
    np.testing.assert_array_equal(torch_view.cpu().numpy(), initial)
    torch_view[1, 2, 3] = -7
    test_utils.torch_synchronize(torch_view.device)

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
def test_compound_layout_dlpack_shape_and_strides(
    backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]
) -> None:
    """DLPack describes canonical values over physical storage using the correct strides."""
    torch = test_utils.require_torch_dlpack(need_field_support=backend == qd.Backend.FIELD)
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    physical_axes = layout + tuple(range(len(shape), len(shape) + len(x.element_shape)))
    physical_values_np = np.ascontiguousarray(initial.transpose(physical_axes))
    # Restores the original axis order without changing the physical storage.
    canonical_view_np = physical_values_np.transpose(np.argsort(physical_axes))
    # Converts NumPy's strides from bytes to scalar elements, because DLPack and Torch express strides in elements.
    expected_strides = tuple(stride // canonical_view_np.itemsize for stride in canonical_view_np.strides)

    # Creates a Torch tensor that shares x's memory.
    torch_view = torch.utils.dlpack.from_dlpack(x.to_dlpack())

    assert tuple(torch_view.shape) == initial.shape
    assert tuple(torch_view.stride()) == expected_strides
    np.testing.assert_array_equal(torch_view.cpu().numpy(), initial)


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
def test_compound_layout_dlpack_writes(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
    """A write through a DLPack view changes the same canonical element in the tensor."""
    torch = test_utils.require_torch_dlpack(need_field_support=backend == qd.Backend.FIELD)
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=layout)
    initial = test_utils.new_numpy_tensor_containing_index_codes(shape, x.element_shape)
    x.from_numpy(initial)
    expected = initial.copy()
    expected[1, 2, 3] = -7

    torch_view = torch.utils.dlpack.from_dlpack(x.to_dlpack())
    torch_view[1, 2, 3] = -7
    test_utils.torch_synchronize(torch_view.device)

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
def test_compound_layout_grad(backend: qd.Backend, dtype: MatrixType) -> None:
    """Check that gradients retain the layout and compute correct values across different input and output layouts."""
    shape = (2, 4, 5)
    x = qd.tensor(dtype, shape, backend=backend, layout=(2, 0, 1), needs_grad=True)
    y = qd.tensor(dtype, x.shape, backend=backend, layout=(1, 0, 2), needs_grad=True)
    loss = qd.tensor(qd.f32, shape=(), backend=backend, needs_grad=True)
    assert x.grad.shape == x.shape
    assert x.grad.layout == x.layout
    assert y.grad.shape == y.shape
    assert y.grad.layout == y.layout

    @qd.kernel
    def scaled_square(x: qd.Tensor, y: qd.Tensor) -> None:
        for I in qd.grouped(x):
            y[I] = 3 * x[I] * x[I]

    @qd.kernel
    def sum_components(y: qd.Tensor, loss: qd.Tensor) -> None:
        for I in qd.grouped(y):
            loss[()] += y[I].sum()

    x.fill(3)
    loss.fill(0)
    x.grad.fill(0)
    y.grad.fill(0)
    # Start from the explicit scalar loss: d(loss)/d(loss) = 1.
    loss.grad.fill(1)
    scaled_square(x, y)
    sum_components(y, loss)
    total_num_elements = np.prod(shape + x.element_shape)
    assert loss[()] == 27 * total_num_elements

    # Differentiate the sum to obtain y.grad = 1, then y = 3*x*x to obtain x.grad = 6*x = 18.
    sum_components.grad(y, loss)
    scaled_square.grad(x, y)
    np.testing.assert_array_equal(x.grad.to_numpy(), np.full(x.shape + x.element_shape, 18))


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
def test_compound_layout_invalid(backend: qd.Backend, dtype: MatrixType, layout: tuple[int, ...]) -> None:
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
def test_compound_layout_kernel_specialization(backend: qd.Backend, dtype: MatrixType) -> None:
    """Check kernel reuse across layouts using distinct codes for every tensor position and element component."""

    shape = (2, 5)
    element_shape = (dtype.n,) if dtype.ndim == 1 else (dtype.n, dtype.m)
    expected = test_utils.new_numpy_tensor_containing_index_codes(shape, element_shape)

    for layout in (None, (1, 0), None):
        x = qd.tensor(dtype, shape, backend=backend, layout=layout)
        test_utils.fill_index_codes(x, dtype)

        np.testing.assert_array_equal(x.to_numpy(), expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(qd.types.vector(3, qd.f32), id="vector"),
        pytest.param(qd.types.matrix(2, 3, qd.f32), id="matrix"),
    ],
)
@test_utils.test()
def test_compound_layout_numpy_import_rejects_physical_shape(dtype: MatrixType) -> None:
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
def test_compound_layout_torch_import_rejects_physical_shape(dtype: MatrixType) -> None:
    """Torch imports require canonical shape rather than the physical storage shape."""
    shape = (2, 5)
    layout = (1, 0)
    x = qd.tensor(dtype, shape, backend=qd.Backend.NDARRAY, layout=layout)
    wrong_shape = np.zeros((5, 2) + x.element_shape, dtype=np.float32)
    torch = pytest.importorskip("torch")
    source = torch.from_numpy(wrong_shape)

    with pytest.raises(ValueError, match="Mismatch shape"):
        x.from_torch(source)
