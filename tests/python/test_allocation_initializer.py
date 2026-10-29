"""Allocation-time initialization through field, ndarray, and tensor factories."""

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils
from tests.helpers.allocation_initializer import _create_array


# Tests shared by fields, ndarrays, and tensors


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class,element_shape", [("scalar", ()), ("vector", (3,)), ("matrix", (2, 3))])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@pytest.mark.parametrize("kwargs", [{}, {"fill": None}], ids=["omitted", "none"])
@test_utils.test()
def test_initializer_defaults(
    container_type: str,
    backend: qd.Backend | None,
    element_class: str,
    element_shape: tuple[int, ...],
    via_dtype: bool,
    kwargs: dict[str, None],
) -> None:
    # Omitting fill or passing None must preserve zero initialization for every constructor.
    num_elements = 4
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=(num_elements,),
        via_dtype=via_dtype,
        **kwargs,
    )
    np.testing.assert_array_equal(array.to_numpy(), np.zeros((num_elements, *element_shape)))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("fill_value", [0, 0.0, 7, -7, 2.5, -2.5])
@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
@test_utils.test()
def test_initializer_scalar(
    container_type: str, backend: qd.Backend | None, fill_value: int | float, shape: tuple[int, ...]
) -> None:
    # Scalar initializers must initialize every array element to the supplied value.
    array = _create_array(
        container_type=container_type, backend=backend, element_class="scalar", shape=shape, fill=fill_value
    )
    np.testing.assert_array_equal(array.to_numpy(), np.full(shape, fill_value))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class,element_shape", [("vector", (3,)), ("matrix", (2, 3))])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@pytest.mark.parametrize("shape", [(), (4,)])
@test_utils.test()
def test_initializer_broadcast(
    container_type: str,
    backend: qd.Backend | None,
    element_class: str,
    element_shape: tuple[int, ...],
    via_dtype: bool,
    shape: tuple[int, ...],
) -> None:
    # A scalar initializer must fill every component of every allocated element.
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=shape,
        via_dtype=via_dtype,
        fill=7,
    )
    np.testing.assert_array_equal(array.to_numpy(), np.full((*shape, *element_shape), 7))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class,element_shape", [("vector", (3,)), ("matrix", (2, 3))])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_captures_components(
    container_type: str, backend: qd.Backend | None, element_class: str, element_shape: tuple[int, ...], via_dtype: bool
) -> None:
    # Every constructor must capture component lists before later mutations.
    num_elements = 3
    expected = test_utils.new_numpy_tensor_containing_index_codes(shape=(), element_shape=element_shape) + 1
    fill_value = expected.tolist()
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=(num_elements,),
        via_dtype=via_dtype,
        fill=fill_value,
    )
    if element_class == "vector":
        fill_value[0] = 99
    else:
        fill_value[0][0] = 99
    np.testing.assert_array_equal(array.to_numpy(), np.broadcast_to(expected, (num_elements, *expected.shape)))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@test_utils.test()
def test_initializer_captures_matrix(container_type: str, backend: qd.Backend | None) -> None:
    # A Matrix used as an initializer must be captured before later changes to that Matrix.
    expected = [[1, 2, 3], [4, 5, 6]]
    fill_value = qd.Matrix(expected)
    array = _create_array(
        container_type=container_type, backend=backend, element_class="matrix", shape=(), dtype=qd.i32, fill=fill_value
    )
    fill_value[0, 0] = 99
    np.testing.assert_array_equal(array.to_numpy(), expected)


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class,element_shape", [("scalar", ()), ("vector", (3,)), ("matrix", (2, 3))])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_leaves_gradients_zero(
    container_type: str, backend: qd.Backend | None, element_class: str, element_shape: tuple[int, ...], via_dtype: bool
) -> None:
    # Initializing forward values must leave the companion gradient values at zero.
    num_elements = 4
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=(num_elements,),
        via_dtype=via_dtype,
        fill=7,
        needs_grad=True,
    )
    np.testing.assert_array_equal(array.to_numpy(), np.full((num_elements, *element_shape), 7))
    np.testing.assert_array_equal(array.grad.to_numpy(), np.zeros((num_elements, *element_shape)))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("fill_value", [[1, 2], "invalid"])
@test_utils.test()
def test_initializer_invalid_scalar(
    container_type: str, backend: qd.Backend | None, fill_value: list[int] | str
) -> None:
    # Invalid scalar initializers must raise at declaration without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=fill_value)
    array = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=7)
    assert array[()] == 7


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class,element_shape", [("vector", (3,)), ("matrix", (2, 3))])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_invalid_components(
    container_type: str, backend: qd.Backend | None, element_class: str, element_shape: tuple[int, ...], via_dtype: bool
) -> None:
    # Component counts that do not match the element type must be rejected by every constructor.
    # A valid allocation afterward must still initialize correctly, checking that the rejection left no broken state.
    with pytest.raises(ValueError, match="fill"):
        _create_array(
            container_type=container_type,
            backend=backend,
            element_class=element_class,
            shape=(),
            via_dtype=via_dtype,
            fill=[1, 2],
        )
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=(),
        via_dtype=via_dtype,
        fill=7,
    )
    np.testing.assert_array_equal(array.to_numpy(), np.full(element_shape, 7))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("fill_value", [[[1, 2]], [[1, 2], [3]], [[1, None, 3], [4, 5, 6]]])
@test_utils.test()
def test_initializer_invalid_matrix(
    container_type: str, backend: qd.Backend | None, fill_value: list[list[int | None]]
) -> None:
    # Invalid matrix shapes or components must raise without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        _create_array(container_type=container_type, backend=backend, element_class="matrix", shape=(), fill=fill_value)
    array = _create_array(container_type=container_type, backend=backend, element_class="matrix", shape=(), fill=7)
    np.testing.assert_array_equal(array.to_numpy(), np.full((2, 3), 7))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("operation", ["fill", "host", "numpy", "kernel"])
@test_utils.test()
def test_initializer_precedes_writes(container_type: str, backend: qd.Backend | None, operation: str) -> None:
    # Initialization must run before each kind of write.
    num_elements = 4
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class="scalar",
        shape=(num_elements,),
        dtype=qd.i32,
        fill=3,
    )
    if operation == "fill":
        array.fill(5)
    elif operation == "host":
        array[0] = 5
    elif operation == "numpy":
        array.from_numpy(np.full(num_elements, 5, dtype=np.int32))
    elif operation == "kernel":

        @qd.kernel
        def k_write(array: qd.Tensor) -> None:
            for i in array:
                array[i] += 2

        k_write(array)
    assert array[0] == 5


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@test_utils.test()
def test_initializer_first_kernel_read(container_type: str, backend: qd.Backend | None) -> None:
    # The first kernel read must see the allocation-time fill without a prior host read or explicit synchronization.
    # After an explicit .fill(5), the same kernel must see the updated value rather than the original initializer.
    array = _create_array(
        container_type=container_type, backend=backend, element_class="scalar", shape=(4,), dtype=qd.i32, fill=3
    )

    @qd.kernel
    def k_read(array: qd.Tensor) -> qd.i32:
        return array[0]

    assert k_read(array) == 3
    array.fill(5)
    assert k_read(array) == 5


@pytest.mark.needs_torch
@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class,element_shape", [("scalar", ()), ("vector", (3,)), ("matrix", (2, 3))])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_dlpack(
    container_type: str, backend: qd.Backend | None, element_class: str, element_shape: tuple[int, ...], via_dtype: bool
) -> None:
    # After synchronization, DLPack consumers must see initialized values from every allocation API.
    is_field = container_type == "field" or (container_type == "tensor" and backend == qd.Backend.FIELD)
    torch = test_utils.require_torch_dlpack(need_field_support=is_field)
    shape = (2, 3)
    # Offset the index codes so the scalar initializer is nonzero too.
    fill_value = (
        test_utils.new_numpy_tensor_containing_index_codes(shape=(), element_shape=element_shape) + 1
    ).tolist()
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=shape,
        via_dtype=via_dtype,
        fill=fill_value,
    )
    qd.sync()
    torch_view = torch.utils.dlpack.from_dlpack(array.to_dlpack())
    np.testing.assert_array_equal(torch_view.cpu().numpy(), np.broadcast_to(fill_value, (*shape, *element_shape)))


# Tensor-only tests


@test_utils.test()
def test_initializer_tensor_default_backend() -> None:
    # The tensor factory must initialize its default ndarray backend and preserve later fills.
    tensor = qd.tensor(dtype=qd.i32, shape=(), fill=7)
    assert isinstance(tensor, qd.Tensor)
    assert isinstance(tensor._unwrap(), qd.Ndarray)
    assert tensor[()] == 7
    tensor.fill(3)
    assert tensor[()] == 3


@pytest.mark.parametrize("backend", [qd.Backend.FIELD, qd.Backend.NDARRAY], ids=["tensor_field", "tensor_ndarray"])
@pytest.mark.parametrize("element_class,element_shape", [("scalar", ()), ("vector", (3,)), ("matrix", (2, 3))])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_tensor_layout(
    backend: qd.Backend, element_class: str, element_shape: tuple[int, ...], via_dtype: bool
) -> None:
    # Initialization and later writes must use canonical indices when the physical shape axes are permuted.
    shape = (2, 3)
    # Offset the index codes so the scalar initializer is nonzero too.
    fill_value = (
        test_utils.new_numpy_tensor_containing_index_codes(shape=(), element_shape=element_shape) + 1
    ).tolist()
    tensor = _create_array(
        container_type="tensor",
        backend=backend,
        element_class=element_class,
        shape=shape,
        layout=(1, 0),
        via_dtype=via_dtype,
        fill=fill_value,
        needs_grad=True,
    )
    assert isinstance(tensor, qd.Tensor)
    expected = np.broadcast_to(fill_value, (*shape, *element_shape)).copy()
    np.testing.assert_array_equal(tensor.to_numpy(), expected)
    np.testing.assert_array_equal(tensor.grad.to_numpy(), np.zeros_like(expected))

    @qd.kernel
    def k_write(tensor: qd.Tensor) -> None:
        tensor[1, 2] += 2

    k_write(tensor)
    expected[1, 2] += 2
    np.testing.assert_array_equal(tensor.to_numpy(), expected)
