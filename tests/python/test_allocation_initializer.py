"""Allocation-time initialization through field, ndarray, and tensor factories."""

from typing import Any

import numpy as np
import pytest

import quadrants as qd
from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants.lang import impl

from tests.helpers import test_utils


def _create_array(
    container_type: str,
    backend: qd.Backend | None,
    element_class: str,
    shape: tuple[int, ...],
    *,
    dtype: DataTypeCxx = qd.f32,
    via_dtype: bool = False,
    **kwargs: Any,
) -> qd.Field | qd.Ndarray | qd.Tensor:
    if container_type == "field":
        scalar_factory, vector_factory, matrix_factory = qd.field, qd.Vector.field, qd.Matrix.field
    elif container_type == "ndarray":
        scalar_factory, vector_factory, matrix_factory = qd.ndarray, qd.Vector.ndarray, qd.Matrix.ndarray
    elif container_type == "tensor":
        scalar_factory, vector_factory, matrix_factory = qd.tensor, qd.Vector.tensor, qd.Matrix.tensor
        kwargs["backend"] = backend
    else:
        raise ValueError(f"Unknown container type: {container_type}")

    if element_class == "scalar":
        return scalar_factory(dtype=dtype, shape=shape, **kwargs)
    if element_class == "vector":
        if via_dtype:
            return scalar_factory(dtype=qd.types.vector(n=3, dtype=dtype), shape=shape, **kwargs)
        return vector_factory(n=3, dtype=dtype, shape=shape, **kwargs)
    if element_class == "matrix":
        if via_dtype:
            return scalar_factory(dtype=qd.types.matrix(n=2, m=3, dtype=dtype), shape=shape, **kwargs)
        return matrix_factory(n=2, m=3, dtype=dtype, shape=shape, **kwargs)
    raise ValueError(f"Unknown element class: {element_class}")


def _initial_value(element_class: str) -> float | list[int] | list[list[int]]:
    if element_class == "scalar":
        return -2.5
    if element_class == "vector":
        return [1, 2, 3]
    # elif "matrix"
    return [[1, 2, 3], [4, 5, 6]]


def _get_underlying_field(array: qd.Field | qd.Tensor) -> qd.Field:
    # Return the underlying field so tests can inspect its allocation tree, unwrapping a Tensor if needed.
    return array._unwrap() if isinstance(array, qd.Tensor) else array


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class", ["scalar", "vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@pytest.mark.parametrize("kwargs", [{}, {"fill": None}], ids=["omitted", "none"])
@test_utils.test()
def test_initializer_defaults(
    container_type: str, backend: qd.Backend | None, element_class: str, via_dtype: bool, kwargs: dict[str, None]
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
    element_shape = np.shape(_initial_value(element_class))
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
@pytest.mark.parametrize("value", [0, 0.0, 7, -7, 2.5, -2.5])
@pytest.mark.parametrize("shape", [(), (4,), (2, 3)])
@test_utils.test()
def test_initializer_scalar(
    container_type: str, backend: qd.Backend | None, value: int | float, shape: tuple[int, ...]
) -> None:
    # Scalar initializers must initialize every array element to the supplied value.
    array = _create_array(
        container_type=container_type, backend=backend, element_class="scalar", shape=shape, fill=value
    )
    np.testing.assert_array_equal(array.to_numpy(), np.full(shape, value))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("element_class", ["vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@pytest.mark.parametrize("shape", [(), (4,)])
@test_utils.test()
def test_initializer_broadcast(
    container_type: str, backend: qd.Backend | None, element_class: str, via_dtype: bool, shape: tuple[int, ...]
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
    element_shape = np.shape(_initial_value(element_class))
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
@pytest.mark.parametrize("element_class", ["vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_captures_components(
    container_type: str, backend: qd.Backend | None, element_class: str, via_dtype: bool
) -> None:
    # Every constructor must capture component lists before later mutations.
    num_elements = 3
    value = _initial_value(element_class)
    expected = np.array(value)
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=(num_elements,),
        via_dtype=via_dtype,
        fill=value,
    )
    if element_class == "vector":
        value[0] = 99
    else:
        value[0][0] = 99
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
    value = qd.Matrix(expected)
    array = _create_array(
        container_type=container_type, backend=backend, element_class="matrix", shape=(), dtype=qd.i32, fill=value
    )
    value[0, 0] = 99
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
@pytest.mark.parametrize("element_class", ["scalar", "vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_leaves_gradients_zero(
    container_type: str, backend: qd.Backend | None, element_class: str, via_dtype: bool
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
    element_shape = np.shape(_initial_value(element_class))
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
@pytest.mark.parametrize("value", [[1, 2], "invalid"])
@test_utils.test()
def test_initializer_invalid_scalar(container_type: str, backend: qd.Backend | None, value: list[int] | str) -> None:
    # Invalid scalar initializers must raise at declaration without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=value)
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
@pytest.mark.parametrize("element_class", ["vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_invalid_components(
    container_type: str, backend: qd.Backend | None, element_class: str, via_dtype: bool
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
    np.testing.assert_array_equal(array.to_numpy(), np.full(np.shape(_initial_value(element_class)), 7))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("ndarray", None, id="ndarray"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
        pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
    ],
)
@pytest.mark.parametrize("value", [[[1, 2]], [[1, 2], [3]], [[1, None, 3], [4, 5, 6]]])
@test_utils.test()
def test_initializer_invalid_matrix(
    container_type: str, backend: qd.Backend | None, value: list[list[int | None]]
) -> None:
    # Invalid matrix shapes or components must raise without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        _create_array(container_type=container_type, backend=backend, element_class="matrix", shape=(), fill=value)
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
@pytest.mark.parametrize("element_class", ["scalar", "vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_dlpack(
    container_type: str, backend: qd.Backend | None, element_class: str, via_dtype: bool
) -> None:
    # After synchronization, DLPack consumers must see initialized values from every allocation API.
    is_field = container_type == "field" or (container_type == "tensor" and backend == qd.Backend.FIELD)
    torch = test_utils.require_torch_dlpack(need_field_support=is_field)
    shape = (2, 3)
    value = _initial_value(element_class)
    array = _create_array(
        container_type=container_type,
        backend=backend,
        element_class=element_class,
        shape=shape,
        via_dtype=via_dtype,
        fill=value,
    )
    qd.sync()
    torch_view = torch.utils.dlpack.from_dlpack(array.to_dlpack())
    np.testing.assert_array_equal(torch_view.cpu().numpy(), np.broadcast_to(value, (*shape, *np.shape(value))))


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
    ],
)
@test_utils.test()
def test_initializer_fields_share_tree(container_type: str, backend: qd.Backend | None) -> None:
    # Field declarations with fill= must defer compilation, share one allocation tree, and initialize every field.
    num_fields = 40
    fields = [
        _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=i)
        for i in range(num_fields)
    ]
    total = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=())
    assert impl.get_runtime().get_num_compiled_functions() == 0

    @qd.kernel
    def k_add_field(field: qd.Tensor, total: qd.Tensor) -> None:
        total[()] += field[()]

    for field in fields:
        k_add_field(field, total)
    assert total[()] == sum(range(num_fields))
    assert len({_get_underlying_field(field).snode._snode_tree_id for field in fields + [total]}) == 1


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
    ],
)
@test_utils.test()
def test_field_initializer_is_not_replayed(container_type: str, backend: qd.Backend | None) -> None:
    # Once field initialization has run, later synchronization must preserve writes made afterward.
    field = _create_array(
        container_type=container_type, backend=backend, element_class="scalar", shape=(), dtype=qd.i32, fill=3
    )
    assert field[()] == 3
    field[()] = 5

    # Each sync calls materialize() again. Repeating it must not replay the field initializer over the later write.
    qd.sync()
    assert field[()] == 5
    qd.sync()
    assert field[()] == 5


@pytest.mark.parametrize(
    "container_type,backend",
    [
        pytest.param("field", None, id="field"),
        pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
    ],
)
@test_utils.test()
def test_field_initializer_after_materialization(container_type: str, backend: qd.Backend | None) -> None:
    # Fields declared after an earlier allocation must initialize together in a new allocation tree.
    first = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=2)
    assert first[()] == 2
    second = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=4)
    third = _create_array(container_type=container_type, backend=backend, element_class="scalar", shape=(), fill=6)
    assert second[()] == 4
    assert third[()] == 6
    assert _get_underlying_field(second).snode._snode_tree_id == _get_underlying_field(third).snode._snode_tree_id
    assert _get_underlying_field(first).snode._snode_tree_id != _get_underlying_field(second).snode._snode_tree_id


@test_utils.test()
def test_field_initializer_manual_placement() -> None:
    # A field declared without a shape must receive its initializer after manual placement.
    num_elements = 4
    field = qd.field(dtype=qd.i32, fill=8)
    qd.root.dense(qd.i, num_elements).place(field)
    np.testing.assert_array_equal(field.to_numpy(), [8] * num_elements)


@test_utils.test()
def test_field_initializer_gradients_and_offset() -> None:
    # Field initialization must respect the index offset and leave gradient and dual fields at zero.
    field = qd.field(dtype=qd.f32, shape=4, offset=-2, needs_grad=True, needs_dual=True, fill=2)
    assert field[-2] == 2
    assert field[1] == 2
    assert field.grad[-2] == 0
    assert field.dual[-2] == 0


@test_utils.test()
def test_field_initializer_reset() -> None:
    # Reset must discard pending field initializers and allow fresh fields to initialize in the new runtime.
    qd.field(dtype=qd.i32, shape=(), fill=8)
    runtime = impl.get_runtime()
    arch = qd.cfg.arch
    qd.reset()
    assert not runtime.field_initializers.pending
    qd.init(arch=arch)
    field = qd.field(dtype=qd.i32, shape=(), fill=4)
    assert field[()] == 4


@test_utils.test()
def test_field_initializer_preserves_immediate_fill() -> None:
    # Calling .fill() on a field must still compile immediately, placing later field declarations in a separate tree.
    first = qd.field(dtype=qd.i32, shape=())
    first.fill(3)
    assert impl.get_runtime().get_num_compiled_functions() > 0
    second = qd.field(dtype=qd.i32, shape=(), fill=4)
    assert second[()] == 4
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


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
@pytest.mark.parametrize("element_class", ["scalar", "vector", "matrix"])
@pytest.mark.parametrize("via_dtype", [False, True], ids=["via_class", "via_dtype"])
@test_utils.test()
def test_initializer_tensor_layout(backend: qd.Backend, element_class: str, via_dtype: bool) -> None:
    # Initialization and later writes must use canonical indices when the physical shape axes are permuted.
    shape = (2, 3)
    value = _initial_value(element_class)
    tensor = _create_array(
        container_type="tensor",
        backend=backend,
        element_class=element_class,
        shape=shape,
        layout=(1, 0),
        via_dtype=via_dtype,
        fill=value,
        needs_grad=True,
    )
    assert isinstance(tensor, qd.Tensor)
    expected = np.broadcast_to(value, (*shape, *np.shape(value))).copy()
    np.testing.assert_array_equal(tensor.to_numpy(), expected)
    np.testing.assert_array_equal(tensor.grad.to_numpy(), np.zeros_like(expected))

    @qd.kernel
    def k_write(tensor: qd.Tensor) -> None:
        tensor[1, 2] += 2

    k_write(tensor)
    expected[1, 2] += 2
    np.testing.assert_array_equal(tensor.to_numpy(), expected)
