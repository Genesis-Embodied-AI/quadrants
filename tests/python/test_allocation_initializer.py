"""Allocation-time initialization through field, ndarray, and tensor factories."""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests.helpers import test_utils

APIS = [
    pytest.param("field", None, id="field"),
    pytest.param("ndarray", None, id="ndarray"),
    pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
    pytest.param("tensor", qd.Backend.NDARRAY, id="tensor_ndarray"),
]
FIELD_APIS = [
    pytest.param("field", None, id="field"),
    pytest.param("tensor", qd.Backend.FIELD, id="tensor_field"),
]
VECTOR_MATRIX_KINDS = ["vector", "matrix", "vector_type", "matrix_type"]
KINDS = ["scalar", *VECTOR_MATRIX_KINDS]


def _allocate(api, backend, kind, shape, *, dtype=qd.f32, **kwargs):
    if api == "field":
        scalar_factory, vector_factory, matrix_factory = qd.field, qd.Vector.field, qd.Matrix.field
    elif api == "ndarray":
        scalar_factory, vector_factory, matrix_factory = qd.ndarray, qd.Vector.ndarray, qd.Matrix.ndarray
    elif api == "tensor":
        scalar_factory, vector_factory, matrix_factory = qd.tensor, qd.Vector.tensor, qd.Matrix.tensor
        kwargs["backend"] = backend
    else:
        raise ValueError(f"Unknown allocation API: {api}")

    if kind == "scalar":
        return scalar_factory(dtype, shape=shape, **kwargs)
    if kind == "vector":
        return vector_factory(3, dtype, shape=shape, **kwargs)
    if kind == "matrix":
        return matrix_factory(2, 3, dtype, shape=shape, **kwargs)
    if kind == "vector_type":
        return scalar_factory(qd.types.vector(3, dtype), shape=shape, **kwargs)
    if kind == "matrix_type":
        return scalar_factory(qd.types.matrix(2, 3, dtype), shape=shape, **kwargs)
    raise ValueError(f"Unknown element kind: {kind}")


def _initial_value(kind):
    if kind == "scalar":
        return -2.5
    if "vector" in kind:
        return [1, 2, 3]
    return [[1, 2, 3], [4, 5, 6]]


def _storage(array):
    return array._unwrap() if isinstance(array, qd.Tensor) else array


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("kwargs", [{}, {"fill": None}], ids=["omitted", "none"])
@test_utils.test()
def test_initializer_defaults(api, backend, kind, kwargs):
    # Omitting fill or passing None must preserve zero initialization for every constructor.
    num_elements = 4
    array = _allocate(api, backend, kind, (num_elements,), **kwargs)
    element_shape = np.shape(_initial_value(kind))
    np.testing.assert_array_equal(array.to_numpy(), np.zeros((num_elements, *element_shape)))


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("value", [0, -2.5])
@test_utils.test()
def test_initializer_scalar(api, backend, value):
    # Scalar initializers must preserve zero and negative fractional values.
    shape = (2, 3)
    array = _allocate(api, backend, "scalar", shape, fill=value)
    np.testing.assert_array_equal(array.to_numpy(), np.full(shape, value))


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(), (4,)])
@test_utils.test()
def test_initializer_broadcast(api, backend, kind, shape):
    # A scalar initializer must fill every component of every allocated element.
    array = _allocate(api, backend, kind, shape, fill=7)
    element_shape = np.shape(_initial_value(kind))
    np.testing.assert_array_equal(array.to_numpy(), np.full((*shape, *element_shape), 7))


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("kind", VECTOR_MATRIX_KINDS)
@test_utils.test()
def test_initializer_captures_components(api, backend, kind):
    # Every constructor must capture component lists before later mutations.
    num_elements = 3
    value = _initial_value(kind)
    expected = np.array(value)
    array = _allocate(api, backend, kind, (num_elements,), fill=value)
    if "vector" in kind:
        value[0] = 99
    else:
        value[0][0] = 99
    np.testing.assert_array_equal(array.to_numpy(), np.broadcast_to(expected, (num_elements, *expected.shape)))


@pytest.mark.parametrize("api,backend", APIS)
@test_utils.test()
def test_initializer_captures_matrix(api, backend):
    # A Matrix used as an initializer must be captured before later changes to that Matrix.
    expected = [[1, 2, 3], [4, 5, 6]]
    value = qd.Matrix(expected)
    array = _allocate(api, backend, "matrix", (), dtype=qd.i32, fill=value)
    value[0, 0] = 99
    np.testing.assert_array_equal(array.to_numpy(), expected)


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("kind", KINDS)
@test_utils.test()
def test_initializer_leaves_gradients_zero(api, backend, kind):
    # Initializing forward values must leave the companion gradient values at zero.
    num_elements = 4
    array = _allocate(api, backend, kind, (num_elements,), fill=7, needs_grad=True)
    element_shape = np.shape(_initial_value(kind))
    np.testing.assert_array_equal(array.to_numpy(), np.full((num_elements, *element_shape), 7))
    np.testing.assert_array_equal(array.grad.to_numpy(), np.zeros((num_elements, *element_shape)))


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("value", [[1, 2], "invalid"])
@test_utils.test()
def test_initializer_invalid_scalar(api, backend, value):
    # Invalid scalar initializers must raise at declaration without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        _allocate(api, backend, "scalar", (), fill=value)
    array = _allocate(api, backend, "scalar", (), fill=7)
    assert array[()] == 7


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("kind", VECTOR_MATRIX_KINDS)
@test_utils.test()
def test_initializer_invalid_components(api, backend, kind):
    # Component counts that do not match the element type must be rejected by every constructor.
    with pytest.raises(ValueError, match="fill"):
        _allocate(api, backend, kind, (), fill=[1, 2])
    array = _allocate(api, backend, kind, (), fill=7)
    np.testing.assert_array_equal(array.to_numpy(), np.full(np.shape(_initial_value(kind)), 7))


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("value", [[[1, 2]], [[1, 2], [3]], [[1, None, 3], [4, 5, 6]]])
@test_utils.test()
def test_initializer_invalid_matrix(api, backend, value):
    # Invalid matrix shapes or components must raise without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        _allocate(api, backend, "matrix", (), fill=value)
    array = _allocate(api, backend, "matrix", (), fill=7)
    np.testing.assert_array_equal(array.to_numpy(), np.full((2, 3), 7))


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("operation", ["fill", "host", "numpy", "kernel"])
@test_utils.test()
def test_initializer_precedes_writes(api, backend, operation):
    # Initialization must run before each kind of write.
    num_elements = 4
    array = _allocate(api, backend, "scalar", (num_elements,), dtype=qd.i32, fill=3)
    if operation == "fill":
        array.fill(5)
    elif operation == "host":
        array[0] = 5
    elif operation == "numpy":
        array.from_numpy(np.full(num_elements, 5, dtype=np.int32))
    elif operation == "kernel":

        @qd.kernel
        def k_write(array: qd.Tensor):
            for i in array:
                array[i] += 2

        k_write(array)
    assert array[0] == 5


@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("dtype", [qd.i16, qd.i32, qd.f32])
@test_utils.test()
def test_initializer_first_kernel_read(api, backend, dtype):
    # The first kernel must see the initializer, and a later kernel must see an explicit fill.
    array = _allocate(api, backend, "scalar", (4,), dtype=dtype, fill=3)

    @qd.kernel
    def k_read(array: qd.Tensor) -> qd.i32:
        return qd.cast(array[0], qd.i32)

    assert k_read(array) == 3
    array.fill(5)
    assert k_read(array) == 5


@pytest.mark.needs_torch
@pytest.mark.parametrize("api,backend", APIS)
@pytest.mark.parametrize("kind", KINDS)
@test_utils.test()
def test_initializer_dlpack(api, backend, kind):
    # After synchronization, DLPack consumers must see initialized values from every allocation API.
    is_field = api == "field" or (api == "tensor" and backend == qd.Backend.FIELD)
    torch = test_utils.require_torch_dlpack(need_field_support=is_field)
    shape = (2, 3)
    value = _initial_value(kind)
    kwargs = {"layout": (1, 0)} if api == "tensor" else {}
    array = _allocate(api, backend, kind, shape, fill=value, **kwargs)
    qd.sync()
    torch_view = torch.utils.dlpack.from_dlpack(array.to_dlpack())
    np.testing.assert_array_equal(torch_view.cpu().numpy(), np.broadcast_to(value, (*shape, *np.shape(value))))


@pytest.mark.parametrize("api,backend", FIELD_APIS)
@test_utils.test()
def test_initializer_fields_share_tree(api, backend):
    # Declarations with fill= must defer compilation, share one allocation tree, and initialize every field.
    num_fields = 40
    fields = [_allocate(api, backend, "scalar", (), fill=i) for i in range(num_fields)]
    total = _allocate(api, backend, "scalar", ())
    assert impl.get_runtime().get_num_compiled_functions() == 0

    @qd.kernel
    def k_add_field(field: qd.Tensor, total: qd.Tensor):
        total[()] += field[()]

    for field in fields:
        k_add_field(field, total)
    assert total[()] == sum(range(num_fields))
    assert len({_storage(field).snode._snode_tree_id for field in fields + [total]}) == 1


@pytest.mark.parametrize("api,backend", FIELD_APIS)
@test_utils.test()
def test_initializer_is_not_replayed(api, backend):
    # Once initialization has run, later synchronization must preserve writes made afterward.
    field = _allocate(api, backend, "scalar", (), dtype=qd.i32, fill=3)
    assert field[()] == 3
    field[()] = 5

    # Each sync calls materialize() again. Repeating it must not replay the initializer over the later write.
    qd.sync()
    assert field[()] == 5
    qd.sync()
    assert field[()] == 5


@pytest.mark.parametrize("api,backend", FIELD_APIS)
@test_utils.test()
def test_initializer_after_materialization(api, backend):
    # Fields declared after an earlier allocation must initialize together in a new allocation tree.
    first = _allocate(api, backend, "scalar", (), fill=2)
    assert first[()] == 2
    second = _allocate(api, backend, "scalar", (), fill=4)
    third = _allocate(api, backend, "scalar", (), fill=6)
    assert second[()] == 4
    assert third[()] == 6
    assert _storage(second).snode._snode_tree_id == _storage(third).snode._snode_tree_id
    assert _storage(first).snode._snode_tree_id != _storage(second).snode._snode_tree_id


@test_utils.test()
def test_initializer_manual_placement():
    # A field declared without a shape must receive its initializer after manual placement.
    num_elements = 4
    field = qd.field(qd.i32, fill=8)
    qd.root.dense(qd.i, num_elements).place(field)
    np.testing.assert_array_equal(field.to_numpy(), [8] * num_elements)


@test_utils.test()
def test_initializer_gradients_and_offset():
    # Initialization must respect the index offset and leave gradient and dual fields at zero.
    field = qd.field(qd.f32, shape=4, offset=-2, needs_grad=True, needs_dual=True, fill=2)
    assert field[-2] == 2
    assert field[1] == 2
    assert field.grad[-2] == 0
    assert field.dual[-2] == 0


@test_utils.test()
def test_initializer_reset():
    # Reset must discard pending initializers and allow fresh fields to initialize in the new runtime.
    qd.field(qd.i32, shape=(), fill=8)
    runtime = impl.get_runtime()
    arch = qd.cfg.arch
    qd.reset()
    assert not runtime.field_initializers.pending
    qd.init(arch=arch)
    field = qd.field(qd.i32, shape=(), fill=4)
    assert field[()] == 4


@test_utils.test()
def test_initializer_preserves_immediate_fill():
    # Calling .fill() must still compile immediately, placing later declarations in a separate tree.
    first = qd.field(qd.i32, shape=())
    first.fill(3)
    assert impl.get_runtime().get_num_compiled_functions() > 0
    second = qd.field(qd.i32, shape=(), fill=4)
    assert second[()] == 4
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


@test_utils.test()
def test_initializer_tensor_default_backend():
    # The tensor factory must initialize its default ndarray backend and preserve later fills.
    tensor = qd.tensor(qd.i32, shape=(), fill=7)
    assert isinstance(tensor, qd.Tensor)
    assert isinstance(tensor._unwrap(), qd.Ndarray)
    assert tensor[()] == 7
    tensor.fill(3)
    assert tensor[()] == 3


@pytest.mark.parametrize("backend", [qd.Backend.FIELD, qd.Backend.NDARRAY], ids=["tensor_field", "tensor_ndarray"])
@pytest.mark.parametrize("kind", KINDS)
@test_utils.test()
def test_initializer_tensor_layout(backend, kind):
    # Initialization and later writes must use canonical indices when the physical shape axes are permuted.
    shape = (2, 3)
    value = _initial_value(kind)
    tensor = _allocate("tensor", backend, kind, shape, layout=(1, 0), fill=value, needs_grad=True)
    assert isinstance(tensor, qd.Tensor)
    expected = np.broadcast_to(value, (*shape, *np.shape(value))).copy()
    np.testing.assert_array_equal(tensor.to_numpy(), expected)
    np.testing.assert_array_equal(tensor.grad.to_numpy(), np.zeros_like(expected))

    @qd.kernel
    def k_write(tensor: qd.Tensor):
        tensor[1, 2] += 2

    k_write(tensor)
    expected[1, 2] += 2
    np.testing.assert_array_equal(tensor.to_numpy(), expected)
