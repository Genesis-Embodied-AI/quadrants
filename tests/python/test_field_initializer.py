import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests.helpers import test_utils


@test_utils.test()
def test_fill_initializer_shares_tree():
    # Declarations with fill= must defer compilation, share one allocation tree, and initialize every field.
    fields = [qd.field(qd.f32, shape=(), fill=i) for i in range(40)]
    total = qd.field(qd.f32, shape=())
    assert impl.get_runtime().get_num_compiled_functions() == 0

    @qd.kernel
    def add_field(field: qd.Template):
        total[()] += field[()]

    for field in fields:
        add_field(field)
    assert total[()] == 780
    assert len({field.snode._snode_tree_id for field in fields + [total]}) == 1


@pytest.mark.parametrize("operation", ["fill", "host", "numpy", "kernel"])
@test_utils.test()
def test_fill_initializer_precedes_writes(operation):
    # Initialization must run before each kind of write.
    field = qd.field(qd.i32, shape=4, fill=3)
    if operation == "fill":
        field.fill(5)
    elif operation == "host":
        field[0] = 5
    elif operation == "numpy":
        field.from_numpy(np.full(4, 5, dtype=np.int32))
    elif operation == "kernel":

        @qd.kernel
        def k_write():
            for i in field:
                field[i] += 2

        k_write()
    assert field[0] == 5


@test_utils.test()
def test_fill_initializer_is_not_replayed():
    # Once initialization has run, later synchronization must preserve writes made afterward.
    field = qd.field(qd.i32, shape=(), fill=3)
    assert field[()] == 3
    field[()] = 5

    # Each sync calls materialize() again. Repeating it must not replay the initializer over the later write.
    qd.sync()
    assert field[()] == 5
    qd.sync()
    assert field[()] == 5


@test_utils.test()
def test_fill_initializer_after_materialization():
    # Fields declared after an earlier allocation must initialize together in a new allocation tree.
    first = qd.field(qd.i32, shape=(), fill=2)
    assert first[()] == 2
    second = qd.field(qd.i32, shape=(), fill=4)
    third = qd.field(qd.i32, shape=(), fill=6)
    assert second[()] == 4
    assert third[()] == 6
    assert second.snode._snode_tree_id == third.snode._snode_tree_id
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


@pytest.mark.parametrize("factory", ["vector", "matrix", "vector_type", "matrix_type"])
@test_utils.test()
def test_fill_initializer_components(factory):
    # All vector and matrix constructors must capture component lists before later mutations.
    num_elements = 3
    value = [1, 2] if "vector" in factory else [[1, 2], [3, 4]]
    expected = np.array(value)
    if factory == "vector":
        field = qd.Vector.field(2, qd.i32, shape=num_elements, fill=value)
    elif factory == "matrix":
        field = qd.Matrix.field(2, 2, qd.i32, shape=num_elements, fill=value)
    elif factory == "vector_type":
        field = qd.field(qd.types.vector(2, qd.i32), shape=num_elements, fill=value)
    else:
        field = qd.field(qd.types.matrix(2, 2, qd.i32), shape=num_elements, fill=value)
    if "vector" in factory:
        value[0] = 99
    else:
        value[0][0] = 99
    for i in range(num_elements):
        np.testing.assert_array_equal(field[i].to_numpy(), expected)


@pytest.mark.parametrize("factory", ["vector", "matrix", "vector_type", "matrix_type"])
@test_utils.test()
def test_fill_initializer_broadcast(factory):
    # A scalar initializer must fill every component of every vector or matrix element.
    num_elements = 3
    if factory == "vector":
        field = qd.Vector.field(2, qd.i32, shape=num_elements, fill=9)
    elif factory == "matrix":
        field = qd.Matrix.field(2, 2, qd.i32, shape=num_elements, fill=9)
    elif factory == "vector_type":
        field = qd.field(qd.types.vector(2, qd.i32), shape=num_elements, fill=9)
    elif factory == "matrix_type":
        field = qd.field(qd.types.matrix(2, 2, qd.i32), shape=num_elements, fill=9)
    expected = np.full((2,) if "vector" in factory else (2, 2), 9)
    for i in range(num_elements):
        np.testing.assert_array_equal(field[i].to_numpy(), expected)


@test_utils.test()
def test_fill_initializer_matrix_value_snapshot():
    # A Matrix used as an initializer must be captured before later changes to that Matrix.
    value = qd.Matrix([[1, 2], [3, 4]])
    field = qd.Matrix.field(2, 2, qd.i32, shape=(), fill=value)
    value[0, 0] = 99
    np.testing.assert_array_equal(field[()].to_numpy(), [[1, 2], [3, 4]])


@test_utils.test()
def test_fill_initializer_manual_placement():
    # A field declared without a shape must receive its initializer after manual placement.
    field = qd.field(qd.i32, fill=8)
    qd.root.dense(qd.i, 4).place(field)
    np.testing.assert_array_equal(field.to_numpy(), [8] * 4)


@test_utils.test()
def test_fill_initializer_gradients_and_offset():
    # Initialization must respect the index offset and leave gradient and dual fields at zero.
    field = qd.field(qd.f32, shape=4, offset=-2, needs_grad=True, needs_dual=True, fill=2)
    assert field[-2] == 2
    assert field[1] == 2
    assert field.grad[-2] == 0
    assert field.dual[-2] == 0


@test_utils.test()
def test_fill_initializer_reset():
    # Reset must discard pending initializers and allow fresh fields to initialize in the new runtime.
    qd.field(qd.i32, shape=(), fill=8)
    runtime = impl.get_runtime()
    arch = qd.cfg.arch
    qd.reset()
    assert not runtime.field_initializers.pending
    qd.init(arch=arch)
    field = qd.field(qd.i32, shape=(), fill=4)
    assert field[()] == 4


@pytest.mark.parametrize("value", [[1, 2], "invalid"])
@test_utils.test()
def test_fill_initializer_invalid_scalar(value):
    # Invalid scalar initializers must raise at declaration without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        qd.field(qd.i32, shape=(), fill=value)
    field = qd.field(qd.i32, shape=(), fill=7)
    assert field[()] == 7


@pytest.mark.parametrize("value", [[[1, 2]], [[1, 2], [3]], [1, 2], [[1, None], [2, 3]]])
@test_utils.test()
def test_fill_initializer_invalid_matrix(value):
    # Invalid matrix shapes or components must raise without preventing later valid allocations.
    with pytest.raises(ValueError, match="fill"):
        qd.Matrix.field(2, 2, qd.i32, shape=(), fill=value)
    field = qd.field(qd.i32, shape=(), fill=7)
    assert field[()] == 7


@test_utils.test()
def test_fill_initializer_preserves_immediate_fill():
    # Calling .fill() must still compile immediately, placing later declarations in a separate tree.
    first = qd.field(qd.i32, shape=())
    first.fill(3)
    assert impl.get_runtime().get_num_compiled_functions() > 0
    second = qd.field(qd.i32, shape=(), fill=4)
    assert second[()] == 4
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


@pytest.mark.needs_torch
@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.metal])
def test_fill_initializer_dlpack(matrix):
    # After synchronization, PyTorch must see initialized scalar and vector values through DLPack.
    torch = pytest.importorskip("torch")
    if qd.cfg.arch == qd.metal:
        from quadrants.lang.field import _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET

        if not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET:
            pytest.skip("Metal DLPack requires torch >= 2.9.2")
    field = qd.Vector.field(2, qd.f32, shape=4, fill=[2, 3]) if matrix else qd.field(qd.f32, shape=4, fill=7)
    qd.sync()
    tensor = torch.utils.dlpack.from_dlpack(field.to_dlpack())
    expected = [[2, 3]] * 4 if matrix else [7] * 4
    np.testing.assert_array_equal(tensor.cpu().numpy(), expected)


@pytest.mark.needs_torch
@pytest.mark.parametrize("kind", ["scalar", "vector", "matrix", "vector_type", "matrix_type"])
def test_fill_initializer_python_backend(kind):
    # The Python backend must initialize scalar, vector, and matrix fields for each supported constructor.
    pytest.importorskip("torch")
    qd.init(arch=qd.python)
    try:
        if kind == "scalar":
            field = qd.field(qd.i32, shape=3, fill=7)
            expected = [7] * 3
        elif kind == "vector":
            field = qd.Vector.field(2, qd.i32, shape=3, fill=[2, 3])
            expected = [[2, 3]] * 3
        elif kind == "matrix":
            field = qd.Matrix.field(2, 2, qd.i32, shape=3, fill=[[1, 2], [3, 4]])
            expected = [[[1, 2], [3, 4]]] * 3
        elif kind == "vector_type":
            field = qd.field(qd.types.vector(2, qd.i32), shape=3, fill=7)
            expected = [[7, 7]] * 3
        else:
            field = qd.field(qd.types.matrix(2, 2, qd.i32), shape=3, fill=7)
            expected = [[[7, 7], [7, 7]]] * 3
        np.testing.assert_array_equal(field.to_numpy(), expected)
    finally:
        qd.reset()
