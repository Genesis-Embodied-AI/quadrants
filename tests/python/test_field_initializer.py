import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests.helpers import test_utils


@test_utils.test()
def test_fill_initializer_shares_tree():
    fields = [qd.field(qd.f32, shape=(), fill=i) for i in range(40)]
    total = qd.field(qd.f32, shape=())
    assert impl.get_runtime().get_num_compiled_functions() == 0

    @qd.kernel
    def add_field(field: qd.Template):
        total[None] += field[None]

    for field in fields:
        add_field(field)
    assert total[None] == 780
    assert len({field.snode._snode_tree_id for field in fields + [total]}) == 1


@pytest.mark.parametrize("operation", ["fill", "host", "numpy", "kernel"])
@test_utils.test()
def test_fill_initializer_precedes_writes(operation):
    field = qd.field(qd.i32, shape=4, fill=3)
    other = qd.field(qd.i32, shape=(), fill=7)
    if operation == "fill":
        field.fill(5)
    elif operation == "host":
        field[0] = 5
    elif operation == "numpy":
        field.from_numpy(np.full(4, 5, dtype=np.int32))
    else:

        @qd.kernel
        def write():
            for i in field:
                field[i] += 2

        write()
    qd.sync()
    qd.sync()
    assert field[0] == 5
    assert field[1] == (3 if operation == "host" else 5)
    assert other[None] == 7


@test_utils.test()
def test_fill_initializer_after_materialization():
    first = qd.field(qd.i32, shape=(), fill=2)
    assert first[None] == 2
    second = qd.field(qd.i32, shape=(), fill=4)
    third = qd.field(qd.i32, shape=(), fill=6)
    qd.sync()
    assert second[None] == 4
    assert third[None] == 6
    assert second.snode._snode_tree_id == third.snode._snode_tree_id
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


@pytest.mark.parametrize("factory", ["vector", "matrix", "vector_type", "matrix_type"])
@pytest.mark.parametrize("scalar", [False, True])
@test_utils.test()
def test_fill_initializer_components(factory, scalar):
    value = [1, 2] if "vector" in factory else [[1, 2], [3, 4]]
    expected = np.array(value)
    fill = 9 if scalar else value
    if factory == "vector":
        field = qd.Vector.field(2, qd.i32, shape=3, fill=fill)
    elif factory == "matrix":
        field = qd.Matrix.field(2, 2, qd.i32, shape=3, fill=fill)
    elif factory == "vector_type":
        field = qd.field(qd.types.vector(2, qd.i32), shape=3, fill=fill)
    else:
        field = qd.field(qd.types.matrix(2, 2, qd.i32), shape=3, fill=fill)
    if "vector" in factory:
        value[0] = 99
    else:
        value[0][0] = 99
    for i in range(3):
        np.testing.assert_array_equal(field[i].to_numpy(), np.full_like(expected, 9) if scalar else expected)


@test_utils.test()
def test_fill_initializer_matrix_value_snapshot():
    value = qd.Matrix([[1, 2], [3, 4]])
    field = qd.Matrix.field(2, 2, qd.i32, shape=(), fill=value)
    value[0, 0] = 99
    np.testing.assert_array_equal(field[None].to_numpy(), [[1, 2], [3, 4]])


@test_utils.test()
def test_fill_initializer_manual_placement():
    field = qd.field(qd.i32, fill=8)
    qd.root.dense(qd.i, 4).place(field)
    np.testing.assert_array_equal(field.to_numpy(), [8] * 4)


@test_utils.test()
def test_fill_initializer_gradients_and_offset():
    field = qd.field(qd.f32, shape=4, offset=-2, needs_grad=True, needs_dual=True, fill=2)
    assert field[-2] == 2
    assert field[1] == 2
    assert field.grad[-2] == 0
    assert field.dual[-2] == 0


@test_utils.test()
def test_fill_initializer_reset():
    qd.field(qd.i32, shape=(), fill=8)
    runtime = impl.get_runtime()
    arch = qd.cfg.arch
    qd.reset()
    assert not runtime.field_initializers.pending
    qd.init(arch=arch)
    field = qd.field(qd.i32, shape=(), fill=4)
    assert field[None] == 4


@pytest.mark.parametrize("value", [[1, 2], "invalid"])
@test_utils.test()
def test_fill_initializer_invalid_scalar(value):
    with pytest.raises(ValueError, match="fill"):
        qd.field(qd.i32, shape=(), fill=value)
    field = qd.field(qd.i32, shape=(), fill=0)
    assert field[None] == 0


@pytest.mark.parametrize("value", [[[1, 2]], [[1, 2], [3]], [1, 2], [[1, None], [2, 3]]])
@test_utils.test()
def test_fill_initializer_invalid_matrix(value):
    with pytest.raises(ValueError, match="fill"):
        qd.Matrix.field(2, 2, qd.i32, shape=(), fill=value)
    field = qd.field(qd.i32, shape=(), fill=0)
    assert field[None] == 0


@test_utils.test()
def test_fill_initializer_preserves_immediate_fill():
    first = qd.field(qd.i32, shape=())
    first.fill(3)
    assert impl.get_runtime().get_num_compiled_functions() > 0
    second = qd.field(qd.i32, shape=(), fill=4)
    assert second[None] == 4
    assert first.snode._snode_tree_id != second.snode._snode_tree_id


@pytest.mark.needs_torch
@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.metal])
def test_fill_initializer_dlpack(matrix):
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
