import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests.helpers import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("value", [None, 0, -2.5])
@pytest.mark.parametrize("layout", [None, (1, 0)])
@test_utils.test()
def test_tensor_initializer_scalar(backend, value, layout):
    tensor = qd.tensor(qd.f32, shape=(2, 3), backend=backend, layout=layout, fill=value, needs_grad=True)
    assert isinstance(tensor, qd.Tensor)
    np.testing.assert_array_equal(tensor.to_numpy(), np.full((2, 3), 0 if value is None else value))
    np.testing.assert_array_equal(tensor.grad.to_numpy(), np.zeros((2, 3)))

    @qd.kernel
    def update(t: qd.Tensor):
        t[1, 2] = 9

    update(tensor)
    qd.sync()
    expected = np.full((2, 3), 0 if value is None else value)
    expected[1, 2] = 9
    np.testing.assert_array_equal(tensor.to_numpy(), expected)


@test_utils.test()
def test_tensor_initializer_default_backend():
    tensor = qd.tensor(qd.i32, shape=(), fill=7)
    assert isinstance(tensor._unwrap(), qd.Ndarray)
    assert tensor[()] == 7
    tensor.fill(3)
    assert tensor[()] == 3


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["vector", "matrix", "vector_type", "matrix_type"])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("layout", [None, (1, 0)])
@test_utils.test()
def test_tensor_initializer_components(backend, kind, scalar, layout):
    shape = (2, 3)
    value = [2, 3] if "vector" in kind else [[1, 2], [3, 4]]
    expected = np.array(value)
    fill = 7 if scalar else value
    if kind == "vector":
        tensor = qd.Vector.tensor(2, qd.f32, shape=shape, backend=backend, layout=layout, fill=fill, needs_grad=True)
    elif kind == "matrix":
        tensor = qd.Matrix.tensor(2, 2, qd.f32, shape=shape, backend=backend, layout=layout, fill=fill, needs_grad=True)
    elif kind == "vector_type":
        tensor = qd.tensor(
            qd.types.vector(2, qd.f32), shape=shape, backend=backend, layout=layout, fill=fill, needs_grad=True
        )
    else:
        tensor = qd.tensor(
            qd.types.matrix(2, 2, qd.f32), shape=shape, backend=backend, layout=layout, fill=fill, needs_grad=True
        )
    if "vector" in kind:
        value[0] = 99
    else:
        value[0][0] = 99
    expected = np.full_like(expected, 7) if scalar else expected
    np.testing.assert_array_equal(tensor.to_numpy(), np.broadcast_to(expected, (*shape, *expected.shape)))
    np.testing.assert_array_equal(tensor.grad.to_numpy(), np.zeros((*shape, *expected.shape)))


@test_utils.test()
def test_tensor_initializer_fields_share_tree():
    tensors = [qd.tensor(qd.i32, shape=(), backend=qd.Backend.FIELD, fill=i) for i in range(40)]
    fields = [tensor._unwrap() for tensor in tensors]
    assert impl.get_runtime().get_num_compiled_functions() == 0

    @qd.kernel
    def total() -> qd.i32:
        result = 0
        for i in qd.static(range(40)):
            result += fields[i][()]
        return result

    assert total() == 780
    assert len({field.snode._snode_tree_id for field in fields}) == 1
    tensors[0].fill(9)
    assert total() == 789


@pytest.mark.parametrize("backend", BACKENDS)
@test_utils.test()
def test_tensor_initializer_matrix_snapshot(backend):
    value = qd.Matrix([[1, 2], [3, 4]])
    tensor = qd.Matrix.tensor(2, 2, qd.i32, shape=(), backend=backend, fill=value)
    value[0, 0] = 99
    np.testing.assert_array_equal(tensor.to_numpy(), [[1, 2], [3, 4]])


@pytest.mark.parametrize("backend", BACKENDS)
@test_utils.test()
def test_tensor_initializer_invalid_values(backend):
    with pytest.raises(ValueError, match="fill"):
        qd.tensor(qd.i32, shape=4, backend=backend, fill=[1, 2])
    with pytest.raises(ValueError, match="fill"):
        qd.Vector.tensor(2, qd.i32, shape=4, backend=backend, fill=[1, 2, 3])
    with pytest.raises(ValueError, match="fill"):
        qd.Matrix.tensor(2, 2, qd.i32, shape=4, backend=backend, fill=[[1, 2], [3]])


@pytest.mark.needs_torch
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("matrix", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.metal])
def test_tensor_initializer_dlpack(backend, matrix):
    torch = pytest.importorskip("torch")
    if qd.cfg.arch == qd.metal:
        from quadrants.lang.field import _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET

        if not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET:
            pytest.skip("Metal DLPack requires torch >= 2.9.2")
    if matrix:
        tensor = qd.Matrix.tensor(2, 2, qd.f32, shape=4, backend=backend, fill=[[1, 2], [3, 4]])
        expected = [[[1, 2], [3, 4]]] * 4
    else:
        tensor = qd.tensor(qd.f32, shape=(2, 3), backend=backend, layout=(1, 0), fill=7)
        expected = [[7] * 3] * 2
    qd.sync()
    external = torch.utils.dlpack.from_dlpack(tensor.to_dlpack())
    np.testing.assert_array_equal(external.cpu().numpy(), expected)
