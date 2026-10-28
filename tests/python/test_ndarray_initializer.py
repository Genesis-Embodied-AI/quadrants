import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils

KINDS = ["scalar", "vector", "matrix", "vector_type", "matrix_type"]


def _allocate(kind, shape, fill, **kwargs):
    if kind == "scalar":
        return qd.ndarray(qd.f32, shape, fill=fill, **kwargs)
    if kind == "vector":
        return qd.Vector.ndarray(3, qd.f32, shape, fill=fill, **kwargs)
    if kind == "matrix":
        return qd.Matrix.ndarray(2, 3, qd.f32, shape, fill=fill, **kwargs)
    dtype = qd.types.vector(3, qd.f32) if kind == "vector_type" else qd.types.matrix(2, 3, qd.f32)
    return qd.ndarray(dtype, shape, fill=fill, **kwargs)


def _initial_value(kind):
    if kind == "scalar":
        return -2.5
    if "vector" in kind:
        return [1, 2, 3]
    return [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["none", "scalar", "components"])
@pytest.mark.parametrize("shape", [(), (4,)])
@test_utils.test()
def test_ndarray_initializer_values(kind, mode, shape):
    components = _initial_value(kind)
    fill = None if mode == "none" else 7 if mode == "scalar" else components
    array = _allocate(kind, shape, fill, needs_grad=True)
    element_shape = np.shape(components)
    expected = np.full(element_shape, 0 if mode == "none" else 7) if mode != "components" else components
    np.testing.assert_array_equal(array.to_numpy(), np.broadcast_to(expected, (*shape, *element_shape)))
    np.testing.assert_array_equal(array.grad.to_numpy(), np.zeros((*shape, *element_shape)))


@pytest.mark.parametrize("dtype", [qd.i16, qd.i32, qd.f32])
@test_utils.test()
def test_ndarray_initializer_first_kernel_and_overwrite(dtype):
    array = qd.ndarray(dtype, shape=4, fill=3)

    @qd.kernel
    def read(a: qd.types.ndarray()) -> qd.i32:
        return qd.cast(a[0], qd.i32)

    assert read(array) == 3
    array.fill(5)
    qd.sync()
    assert read(array) == 5


@test_utils.test()
def test_ndarray_initializer_matrix_snapshot():
    value = qd.Matrix([[1, 2, 3], [4, 5, 6]])
    array = qd.Matrix.ndarray(2, 3, qd.i32, shape=(), fill=value)
    value[0, 0] = 99
    np.testing.assert_array_equal(array.to_numpy(), [[1, 2, 3], [4, 5, 6]])


@pytest.mark.parametrize("kind", KINDS)
@test_utils.test()
def test_ndarray_initializer_invalid_values(kind):
    with pytest.raises(ValueError, match="fill"):
        _allocate(kind, (4,), [1, 2])


@pytest.mark.needs_torch
@pytest.mark.parametrize("kind", KINDS)
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.metal])
def test_ndarray_initializer_dlpack(kind):
    torch = pytest.importorskip("torch")
    if qd.cfg.arch == qd.metal:
        from quadrants.lang.field import _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET

        if not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET:
            pytest.skip("Metal DLPack requires torch >= 2.9.2")
    value = _initial_value(kind)
    array = _allocate(kind, (4,), value)
    external = torch.utils.dlpack.from_dlpack(array.to_dlpack())
    np.testing.assert_array_equal(external.cpu().numpy(), np.broadcast_to(value, (4, *np.shape(value))))


@pytest.mark.needs_torch
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("mode", ["none", "scalar", "components"])
def test_ndarray_initializer_python_backend(kind, mode):
    pytest.importorskip("torch")
    qd.init(arch=qd.python)
    try:
        components = _initial_value(kind)
        fill = None if mode == "none" else 7 if mode == "scalar" else components
        array = _allocate(kind, (4,), fill)
        element_shape = np.shape(components)
        expected = np.full(element_shape, 0 if mode == "none" else 7) if mode != "components" else components
        np.testing.assert_array_equal(array.to_numpy(), np.broadcast_to(expected, (4, *element_shape)))
    finally:
        qd.reset()
