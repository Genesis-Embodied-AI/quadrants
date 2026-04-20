"""Tests for ``qd.tensor`` scalar dispatch.

Scope: scalar-element tensor allocation via ``qd.tensor()`` dispatching to
``qd.field`` or ``qd.ndarray`` based on the ``backend=`` kwarg. No layout,
no vec/mat.
"""

import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_tensor_default_backend_is_field():
    a = qd.tensor(qd.f32, shape=(4, 5))
    assert isinstance(a, qd.ScalarField)
    assert a.shape == (4, 5)


@test_utils.test(arch=qd.cpu)
def test_tensor_field_backend_explicit():
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.FIELD)
    assert isinstance(a, qd.ScalarField)
    assert a.shape == (4, 5)


@test_utils.test(arch=qd.cpu)
def test_tensor_ndarray_backend():
    a = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY)
    assert isinstance(a, qd.Ndarray)
    assert a.shape == (4, 5)


@test_utils.test(arch=qd.cpu)
def test_tensor_int_backend_value_accepted():
    """``backend=0`` and ``backend=1`` work too — IntEnum coercion."""
    field_t = qd.tensor(qd.f32, shape=(3,), backend=0)
    ndarray_t = qd.tensor(qd.f32, shape=(3,), backend=1)
    assert isinstance(field_t, qd.ScalarField)
    assert isinstance(ndarray_t, qd.Ndarray)


@test_utils.test(arch=qd.cpu)
def test_tensor_dtype_propagation():
    a = qd.tensor(qd.i32, shape=(4,))
    b = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)
    assert a.dtype == qd.i32
    assert b.dtype == qd.i32


@test_utils.test(arch=qd.cpu)
def test_tensor_int_shape_normalised():
    """Passing an int as shape works the same as a 1-tuple."""
    a = qd.tensor(qd.f32, shape=8)
    b = qd.tensor(qd.f32, shape=8, backend=qd.Backend.NDARRAY)
    assert a.shape == (8,)
    assert b.shape == (8,)


@test_utils.test(arch=qd.cpu)
def test_tensor_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend="):
        qd.tensor(qd.f32, shape=(3,), backend=42)
    with pytest.raises(ValueError, match="backend="):
        qd.tensor(qd.f32, shape=(3,), backend="field")


@test_utils.test(arch=qd.cpu)
def test_tensor_kwargs_pass_through_to_field():
    """Field-only kwargs like ``order=`` reach the underlying ``qd.field``."""
    a = qd.tensor(qd.f32, shape=(4, 5), order="ji")
    assert isinstance(a, qd.ScalarField)
    assert a.shape == (4, 5)


@test_utils.test(arch=qd.cpu)
def test_tensor_field_then_kernel_roundtrip():
    a = qd.tensor(qd.i32, shape=(4,))

    @qd.kernel
    def fill(x: qd.template()):
        for i in range(4):
            x[i] = i * 2

    fill(a)
    assert list(a.to_numpy()) == [0, 2, 4, 6]


@test_utils.test(arch=qd.cpu)
def test_tensor_ndarray_then_kernel_roundtrip():
    a = qd.tensor(qd.i32, shape=(4,), backend=qd.Backend.NDARRAY)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i in range(4):
            x[i] = i * 3

    fill(a)
    assert list(a.to_numpy()) == [0, 3, 6, 9]
