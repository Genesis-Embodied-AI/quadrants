import pickle

import numpy as np
import pytest

import quadrants as qd
from quadrants.types.enums import Layout

from tests import test_utils


def _roundtrip(obj):
    return pickle.loads(pickle.dumps(obj))


@test_utils.test()
def test_pickle_scalar_ndarray_i32():
    a = qd.ndarray(qd.i32, (3, 2))
    a[0, 1] = 3
    a[1, 1] = 5

    b = _roundtrip(a)

    assert b[0, 1] == 3
    assert b[1, 1] == 5
    assert b.shape == a.shape
    assert b.dtype == a.dtype


@test_utils.test()
def test_pickle_scalar_ndarray_f32():
    a = qd.ndarray(qd.f32, (4,))
    a[0] = 1.5
    a[3] = -2.25

    b = _roundtrip(a)

    np.testing.assert_allclose(b.to_numpy(), a.to_numpy())
    assert b.shape == a.shape
    assert b.dtype == a.dtype


@test_utils.test(exclude=[qd.metal])
def test_pickle_scalar_ndarray_f64():
    a = qd.ndarray(qd.f64, (2, 3))
    a[1, 2] = 3.141592653589793

    b = _roundtrip(a)

    np.testing.assert_allclose(b.to_numpy(), a.to_numpy())
    assert b.dtype == a.dtype


@test_utils.test()
def test_pickle_vector_ndarray():
    a = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    a[0] = [1.0, 2.0, 3.0]
    a[2] = [4.0, 5.0, 6.0]

    b = _roundtrip(a)

    np.testing.assert_allclose(b.to_numpy(), a.to_numpy())
    assert b.shape == a.shape
    assert b.dtype == a.dtype
    assert b.element_shape == a.element_shape


@test_utils.test()
def test_pickle_matrix_ndarray():
    a = qd.Matrix.ndarray(2, 3, qd.f32, shape=(4,))
    a[0] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a[2] = [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]

    b = _roundtrip(a)

    np.testing.assert_allclose(b.to_numpy(), a.to_numpy())
    assert b.shape == a.shape
    assert b.dtype == a.dtype
    assert b.element_shape == a.element_shape


@test_utils.test(exclude=[qd.metal])
def test_pickle_square_matrix_ndarray():
    a = qd.Matrix.ndarray(3, 3, qd.f64, shape=(2, 2))
    a[0, 0] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    b = _roundtrip(a)

    np.testing.assert_allclose(b.to_numpy(), a.to_numpy())
    assert b.shape == a.shape
    assert b.dtype == a.dtype
    assert b.element_shape == (3, 3)


@test_utils.test()
def test_pickle_scalar_ndarray_preserves_zeros():
    a = qd.ndarray(qd.f32, (5, 5))

    b = _roundtrip(a)

    np.testing.assert_array_equal(b.to_numpy(), np.zeros((5, 5), dtype=np.float32))
    assert b.shape == a.shape


@test_utils.test()
def test_pickle_soa_raises():
    a = qd.ndarray(qd.f32, (3,))
    a.layout = Layout.SOA

    with pytest.raises(TypeError, match="SOA layout"):
        pickle.dumps(a)
