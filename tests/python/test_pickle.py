import pickle

import numpy as np
import pytest

import quadrants as qd
from quadrants.types.enums import Layout

from tests import test_utils

_ALL_DTYPES = [qd.f16, qd.f32, qd.f64, qd.i8, qd.i16, qd.i32, qd.i64, qd.u1, qd.u8, qd.u16, qd.u32, qd.u64]
_64BIT_DTYPES = {qd.f64, qd.i64, qd.u64}


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


@test_utils.test()
def test_pickle_scalar_ndarray_f32_2d():
    a = qd.ndarray(qd.f32, (2, 3))
    a[1, 2] = 3.140625

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


@test_utils.test()
def test_pickle_square_matrix_ndarray():
    a = qd.Matrix.ndarray(3, 3, qd.f32, shape=(2, 2))
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


@pytest.mark.parametrize("dtype", _ALL_DTYPES, ids=lambda d: d.to_string())
@test_utils.test()
def test_pickle_all_dtypes(dtype):
    if dtype in _64BIT_DTYPES and qd.cfg.arch in (qd.metal, qd.vulkan):
        pytest.skip(f"{qd.cfg.arch} does not support 64-bit types")
    a = qd.ndarray(dtype, (4,))
    if dtype == qd.u1:
        a[0] = 1
        a[2] = 1
    elif dtype in (qd.f16, qd.f32, qd.f64):
        a[0] = 1.5
        a[2] = -3.25
    else:
        a[0] = 7
        a[2] = 42
    b = _roundtrip(a)
    np.testing.assert_array_equal(b.to_numpy(), a.to_numpy())
    assert b.shape == a.shape
    assert b.dtype == a.dtype


@test_utils.test()
def test_pickle_soa_raises():
    a = qd.ndarray(qd.f32, (3,))
    a.layout = Layout.SOA

    with pytest.raises(TypeError, match="SOA layout"):
        pickle.dumps(a)
