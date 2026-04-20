"""Tests for ``qd.tensor_vec`` / ``qd.tensor_mat`` (PR 3)."""

import pytest

import quadrants as qd

from tests import test_utils


# ----------------------------------------------------------------------------
# qd.tensor_vec
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_default_backend_matches_vector_field():
    a = qd.tensor_vec(3, qd.f32, shape=(4,))
    b = qd.Vector.field(3, qd.f32, shape=(4,))
    assert type(a) is type(b)
    assert a.shape == b.shape == (4,)


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_field_explicit():
    a = qd.tensor_vec(3, qd.f32, shape=(4,), backend=qd.Backend.FIELD)
    ref = qd.Vector.field(3, qd.f32, shape=(4,))
    assert type(a) is type(ref)


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_ndarray_matches_vector_ndarray():
    a = qd.tensor_vec(3, qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)
    ref = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    assert type(a) is type(ref)
    assert a.shape == ref.shape == (4,)


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend="):
        qd.tensor_vec(3, qd.f32, shape=(4,), backend="oops")


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_kernel_roundtrip_field():
    v = qd.tensor_vec(3, qd.f32, shape=(4,))

    @qd.kernel
    def fill(x: qd.template()):
        for i in range(4):
            for j in qd.static(range(3)):
                x[i][j] = i * 10.0 + j

    fill(v)
    arr = v.to_numpy()
    assert arr.shape == (4, 3)
    assert arr[2, 1] == 21.0


@test_utils.test(arch=qd.cpu)
def test_tensor_vec_kernel_roundtrip_ndarray():
    v = qd.tensor_vec(3, qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i in range(4):
            for j in qd.static(range(3)):
                x[i][j] = i * 10.0 + j

    fill(v)
    arr = v.to_numpy()
    assert arr.shape == (4, 3)
    assert arr[2, 1] == 21.0


# ----------------------------------------------------------------------------
# qd.tensor_mat
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_default_backend_matches_matrix_field():
    a = qd.tensor_mat(2, 3, qd.f32, shape=(4,))
    b = qd.Matrix.field(2, 3, qd.f32, shape=(4,))
    assert type(a) is type(b)
    assert a.shape == b.shape == (4,)


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_ndarray_matches_matrix_ndarray():
    a = qd.tensor_mat(2, 3, qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)
    ref = qd.Matrix.ndarray(2, 3, qd.f32, shape=(4,))
    assert type(a) is type(ref)
    assert a.shape == ref.shape == (4,)


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend="):
        qd.tensor_mat(2, 3, qd.f32, shape=(4,), backend=99)


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_kernel_roundtrip_field():
    m = qd.tensor_mat(2, 2, qd.f32, shape=(3,))

    @qd.kernel
    def fill(x: qd.template()):
        for i in range(3):
            for r in qd.static(range(2)):
                for c in qd.static(range(2)):
                    x[i][r, c] = i * 100.0 + r * 10.0 + c

    fill(m)
    arr = m.to_numpy()
    assert arr.shape == (3, 2, 2)
    assert arr[1, 0, 1] == 101.0
    assert arr[2, 1, 0] == 210.0


@test_utils.test(arch=qd.cpu)
def test_tensor_mat_kernel_roundtrip_ndarray():
    m = qd.tensor_mat(2, 2, qd.f32, shape=(3,), backend=qd.Backend.NDARRAY)

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i in range(3):
            for r in qd.static(range(2)):
                for c in qd.static(range(2)):
                    x[i][r, c] = i * 100.0 + r * 10.0 + c

    fill(m)
    arr = m.to_numpy()
    assert arr.shape == (3, 2, 2)
    assert arr[1, 0, 1] == 101.0
    assert arr[2, 1, 0] == 210.0
