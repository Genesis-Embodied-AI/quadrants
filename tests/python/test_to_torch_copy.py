"""Tests for the ``copy=`` parameter on ``to_torch`` across field and ndarray types.

Covers:
- ``copy=False`` -> DLPack zero-copy (shares memory, no data movement)
- ``copy=None``  -> kernel copy (default, always correct, independent memory)
- AoS struct members -> ``copy=None`` correct, ``copy=False`` returns garbage
- SoA struct members -> ``copy=False`` zero-copy works correctly
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.needs_torch

_NO_ZEROCOPY_ARCHS = {qd.vulkan}


def _skip_if_no_zerocopy():
    """Skip the current test if the active backend doesn't support DLPack zero-copy."""
    if qd.cfg().arch in _NO_ZEROCOPY_ARCHS:
        pytest.skip(f"DLPack zero-copy not supported on {qd.cfg().arch.name}")


# ---------------------------------------------------------------------------
# ScalarField
# ---------------------------------------------------------------------------


@test_utils.test()
@pytest.mark.parametrize("dtype", [qd.f32, qd.i32])
def test_scalar_field_copy_false(dtype):
    _skip_if_no_zerocopy()
    f = qd.field(dtype, shape=(4,))
    f.from_numpy(np.arange(4, dtype=np.float32 if dtype is qd.f32 else np.int32))
    qd.sync()

    t = f.to_torch(copy=False)
    expected = np.arange(4, dtype=np.float32 if dtype is qd.f32 else np.int32)
    np.testing.assert_allclose(t.cpu().numpy(), expected)


@test_utils.test()
def test_scalar_field_copy_none():
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    qd.sync()

    t = f.to_torch(copy=None)
    np.testing.assert_allclose(t.cpu().numpy(), [1, 2, 3, 4])


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_copy_false_shares_memory():
    """copy=False on CPU should produce a zero-copy view that reflects mutations."""
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    t = f.to_torch(copy=False)
    np.testing.assert_allclose(t.numpy(), [10, 20, 30, 40])

    f.from_numpy(np.array([99, 88, 77, 66], dtype=np.float32))
    qd.sync()
    np.testing.assert_allclose(t.numpy(), [99, 88, 77, 66])


# ---------------------------------------------------------------------------
# MatrixField (vector)
# ---------------------------------------------------------------------------


@test_utils.test()
def test_vector_field_copy_false():
    _skip_if_no_zerocopy()
    f = qd.Vector.field(3, qd.f32, shape=(2,))
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    qd.sync()

    t = f.to_torch(copy=False)
    np.testing.assert_allclose(t.cpu().numpy(), [[1, 2, 3], [4, 5, 6]])


@test_utils.test()
def test_vector_field_copy_none():
    f = qd.Vector.field(3, qd.f32, shape=(2,))
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    qd.sync()

    t = f.to_torch(copy=None)
    np.testing.assert_allclose(t.cpu().numpy(), [[1, 2, 3], [4, 5, 6]])


# ---------------------------------------------------------------------------
# ScalarNdarray
# ---------------------------------------------------------------------------


@test_utils.test()
def test_scalar_ndarray_copy_false():
    _skip_if_no_zerocopy()
    nd = qd.ndarray(qd.f32, shape=(4,))
    nd.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    t = nd.to_torch(copy=False)
    np.testing.assert_allclose(t.cpu().numpy(), [10, 20, 30, 40])


@test_utils.test()
def test_scalar_ndarray_copy_none():
    nd = qd.ndarray(qd.f32, shape=(4,))
    nd.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    t = nd.to_torch(copy=None)
    np.testing.assert_allclose(t.cpu().numpy(), [10, 20, 30, 40])


# ---------------------------------------------------------------------------
# MatrixNdarray
# ---------------------------------------------------------------------------


@test_utils.test()
def test_matrix_ndarray_copy_false():
    _skip_if_no_zerocopy()
    mat_type = qd.types.matrix(2, 3, qd.f32)
    nd = qd.ndarray(mat_type, shape=(2,))
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    nd.from_numpy(data)
    qd.sync()

    t = nd.to_torch(copy=False)
    np.testing.assert_allclose(t.cpu().numpy(), data)


@test_utils.test()
def test_matrix_ndarray_copy_none():
    mat_type = qd.types.matrix(2, 3, qd.f32)
    nd = qd.ndarray(mat_type, shape=(2,))
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    nd.from_numpy(data)
    qd.sync()

    t = nd.to_torch(copy=None)
    np.testing.assert_allclose(t.cpu().numpy(), data)


# ---------------------------------------------------------------------------
# VectorNdarray
# ---------------------------------------------------------------------------


@test_utils.test()
def test_vector_ndarray_copy_false():
    _skip_if_no_zerocopy()
    vec_type = qd.types.vector(3, qd.f32)
    nd = qd.ndarray(vec_type, shape=(2,))
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    nd.from_numpy(data)
    qd.sync()

    t = nd.to_torch(copy=False)
    np.testing.assert_allclose(t.cpu().numpy(), data)


@test_utils.test()
def test_vector_ndarray_copy_none():
    vec_type = qd.types.vector(3, qd.f32)
    nd = qd.ndarray(vec_type, shape=(2,))
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    nd.from_numpy(data)
    qd.sync()

    t = nd.to_torch(copy=None)
    np.testing.assert_allclose(t.cpu().numpy(), data)


# ---------------------------------------------------------------------------
# Struct fields: AoS vs SoA
# ---------------------------------------------------------------------------


@test_utils.test()
def test_struct_aos_copy_none_correct():
    """AoS struct member to_torch(copy=None) returns correct values via kernel copy."""
    s = qd.types.struct(a=qd.i32, b=qd.f32)
    f = s.field(shape=(4,), layout=qd.Layout.AOS)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = i * 10
            f[i].b = i * 0.5

    fill()
    qd.sync()

    t_a = f.a.to_torch(copy=None)
    t_b = f.b.to_torch(copy=None)
    np.testing.assert_array_equal(t_a.cpu().numpy(), [0, 10, 20, 30])
    np.testing.assert_allclose(t_b.cpu().numpy(), [0.0, 0.5, 1.0, 1.5])


@test_utils.test()
def test_struct_aos_copy_false_garbage():
    """AoS struct member to_torch(copy=False) returns garbage because DLPack assumes contiguous
    strides but AoS interleaves members. This documents the known limitation."""
    _skip_if_no_zerocopy()
    s = qd.types.struct(a=qd.i32, b=qd.f32)
    f = s.field(shape=(4,), layout=qd.Layout.AOS)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = i * 10
            f[i].b = i * 0.5

    fill()
    qd.sync()

    t_a = f.a.to_torch(copy=False)
    assert not np.array_equal(t_a.cpu().numpy(), [0, 10, 20, 30])


@test_utils.test()
def test_struct_soa_copy_false_zerocopy():
    """SoA struct member to_torch(copy=False) returns correct values via DLPack zero-copy."""
    _skip_if_no_zerocopy()
    s = qd.types.struct(a=qd.i32, b=qd.f32)
    f = s.field(shape=(4,), layout=qd.Layout.SOA)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = i * 10
            f[i].b = i * 0.5

    fill()
    qd.sync()

    t_a = f.a.to_torch(copy=False)
    t_b = f.b.to_torch(copy=False)
    np.testing.assert_array_equal(t_a.cpu().numpy(), [0, 10, 20, 30])
    np.testing.assert_allclose(t_b.cpu().numpy(), [0.0, 0.5, 1.0, 1.5])


@test_utils.test(arch=[qd.cpu])
def test_struct_soa_zerocopy_shares_memory():
    """SoA struct member to_torch(copy=False) on CPU should share memory with the field."""
    s = qd.types.struct(a=qd.i32, b=qd.f32)
    f = s.field(shape=(4,), layout=qd.Layout.SOA)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = i * 10
            f[i].b = i * 0.5

    fill()
    qd.sync()

    t_a = f.a.to_torch(copy=False)
    np.testing.assert_array_equal(t_a.cpu().numpy(), [0, 10, 20, 30])

    @qd.kernel
    def mutate():
        f[2].a = 999

    mutate()
    qd.sync()

    np.testing.assert_array_equal(t_a.numpy(), [0, 10, 999, 30])


@test_utils.test()
def test_struct_default_layout_copy_none():
    """Struct fields with no explicit layout= default to AoS. copy=None should still be correct."""
    s = qd.types.struct(x=qd.f32, y=qd.f32)
    f = s.field(shape=(4,))

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].x = i * 1.0
            f[i].y = i * 2.0

    fill()
    qd.sync()

    t_x = f.x.to_torch(copy=None)
    t_y = f.y.to_torch(copy=None)
    np.testing.assert_allclose(t_x.cpu().numpy(), [0, 1, 2, 3])
    np.testing.assert_allclose(t_y.cpu().numpy(), [0, 2, 4, 6])


# ---------------------------------------------------------------------------
# Backends without DLPack: copy=False should raise
# ---------------------------------------------------------------------------


@test_utils.test()
def test_no_zerocopy_copy_false_raises():
    """Backends that don't support DLPack (e.g. Vulkan) should raise on copy=False."""
    if qd.cfg().arch not in _NO_ZEROCOPY_ARCHS:
        pytest.skip(f"DLPack zero-copy is available on {qd.cfg().arch.name}")

    f = qd.field(qd.f32, shape=(4,))
    f.fill(1.0)
    qd.sync()

    with pytest.raises(Exception):
        f.to_torch(copy=False)
