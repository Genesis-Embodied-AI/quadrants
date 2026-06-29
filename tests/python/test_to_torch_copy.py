"""Tests for the ``copy=`` parameter on ``to_torch`` and ``to_numpy`` across field and ndarray types.

Covers:
- ``copy=False`` -> DLPack zero-copy (shares memory, no data movement)
- ``copy=True`` (default) -> kernel copy (always correct, independent memory)
- AoS struct members -> default copy correct, ``copy=False`` raises
- SoA struct members -> ``copy=False`` zero-copy works correctly
- ``to_numpy(copy=False)`` -> zero-copy on CPU via ``np.from_dlpack``
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
    if qd.cfg.arch in _NO_ZEROCOPY_ARCHS:
        pytest.skip(f"DLPack zero-copy not supported on {qd.cfg.arch.name}")
    if qd.cfg.arch == qd.metal:
        from quadrants.lang.field import (
            _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET,  # pylint: disable=C0415
        )

        if not _TORCH_MPS_SUPPORTS_DLPACK_BYTES_OFFSET:
            pytest.skip("Metal DLPack zero-copy requires torch >= 2.9.2")


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


_TORCH_DEVICE_NAME = {qd.cuda: "cuda", qd.amdgpu: "cuda", qd.metal: "mps"}


@test_utils.test(exclude=[qd.cpu])
def test_scalar_field_copy_false_device_no_index():
    """copy=False with device='<type>' (no index) should not raise when data is already on that device."""
    _skip_if_no_zerocopy()
    torch_device = _TORCH_DEVICE_NAME.get(qd.cfg.arch)
    if torch_device is None:
        pytest.skip(f"No torch device mapping for {qd.cfg.arch.name}")
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    qd.sync()

    t = f.to_torch(device=torch_device, copy=False)
    np.testing.assert_allclose(t.cpu().numpy(), [1, 2, 3, 4])


@test_utils.test()
def test_scalar_field_copy_none():
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    qd.sync()

    t = f.to_torch()
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

    t = f.to_torch()
    np.testing.assert_allclose(t.cpu().numpy(), [[1, 2, 3], [4, 5, 6]])


@test_utils.test()
def test_vector_field_soa_copy_false_raises():
    """SOA Vector.field has non-contiguous component layout -- copy=False should raise."""
    _skip_if_no_zerocopy()
    f = qd.Vector.field(3, qd.f32, shape=(4,), layout=qd.Layout.SOA)
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32))
    qd.sync()

    with pytest.raises(ValueError, match="Zero-copy not available"):
        f.to_torch(copy=False)

    t = f.to_torch(copy=True)
    np.testing.assert_allclose(t.cpu().numpy(), [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])


@test_utils.test()
def test_vector_field_copy_false_keep_dims():
    """copy=False with keep_dims=True should produce the same shape as copy=True with keep_dims=True."""
    _skip_if_no_zerocopy()
    f = qd.Vector.field(3, qd.f32, shape=(2,))
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    qd.sync()

    t_copy = f.to_torch(copy=True, keep_dims=True)
    t_view = f.to_torch(copy=False, keep_dims=True)
    assert t_copy.shape == t_view.shape, f"shape mismatch: copy=True {t_copy.shape} vs copy=False {t_view.shape}"
    np.testing.assert_allclose(t_view.cpu().numpy(), t_copy.cpu().numpy())


@test_utils.test()
def test_vector_field_to_numpy_copy_false_keep_dims():
    """to_numpy(copy=False) with keep_dims=True should produce the same shape as copy=True with keep_dims=True."""
    _skip_if_no_zerocopy()
    if qd.cfg.arch not in (qd.cpu, qd.arm64):
        pytest.skip("numpy zero-copy requires CPU backend")
    f = qd.Vector.field(3, qd.f32, shape=(2,))
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    qd.sync()

    arr_copy = f.to_numpy(copy=True, keep_dims=True)
    arr_view = f.to_numpy(copy=False, keep_dims=True)
    assert (
        arr_copy.shape == arr_view.shape
    ), f"shape mismatch: copy=True {arr_copy.shape} vs copy=False {arr_view.shape}"
    np.testing.assert_allclose(arr_view, arr_copy)


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

    t = nd.to_torch(copy=True)
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

    t = nd.to_torch(copy=True)
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

    t = nd.to_torch(copy=True)
    np.testing.assert_allclose(t.cpu().numpy(), data)


# ---------------------------------------------------------------------------
# Struct fields: AoS vs SoA
# ---------------------------------------------------------------------------


@test_utils.test()
def test_struct_aos_copy_none_correct():
    """AoS struct member to_torch(copy=True) returns correct values via kernel copy."""
    s = qd.types.struct(a=qd.i32, b=qd.f32)
    f = s.field(shape=(4,), layout=qd.Layout.AOS)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = i * 10
            f[i].b = i * 0.5

    fill()
    qd.sync()

    t_a = f.a.to_torch(copy=True)
    t_b = f.b.to_torch(copy=True)
    np.testing.assert_array_equal(t_a.cpu().numpy(), [0, 10, 20, 30])
    np.testing.assert_allclose(t_b.cpu().numpy(), [0.0, 0.5, 1.0, 1.5])


@test_utils.test()
def test_struct_aos_copy_false_raises():
    """AoS struct member to_torch(copy=False) should raise because DLPack can't represent the interleaved strides."""
    s = qd.types.struct(a=qd.i32, b=qd.f32)
    f = s.field(shape=(4,), layout=qd.Layout.AOS)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = i * 10
            f[i].b = i * 0.5

    fill()
    qd.sync()

    with pytest.raises(ValueError, match="Zero-copy not available"):
        f.a.to_torch(copy=False)


@test_utils.test()
def test_struct_aos_vec_member_copy_false_raises():
    """AoS struct with a vec3 member: copy=False on the vec member should raise."""
    vec3 = qd.types.vector(3, qd.f32)
    s = qd.types.struct(a=qd.f32, b=vec3)
    f = s.field(shape=(4,), layout=qd.Layout.AOS)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = qd.f32(i)
            f[i].b = qd.Vector([qd.f32(i * 10), qd.f32(i * 20), qd.f32(i * 30)])

    fill()
    qd.sync()

    with pytest.raises(ValueError, match="Zero-copy not available"):
        f.b.to_torch(copy=False)

    t_b = f.b.to_torch(copy=True)
    np.testing.assert_allclose(t_b.cpu().numpy(), [[0, 0, 0], [10, 20, 30], [20, 40, 60], [30, 60, 90]])


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
    """Struct fields with no explicit layout= default to AoS. copy=True should still be correct."""
    s = qd.types.struct(x=qd.f32, y=qd.f32)
    f = s.field(shape=(4,))

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].x = i * 1.0
            f[i].y = i * 2.0

    fill()
    qd.sync()

    t_x = f.x.to_torch(copy=True)
    t_y = f.y.to_torch(copy=True)
    np.testing.assert_allclose(t_x.cpu().numpy(), [0, 1, 2, 3])
    np.testing.assert_allclose(t_y.cpu().numpy(), [0, 2, 4, 6])


# ---------------------------------------------------------------------------
# Backends without DLPack: copy=False should raise
# ---------------------------------------------------------------------------


@test_utils.test()
def test_no_zerocopy_copy_false_raises():
    """Backends that don't support DLPack (e.g. Vulkan) should raise on copy=False."""
    if qd.cfg.arch not in _NO_ZEROCOPY_ARCHS:
        pytest.skip(f"DLPack zero-copy is available on {qd.cfg.arch.name}")

    f = qd.field(qd.f32, shape=(4,))
    f.fill(1.0)
    qd.sync()

    with pytest.raises(ValueError, match="Zero-copy not available"):
        f.to_torch(copy=False)


# ---------------------------------------------------------------------------
# Apple Metal: MPS synchronisation
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.metal])
def test_metal_copy_true_syncs_mps():
    """copy=True on Metal should call torch.mps.synchronize() so the tensor is immediately usable."""
    f = qd.field(qd.f32, shape=(64,))

    @qd.kernel
    def fill():
        for i in range(64):
            f[i] = qd.f32(i)

    fill()
    t = f.to_torch(copy=True)
    np.testing.assert_allclose(t.cpu().numpy(), np.arange(64, dtype=np.float32))


@test_utils.test(arch=[qd.metal])
def test_metal_copy_false_no_mps_sync():
    """copy=False on Metal should NOT call torch.mps.synchronize() (only qd.sync())."""
    f = qd.field(qd.f32, shape=(64,))

    @qd.kernel
    def fill():
        for i in range(64):
            f[i] = qd.f32(i)

    fill()
    t = f.to_torch(copy=False)
    torch.mps.synchronize()
    np.testing.assert_allclose(t.cpu().numpy(), np.arange(64, dtype=np.float32))


# ---------------------------------------------------------------------------
# to_numpy(copy=...) -- CPU only (numpy arrays cannot reference GPU memory)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_copy_false():
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    qd.sync()

    arr = f.to_numpy(copy=False)
    np.testing.assert_allclose(arr, [1, 2, 3, 4])


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_copy_false_shares_memory():
    """to_numpy(copy=False) on CPU should produce a zero-copy view that reflects mutations."""
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    arr = f.to_numpy(copy=False)
    np.testing.assert_allclose(arr, [10, 20, 30, 40])

    f.from_numpy(np.array([99, 88, 77, 66], dtype=np.float32))
    qd.sync()
    np.testing.assert_allclose(arr, [99, 88, 77, 66])


def _np_supports_dlpack_v1():
    """NumPy >= 2.1 can consume DLPack v1 capsules, which yield writable arrays."""
    return tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 1)


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_copy_false_is_writable():
    """to_numpy(copy=False) should return a writable array on NumPy >= 2.1 (DLPack v1 capsule)."""
    if not _np_supports_dlpack_v1():
        pytest.skip("NumPy < 2.1 returns read-only arrays from DLPack v0 capsules")
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    arr = f.to_numpy(copy=False)
    assert arr.flags.writeable, "to_numpy(copy=False) should return a writable array"
    arr[0] = 99.0
    np.testing.assert_allclose(arr[0], 99.0)


@test_utils.test(arch=[qd.cpu])
def test_scalar_ndarray_to_numpy_copy_false_is_writable():
    """Ndarray to_numpy(copy=False) should return a writable array on NumPy >= 2.1 (DLPack v1 capsule)."""
    if not _np_supports_dlpack_v1():
        pytest.skip("NumPy < 2.1 returns read-only arrays from DLPack v0 capsules")
    nd = qd.ndarray(qd.f32, shape=(4,))
    nd.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    arr = nd.to_numpy(copy=False)
    assert arr.flags.writeable, "to_numpy(copy=False) should return a writable array"
    arr[0] = 99.0
    np.testing.assert_allclose(arr[0], 99.0)


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_copy_false_write_visible():
    """Writing to a to_numpy(copy=False) view should be visible to subsequent field reads (NumPy >= 2.1)."""
    if not _np_supports_dlpack_v1():
        pytest.skip("NumPy < 2.1 returns read-only arrays from DLPack v0 capsules")
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    view = f.to_numpy(copy=False)
    view[0] = 100.0
    qd.sync()
    fresh = f.to_numpy()
    np.testing.assert_allclose(fresh[0], 100.0)


# ---------------------------------------------------------------------------
# Layout-tagged fields: to_numpy(copy=False) with order=
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_layout_tagged_field_to_numpy_copy_false():
    """to_numpy(copy=False) on a layout-tagged (order='ji') field should return a correct canonical view."""
    f = qd.field(qd.f32, shape=(3, 4), order="ji")
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    f.from_numpy(data)
    qd.sync()

    arr = f.to_numpy(copy=False)
    np.testing.assert_allclose(arr, data)
    assert arr.shape == (3, 4)


@test_utils.test(arch=[qd.cpu])
def test_layout_tagged_field_to_numpy_copy_false_write_visible():
    """Writes to a layout-tagged to_numpy(copy=False) view should be visible to subsequent field reads."""
    if not _np_supports_dlpack_v1():
        pytest.skip("NumPy < 2.1 returns read-only arrays from DLPack v0 capsules")
    f = qd.field(qd.f32, shape=(3, 4), order="ji")
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    f.from_numpy(data)
    qd.sync()

    view = f.to_numpy(copy=False)
    view[0, 0] = 999.0
    qd.sync()
    fresh = f.to_numpy()
    np.testing.assert_allclose(fresh[0, 0], 999.0)


# ---------------------------------------------------------------------------
# MatrixField / Tensor wrapper: to_dlpack(versioned=True)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_matrix_field_to_dlpack_versioned():
    """MatrixField.to_dlpack(versioned=True) should produce a writable numpy array on NumPy >= 2.1."""
    if not _np_supports_dlpack_v1():
        pytest.skip("NumPy < 2.1 cannot consume v1 capsules")
    from quadrants.lang.field import _DLPackV1Adapter

    f = qd.Matrix.field(2, 3, qd.f32, shape=(2,))
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    f.from_numpy(data)
    qd.sync()

    capsule = f.to_dlpack(versioned=True)
    arr = np.from_dlpack(_DLPackV1Adapter(capsule))
    assert arr.flags.writeable
    np.testing.assert_allclose(arr, data)


@test_utils.test(arch=[qd.cpu])
def test_tensor_to_dlpack_versioned():
    """Tensor.to_dlpack(versioned=True) should forward to the impl and produce a valid v1 capsule."""
    if not _np_supports_dlpack_v1():
        pytest.skip("NumPy < 2.1 cannot consume v1 capsules")
    from quadrants.lang.field import _DLPackV1Adapter

    t = qd.tensor(qd.f32, shape=(8,))
    t.fill(42.0)
    qd.sync()

    capsule = t.to_dlpack(versioned=True)
    arr = np.from_dlpack(_DLPackV1Adapter(capsule))
    assert arr.flags.writeable
    np.testing.assert_allclose(arr, 42.0)


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_default_is_copy():
    """Default to_numpy() returns an independent copy, not a view."""
    f = qd.field(qd.f32, shape=(4,))
    f.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    arr = f.to_numpy()
    f.from_numpy(np.array([99, 88, 77, 66], dtype=np.float32))
    qd.sync()
    np.testing.assert_allclose(arr, [10, 20, 30, 40])


@test_utils.test(arch=[qd.cpu])
def test_vector_field_to_numpy_copy_false():
    f = qd.Vector.field(3, qd.f32, shape=(2,))
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    qd.sync()

    arr = f.to_numpy(copy=False)
    np.testing.assert_allclose(arr, [[1, 2, 3], [4, 5, 6]])


@test_utils.test(arch=[qd.cpu])
def test_scalar_ndarray_to_numpy_copy_false():
    nd = qd.ndarray(qd.f32, shape=(4,))
    nd.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    arr = nd.to_numpy(copy=False)
    np.testing.assert_allclose(arr, [10, 20, 30, 40])


@test_utils.test(arch=[qd.cpu])
def test_vector_ndarray_to_numpy_copy_false():
    vec_type = qd.types.vector(3, qd.f32)
    nd = qd.ndarray(vec_type, shape=(2,))
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    nd.from_numpy(data)
    qd.sync()

    arr = nd.to_numpy(copy=False)
    np.testing.assert_allclose(arr, data)


@test_utils.test(exclude=[qd.cpu])
def test_to_numpy_copy_false_raises_on_gpu():
    """to_numpy(copy=False) should raise on GPU backends since numpy can't reference GPU memory."""
    _skip_if_no_zerocopy()
    f = qd.field(qd.f32, shape=(4,))
    f.fill(1.0)
    qd.sync()

    with pytest.raises(ValueError):
        f.to_numpy(copy=False)


# ---------------------------------------------------------------------------
# 0-dim ScalarField: copy=False should raise
# ---------------------------------------------------------------------------


@test_utils.test()
def test_0dim_scalar_field_copy_false_raises():
    """0-dim ScalarField is not zero-copyable (PyTorch DLPack bytes_offset limitation)."""
    _skip_if_no_zerocopy()
    f = qd.field(qd.f32, shape=())
    f.fill(42.0)
    qd.sync()

    with pytest.raises(ValueError, match="Zero-copy not available"):
        f.to_torch(copy=False)

    t = f.to_torch(copy=True)
    np.testing.assert_allclose(t.cpu().numpy(), 42.0)


@test_utils.test(arch=[qd.cpu])
def test_0dim_scalar_field_to_numpy_copy_false_raises():
    """0-dim ScalarField to_numpy(copy=False) should also raise."""
    f = qd.field(qd.f32, shape=())
    f.fill(42.0)
    qd.sync()

    with pytest.raises(ValueError):
        f.to_numpy(copy=False)

    arr = f.to_numpy(copy=True)
    np.testing.assert_allclose(arr, 42.0)


# ---------------------------------------------------------------------------
# Unsupported dtype: copy=False should raise
# ---------------------------------------------------------------------------


@test_utils.test()
def test_unsupported_dtype_copy_false_raises():
    """Dtypes not in _DLPACK_SUPPORTED_DTYPES (e.g. f16) should raise on copy=False."""
    _skip_if_no_zerocopy()
    f = qd.field(qd.f16, shape=(4,))
    f.fill(1.0)
    qd.sync()

    with pytest.raises(ValueError, match="Zero-copy not available"):
        f.to_torch(copy=False)

    t = f.to_torch(copy=True)
    assert t.shape == (4,)


@test_utils.test(arch=[qd.cpu])
def test_unsupported_dtype_to_numpy_copy_false_raises():
    """Unsupported dtype to_numpy(copy=False) should raise."""
    f = qd.field(qd.f16, shape=(4,))
    f.fill(1.0)
    qd.sync()

    with pytest.raises(ValueError):
        f.to_numpy(copy=False)


# ---------------------------------------------------------------------------
# copy=False with cross-device transfer should raise
# ---------------------------------------------------------------------------


@test_utils.test(exclude=[qd.cpu])
def test_copy_false_cross_device_raises():
    """copy=False with device='cpu' when data is on GPU should raise (can't move without copying)."""
    _skip_if_no_zerocopy()
    f = qd.field(qd.f32, shape=(4,))
    f.fill(1.0)
    qd.sync()

    with pytest.raises(ValueError, match="incompatible with device transfer"):
        f.to_torch(device="cpu", copy=False)


# ---------------------------------------------------------------------------
# MatrixField (n>1, m>1) copy=False
# ---------------------------------------------------------------------------


@test_utils.test()
def test_matrix_field_copy_false():
    """Actual matrix field (2x3, not vector) with copy=False."""
    _skip_if_no_zerocopy()
    f = qd.Matrix.field(2, 3, qd.f32, shape=(2,))
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    f.from_numpy(data)
    qd.sync()

    t = f.to_torch(copy=False)
    np.testing.assert_allclose(t.cpu().numpy(), data)


@test_utils.test()
def test_matrix_field_copy_true():
    f = qd.Matrix.field(2, 3, qd.f32, shape=(2,))
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    f.from_numpy(data)
    qd.sync()

    t = f.to_torch(copy=True)
    np.testing.assert_allclose(t.cpu().numpy(), data)


@test_utils.test(arch=[qd.cpu])
def test_matrix_field_to_numpy_copy_false():
    """Actual matrix field (2x3) to_numpy(copy=False) on CPU."""
    f = qd.Matrix.field(2, 3, qd.f32, shape=(2,))
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    f.from_numpy(data)
    qd.sync()

    arr = f.to_numpy(copy=False)
    np.testing.assert_allclose(arr, data)


# ---------------------------------------------------------------------------
# MatrixNdarray to_numpy(copy=False)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_matrix_ndarray_to_numpy_copy_false():
    mat_type = qd.types.matrix(2, 3, qd.f32)
    nd = qd.ndarray(mat_type, shape=(2,))
    data = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float32)
    nd.from_numpy(data)
    qd.sync()

    arr = nd.to_numpy(copy=False)
    np.testing.assert_allclose(arr, data)


# ---------------------------------------------------------------------------
# AoS struct member to_numpy(copy=False) should raise
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_struct_aos_to_numpy_copy_false_raises():
    """AoS struct member to_numpy(copy=False) should raise, same as to_torch(copy=False)."""
    s = qd.types.struct(a=qd.i32, b=qd.f32)
    f = s.field(shape=(4,), layout=qd.Layout.AOS)

    @qd.kernel
    def fill():
        for i in range(4):
            f[i].a = i * 10
            f[i].b = i * 0.5

    fill()
    qd.sync()

    with pytest.raises(ValueError):
        f.a.to_numpy(copy=False)

    arr_a = f.a.to_numpy(copy=True)
    arr_b = f.b.to_numpy(copy=True)
    np.testing.assert_array_equal(arr_a, [0, 10, 20, 30])
    np.testing.assert_allclose(arr_b, [0.0, 0.5, 1.0, 1.5])


# ---------------------------------------------------------------------------
# copy=True independence for vector/matrix/ndarray types
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_vector_field_copy_true_independent():
    """Default copy=True on VectorField should return a buffer independent of the field."""
    f = qd.Vector.field(3, qd.f32, shape=(2,))
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    qd.sync()

    t = f.to_torch()
    f.from_numpy(np.array([[99, 88, 77], [66, 55, 44]], dtype=np.float32))
    qd.sync()
    np.testing.assert_allclose(t.numpy(), [[1, 2, 3], [4, 5, 6]])


@test_utils.test(arch=[qd.cpu])
def test_matrix_field_copy_true_independent():
    """Default copy=True on MatrixField should return a buffer independent of the field."""
    f = qd.Matrix.field(2, 2, qd.f32, shape=(2,))
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    f.from_numpy(data)
    qd.sync()

    t = f.to_torch()
    f.from_numpy(np.zeros_like(data))
    qd.sync()
    np.testing.assert_allclose(t.numpy(), data)


@test_utils.test(arch=[qd.cpu])
def test_scalar_ndarray_copy_true_independent():
    """Default copy=True on ScalarNdarray should return a buffer independent of the ndarray."""
    nd = qd.ndarray(qd.f32, shape=(4,))
    nd.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    qd.sync()

    t = nd.to_torch()
    nd.from_numpy(np.zeros(4, dtype=np.float32))
    qd.sync()
    np.testing.assert_allclose(t.numpy(), [10, 20, 30, 40])


@test_utils.test(arch=[qd.cpu])
def test_vector_ndarray_copy_true_independent():
    """Default copy=True on VectorNdarray should return a buffer independent of the ndarray."""
    vec_type = qd.types.vector(3, qd.f32)
    nd = qd.ndarray(vec_type, shape=(2,))
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    nd.from_numpy(data)
    qd.sync()

    t = nd.to_torch()
    nd.from_numpy(np.zeros_like(data))
    qd.sync()
    np.testing.assert_allclose(t.numpy(), data)


@test_utils.test(arch=[qd.cpu])
def test_vector_field_to_numpy_copy_true_independent():
    """Default to_numpy() on VectorField should return an independent copy."""
    f = qd.Vector.field(3, qd.f32, shape=(2,))
    f.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    qd.sync()

    arr = f.to_numpy()
    f.from_numpy(np.array([[99, 88, 77], [66, 55, 44]], dtype=np.float32))
    qd.sync()
    np.testing.assert_allclose(arr, [[1, 2, 3], [4, 5, 6]])


@test_utils.test(debug=True)
def test_debug_needs_grad_zerocopy_view_indices_match():
    """Writes through a zero-copy torch view of a ``needs_grad`` field land at the same indices in the field.

    Internal details: pins that a standalone ``needs_grad`` field's primal dense container has a single-member cell in
    debug mode, so the DLPack contiguous strides exported by ``field_to_dlpack`` match the actual per-cell byte stride.
    Adding any sibling place child (e.g. an adjoint-checkbit) to the same dense doubles the cell stride while DLPack
    keeps reporting tight strides, and writes via the torch view land at half-cell offsets.
    """
    _skip_if_no_zerocopy()
    N = 16
    x = qd.field(qd.f32, shape=(N,), needs_grad=True)
    view = x.to_torch(copy=False)
    view[7] = 7.0
    view[10] = 10.0
    if qd.cfg.arch == qd.metal:
        torch.mps.synchronize()
    out = x.to_numpy()
    expected = np.zeros(N, dtype=np.float32)
    expected[7] = 7.0
    expected[10] = 10.0
    np.testing.assert_allclose(out, expected)


@test_utils.test(debug=True)
def test_debug_needs_grad_parent_is_single_child():
    """A standalone ``needs_grad`` field accepts ``to_torch(copy=False)`` in debug mode.

    Internal details: pins ``parent.get_num_ch() == 1`` for the primal dense, which is what ``_is_aos_struct_member``
    checks to allow zero-copy export. Anything that adds a sibling place child to the primal's dense (notably the
    adjoint-checkbit when it is placed there instead of in its own root-level dense) trips that check and makes
    ``_can_zerocopy_field`` reject the field with ``ValueError: Zero-copy not available``.
    """
    _skip_if_no_zerocopy()
    x = qd.field(qd.f32, shape=(16, 1), needs_grad=True)

    @qd.kernel
    def touch():
        x[0, 0] = x[0, 0]

    touch()
    parent = x.parent()._snode.ptr
    assert parent.get_num_ch() == 1
    view = x.to_torch(copy=False)
    assert tuple(view.shape) == (16, 1)


@test_utils.test()
def test_grad_fill_on_unallocated_field_raises_clearly():
    """``field.grad.fill(...)`` on a field allocated without ``needs_grad`` raises a clear error, not a kernel crash.

    Internal details: pins the ``_require_placed`` guard on ``ScalarField.fill`` / ``to_numpy`` / ``from_numpy`` /
    ``__setitem__`` / ``__getitem__``. ``create_field_member`` always allocates the adjoint ``FieldExpression`` for
    real-dtype fields, but ``_field()`` only places the SNode when ``needs_grad=True``; reaching the un-placed wrapper
    via ``field.grad`` and writing to it used to crash deep in ``fill_field`` AST compilation with
    ``AttributeError: 'NoneType' object has no attribute 'data_type'``. The guard surfaces a
    ``QuadrantsRuntimeError`` instead so callers can ``try/except`` (or check) cleanly.
    """
    x = qd.tensor(dtype=qd.f32, shape=(16, 1), backend=qd.Backend.FIELD)
    assert x.grad is not None
    with pytest.raises(qd.QuadrantsRuntimeError, match="no allocation"):
        x.grad.fill(0.0)
    y = qd.tensor(dtype=qd.f32, shape=(4,), backend=qd.Backend.FIELD, needs_grad=True)
    y.grad.fill(0.0)
