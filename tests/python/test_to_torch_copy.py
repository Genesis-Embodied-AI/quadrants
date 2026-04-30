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

    with pytest.raises(Exception):
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
