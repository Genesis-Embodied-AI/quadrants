"""Tests for zero-copy to_torch / to_numpy via DLPack (the ``copy`` parameter)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import quadrants as qd

from tests import test_utils

pytestmark = pytest.mark.needs_torch

dlpack_arch = [qd.cpu, qd.cuda, qd.metal, qd.amdgpu]


def is_v520_amdgpu():
    return os.environ.get("QD_AMDGPU_V520", None) == "1" and qd.cfg.arch == qd.amdgpu


def _to_cpu(t):
    """Move tensor to CPU for value comparison on arches where torch accessors may not work."""
    if is_v520_amdgpu():
        return t.cpu()
    return t


# ---------------------------------------------------------------------------
# ScalarField.to_torch  --  zero-copy
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32])
def test_scalar_field_to_torch_zerocopy(dtype):
    f = qd.field(dtype, shape=(4,))
    f[0] = 10
    f[1] = 20
    qd.sync()

    tc = f.to_torch()
    tc = _to_cpu(tc)
    assert tc[0] == 10
    assert tc[1] == 20
    assert tuple(tc.shape) == (4,)


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_caching():
    """Calling to_torch() twice returns the same cached object."""
    f = qd.field(qd.f32, shape=(3,))
    qd.sync()
    t1 = f.to_torch()
    t2 = f.to_torch()
    assert t1.data_ptr() == t2.data_ptr()


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_copy_true():
    """copy=True must return an independent tensor."""
    f = qd.field(qd.f32, shape=(3,))
    f[0] = 42
    qd.sync()
    tc_view = f.to_torch()
    tc_copy = f.to_torch(copy=True)
    assert _to_cpu(tc_copy)[0] == 42
    assert tc_view.data_ptr() != tc_copy.data_ptr()


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_copy_false():
    """copy=False should succeed on DLPack-capable arches with non-empty shape."""
    f = qd.field(qd.f32, shape=(3,))
    f[0] = 7
    qd.sync()
    tc = f.to_torch(copy=False)
    assert _to_cpu(tc)[0] == 7


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_torch_aliases_memory():
    """Zero-copy tensor must alias the field's device memory."""
    if is_v520_amdgpu():
        pytest.skip("can't run torch accessor kernels on v520")
    f = qd.field(qd.i32, shape=(4,))
    f[0] = 1
    qd.sync()
    tc = f.to_torch()

    @qd.kernel
    def write(f: qd.template()):
        f[0] = 99

    write(f)
    qd.sync()
    assert tc[0] == 99


# ---------------------------------------------------------------------------
# ScalarField.to_numpy  --  zero-copy (CPU only)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_zerocopy_cpu():
    f = qd.field(qd.f32, shape=(5,))
    f[0] = 3.5
    qd.sync()
    arr = f.to_numpy()
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr[0], 3.5)


@test_utils.test(arch=[qd.cpu])
def test_scalar_field_to_numpy_copy_true_cpu():
    f = qd.field(qd.f32, shape=(3,))
    f[0] = 1.0
    qd.sync()
    arr = f.to_numpy(copy=True)
    np.testing.assert_allclose(arr[0], 1.0)


@test_utils.test(arch=dlpack_arch)
def test_scalar_field_to_numpy_matches_copy_path():
    """Zero-copy and copy path produce the same values."""
    f = qd.field(qd.f32, shape=(6,))
    for i in range(6):
        f[i] = float(i * 11)
    qd.sync()
    arr_default = f.to_numpy()
    arr_copy = f.to_numpy(copy=True)
    np.testing.assert_allclose(arr_default, arr_copy)


# ---------------------------------------------------------------------------
# MatrixField.to_torch  --  zero-copy with keep_dims handling
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_vector_field_to_torch_zerocopy():
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(4,))
    f[0] = (1, 2, 3)
    f[1] = (4, 5, 6)
    qd.sync()
    tc = _to_cpu(f.to_torch())
    assert tuple(tc.shape) == (4, 3)
    assert tc[0, 0] == 1
    assert tc[1, 2] == 6


@test_utils.test(arch=dlpack_arch)
def test_vector_field_to_torch_keep_dims():
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(4,))
    f[0] = (10, 20, 30)
    qd.sync()
    tc_no_keep = _to_cpu(f.to_torch(keep_dims=False))
    tc_keep = _to_cpu(f.to_torch(keep_dims=True))
    assert tuple(tc_no_keep.shape) == (4, 3)
    assert tuple(tc_keep.shape) == (4, 3, 1)
    assert tc_no_keep[0, 0] == tc_keep[0, 0, 0]


@test_utils.test(arch=dlpack_arch)
def test_matrix_field_to_torch_zerocopy():
    mat = qd.types.matrix(2, 3, qd.f32)
    f = qd.field(mat, shape=(5,))
    f[0] = ((1, 2, 3), (4, 5, 6))
    qd.sync()
    tc = _to_cpu(f.to_torch())
    assert tuple(tc.shape) == (5, 2, 3)
    assert tc[0, 0, 0] == 1
    assert tc[0, 1, 2] == 6


@test_utils.test(arch=dlpack_arch)
def test_matrix_field_to_torch_matches_copy():
    mat = qd.types.matrix(2, 3, qd.f32)
    f = qd.field(mat, shape=(4,))
    f[0] = ((1, 2, 3), (4, 5, 6))
    f[1] = ((7, 8, 9), (10, 11, 12))
    qd.sync()
    tc_zc = _to_cpu(f.to_torch())
    tc_cp = _to_cpu(f.to_torch(copy=True))
    assert torch.allclose(tc_zc, tc_cp)


# ---------------------------------------------------------------------------
# MatrixField.to_numpy  --  zero-copy (CPU only)
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_vector_field_to_numpy_zerocopy_cpu():
    vec3 = qd.types.vector(3, qd.f32)
    f = qd.field(vec3, shape=(3,))
    f[0] = (10, 20, 30)
    qd.sync()
    arr = f.to_numpy()
    assert arr.shape == (3, 3)
    np.testing.assert_allclose(arr[0], [10, 20, 30])


@test_utils.test(arch=[qd.cpu])
def test_matrix_field_to_numpy_zerocopy_cpu():
    mat = qd.types.matrix(2, 2, qd.f32)
    f = qd.field(mat, shape=(2,))
    f[0] = ((1, 2), (3, 4))
    qd.sync()
    arr = f.to_numpy()
    assert arr.shape == (2, 2, 2)
    np.testing.assert_allclose(arr[0], [[1, 2], [3, 4]])


# ---------------------------------------------------------------------------
# Ndarray.to_torch  --  new method, always uses DLPack
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_scalar_ndarray_to_torch():
    nd = qd.ndarray(qd.f32, shape=(5,))
    nd[0] = 42.0
    qd.sync()
    tc = _to_cpu(nd.to_torch())
    assert tuple(tc.shape) == (5,)
    assert tc[0] == 42.0


@test_utils.test(arch=dlpack_arch)
def test_vector_ndarray_to_torch():
    nd = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    nd[0] = (1, 2, 3)
    qd.sync()
    tc = _to_cpu(nd.to_torch())
    assert tuple(tc.shape) == (4, 3)
    assert tc[0, 0] == 1
    assert tc[0, 2] == 3


@test_utils.test(arch=dlpack_arch)
def test_matrix_ndarray_to_torch():
    nd = qd.Matrix.ndarray(2, 3, qd.f32, shape=(4,))
    nd[0] = ((10, 20, 30), (40, 50, 60))
    qd.sync()
    tc = _to_cpu(nd.to_torch())
    assert tuple(tc.shape) == (4, 2, 3)
    assert tc[0, 0, 0] == 10
    assert tc[0, 1, 2] == 60


@test_utils.test(arch=dlpack_arch)
def test_ndarray_to_torch_caching():
    nd = qd.ndarray(qd.f32, shape=(3,))
    qd.sync()
    t1 = nd.to_torch()
    t2 = nd.to_torch()
    assert t1.data_ptr() == t2.data_ptr()


@test_utils.test(arch=dlpack_arch)
def test_ndarray_to_torch_copy_true():
    nd = qd.ndarray(qd.f32, shape=(3,))
    nd[0] = 7.0
    qd.sync()
    tc = _to_cpu(nd.to_torch(copy=True))
    assert tc[0] == 7.0
    view = nd.to_torch()
    assert view.data_ptr() != tc.data_ptr()


# ---------------------------------------------------------------------------
# Ndarray.to_numpy  --  zero-copy on CPU
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu])
def test_scalar_ndarray_to_numpy_zerocopy():
    nd = qd.ndarray(qd.f32, shape=(4,))
    nd[0] = 3.0
    nd[1] = 5.0
    qd.sync()
    arr = nd.to_numpy()
    np.testing.assert_allclose(arr[:2], [3.0, 5.0])


@test_utils.test(arch=[qd.cpu])
def test_vector_ndarray_to_numpy_zerocopy():
    nd = qd.Vector.ndarray(3, qd.f32, shape=(2,))
    nd[0] = (1, 2, 3)
    qd.sync()
    arr = nd.to_numpy()
    np.testing.assert_allclose(arr[0], [1, 2, 3])


@test_utils.test(arch=dlpack_arch)
def test_ndarray_to_numpy_matches_copy():
    nd = qd.ndarray(qd.f32, shape=(6,))
    for i in range(6):
        nd[i] = float(i * 7)
    qd.sync()
    arr_default = nd.to_numpy()
    arr_copy = nd.to_numpy(copy=True)
    np.testing.assert_allclose(arr_default, arr_copy)


# ---------------------------------------------------------------------------
# Two fields in the same SNode tree (non-zero offset)
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_to_torch_two_fields_same_tree():
    a = qd.field(qd.i32, (100,))
    b = qd.field(qd.i32, (100,))
    a[0] = 111
    b[0] = 222
    qd.sync()
    at = _to_cpu(a.to_torch())
    bt = _to_cpu(b.to_torch())
    assert at[0] == 111
    assert bt[0] == 222


# ---------------------------------------------------------------------------
# copy=False  --  error paths
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_copy_false_with_device_transfer_raises():
    """copy=False + device that differs from data device should raise."""
    if qd.cfg.arch == qd.cpu:
        pytest.skip("need GPU arch to test device mismatch")
    f = qd.field(qd.f32, shape=(3,))
    qd.sync()
    with pytest.raises(ValueError, match="copy=False"):
        f.to_torch(device="cpu", copy=False)


@test_utils.test(arch=[qd.cpu])
def test_copy_false_numpy_dtype_conversion_raises():
    """copy=False + dtype that requires conversion should raise."""
    f = qd.field(qd.f32, shape=(3,))
    qd.sync()
    with pytest.raises(ValueError, match="copy=False"):
        f.to_numpy(dtype=np.float64, copy=False)


# ---------------------------------------------------------------------------
# StructField pass-through
# ---------------------------------------------------------------------------


@test_utils.test(arch=dlpack_arch)
def test_struct_field_to_torch():
    s = qd.Struct.field({"a": qd.f32, "b": qd.f32}, shape=(4,))
    s[0] = {"a": 1.0, "b": 2.0}
    qd.sync()
    d = s.to_torch()
    assert isinstance(d, dict)
    assert _to_cpu(d["a"])[0] == 1.0
    assert _to_cpu(d["b"])[0] == 2.0


@test_utils.test(arch=dlpack_arch)
def test_struct_field_to_torch_copy_true():
    s = qd.Struct.field({"x": qd.f32}, shape=(3,))
    s[0] = {"x": 5.0}
    qd.sync()
    d1 = s.to_torch()
    d2 = s.to_torch(copy=True)
    assert d1["x"].data_ptr() != d2["x"].data_ptr()
    assert _to_cpu(d2["x"])[0] == 5.0
