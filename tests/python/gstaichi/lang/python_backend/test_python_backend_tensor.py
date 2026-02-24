"""Tests for MyTorchTensor batch_ndim, shape property, and _tc/_T_tc views."""

import numpy as np
import torch

import quadrants as qd
from quadrants.lang.py_tensor import MyTorchTensor, create_tensor, _setup_views


def test_create_tensor_scalar_1d():
    qd.init(qd.python)
    t = create_tensor((5,), torch.float32, batch_ndim=1)
    assert isinstance(t, MyTorchTensor)
    assert t.size() == torch.Size([5])
    assert t.shape == torch.Size([5])
    assert t._batch_shape == torch.Size([5])


def test_create_tensor_scalar_2d():
    qd.init(qd.python)
    t = create_tensor((4, 3), torch.float32, batch_ndim=2)
    assert t.size() == torch.Size([4, 3])
    assert t.shape == torch.Size([4, 3])
    assert t._batch_shape == torch.Size([4, 3])


def test_create_tensor_vec3_1d_batch():
    """vec3 field with shape (N,) -> real size (N, 3), batch_shape (N,)."""
    qd.init(qd.python)
    t = create_tensor((10, 3), torch.float32, batch_ndim=1)
    assert t.size() == torch.Size([10, 3])
    assert t.shape == torch.Size([10])
    assert t._batch_shape == torch.Size([10])


def test_create_tensor_vec3_2d_batch():
    """vec3 field with shape (N, M) -> real size (N, M, 3), batch_shape (N, M)."""
    qd.init(qd.python)
    t = create_tensor((10, 4, 3), torch.float32, batch_ndim=2)
    assert t.size() == torch.Size([10, 4, 3])
    assert t.shape == torch.Size([10, 4])
    assert t._batch_shape == torch.Size([10, 4])


def test_create_tensor_mat3x3_1d_batch():
    """mat3x3 field with shape (N,) -> real size (N, 3, 3), batch_shape (N,)."""
    qd.init(qd.python)
    t = create_tensor((5, 3, 3), torch.float32, batch_ndim=1)
    assert t.size() == torch.Size([5, 3, 3])
    assert t.shape == torch.Size([5])
    assert t._batch_shape == torch.Size([5])


def test_tc_is_self():
    qd.init(qd.python)
    t = create_tensor((10, 3), torch.float32, batch_ndim=1)
    assert t._tc is t


def test_T_tc_1d_batch_is_self():
    """With 1D batch, _T_tc should be the same tensor (no transpose needed)."""
    qd.init(qd.python)
    t = create_tensor((10, 3), torch.float32, batch_ndim=1)
    assert t._T_tc is t


def test_T_tc_2d_batch_transposes_batch_dims():
    """With 2D batch (N, M), _T_tc moves dim 1 to front -> (M, N, ...)."""
    qd.init(qd.python)
    t = create_tensor((10, 4, 3), torch.float32, batch_ndim=2)
    t[2, 1, :] = torch.tensor([1.0, 2.0, 3.0])

    tt = t._T_tc
    assert tt.size() == torch.Size([4, 10, 3])
    assert torch.equal(tt[1, 2, :], torch.tensor([1.0, 2.0, 3.0]))


def test_np_view():
    qd.init(qd.python)
    t = create_tensor((3, 2), torch.float32, batch_ndim=1)
    t[1, :] = torch.tensor([5.0, 6.0])
    arr = t._np
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 2)
    np.testing.assert_array_equal(arr[1], [5.0, 6.0])


def test_T_np_view():
    qd.init(qd.python)
    t = create_tensor((3, 4, 2), torch.float32, batch_ndim=2)
    t[1, 2, :] = torch.tensor([7.0, 8.0])
    arr = t._T_np
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 3, 2)
    np.testing.assert_array_equal(arr[2, 1], [7.0, 8.0])


def test_batch_ndim_default_all_dims():
    """When batch_ndim is None, all dims are batch dims."""
    qd.init(qd.python)
    t = create_tensor((3, 4), torch.float32)
    assert t.shape == torch.Size([3, 4])
    assert t._batch_shape == torch.Size([3, 4])


def test_field_scalar_shape():
    """qd.field(scalar, shape) should have batch_shape == shape."""
    qd.init(qd.python)
    f = qd.field(qd.f32, shape=(8,))
    assert f.shape == torch.Size([8])
    assert f.size() == torch.Size([8])


def test_field_vec3_shape():
    """qd.field(vec3, shape) should have batch_shape == shape, size includes element dim."""
    qd.init(qd.python)
    f = qd.field(qd.math.vec3, shape=(10,))
    assert f.shape == torch.Size([10])
    assert f.size() == torch.Size([10, 3])


def test_field_vec3_2d_shape():
    """qd.field(vec3, shape=(N,M)) -> batch_shape (N,M), size (N,M,3)."""
    qd.init(qd.python)
    f = qd.field(qd.math.vec3, shape=(10, 4))
    assert f.shape == torch.Size([10, 4])
    assert f.size() == torch.Size([10, 4, 3])


def test_field_mat3x3_shape():
    """qd.field(mat3, shape=(N,)) -> batch_shape (N,), size (N,3,3)."""
    qd.init(qd.python)
    f = qd.field(qd.math.mat3, shape=(5,))
    assert f.shape == torch.Size([5])
    assert f.size() == torch.Size([5, 3, 3])


def test_ndarray_vec3_shape():
    """qd.ndarray(vec3, shape) should have correct batch_shape and size."""
    qd.init(qd.python)
    a = qd.ndarray(qd.math.vec3, shape=(6,))
    assert a.shape == torch.Size([6])
    assert a.size() == torch.Size([6, 3])


def test_ndarray_mat4x4_shape():
    qd.init(qd.python)
    a = qd.ndarray(qd.math.mat4, shape=(3, 2))
    assert a.shape == torch.Size([3, 2])
    assert a.size() == torch.Size([3, 2, 4, 4])


def test_field_vec3_T_tc_2d_batch():
    """qd.field(vec3, (N,M)) -> _T_tc has shape (M, N, 3)."""
    qd.init(qd.python)
    f = qd.field(qd.math.vec3, shape=(10, 4))
    f[2, 1] = torch.tensor([1.0, 2.0, 3.0])

    tt = f._T_tc
    assert tt.size() == torch.Size([4, 10, 3])
    assert torch.equal(tt[1, 2], torch.tensor([1.0, 2.0, 3.0]))


def test_field_none_shape():
    """qd.field with shape=None should work (0-d scalar)."""
    qd.init(qd.python)
    f = qd.field(qd.f32, shape=None)
    assert f.size() == torch.Size([])
    assert f.shape == torch.Size([])


def test_struct_field():
    """Struct fields should create tensors for each member with the correct shape."""
    qd.init(qd.python)
    my_struct = qd.types.struct(x=qd.f32, y=qd.f32, z=qd.f32)
    sf = my_struct.field(shape=(4,))
    assert sf.field_dict["x"].size() == torch.Size([4])
    assert sf.field_dict["y"].size() == torch.Size([4])
    assert sf.field_dict["z"].size() == torch.Size([4])
    sf.field_dict["x"][2] = 42.0
    assert sf.field_dict["x"][2] == 42.0


def test_struct_field_with_vec():
    """Struct fields with vector members."""
    qd.init(qd.python)
    my_struct = qd.types.struct(pos=qd.math.vec3, mass=qd.f32)
    sf = my_struct.field(shape=(5,))
    assert sf.field_dict["pos"].size() == torch.Size([5, 3])
    assert sf.field_dict["pos"].shape == torch.Size([5])
    assert sf.field_dict["mass"].size() == torch.Size([5])
    sf.field_dict["pos"][2] = torch.tensor([1.0, 2.0, 3.0])
    assert sf.field_dict["pos"][2, 1].item() == 2.0


def test_size_in_kernel():
    """Kernel code uses .shape[0] which should return the first batch dim."""
    qd.init(qd.python)
    a = qd.ndarray(qd.math.vec3, shape=(10,))

    @qd.kernel
    def fill_vecs(a: qd.types.ndarray(dtype=qd.math.vec3, ndim=1)):
        for i in range(a.shape[0]):
            a[i] = qd.Vector([float(i), float(i * 2), float(i * 3)])

    fill_vecs(a)
    assert a[3, 0].item() == 3.0
    assert a[3, 1].item() == 6.0
    assert a[3, 2].item() == 9.0
