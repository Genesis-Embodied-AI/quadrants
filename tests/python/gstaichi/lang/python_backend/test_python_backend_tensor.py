"""Tests for MyTorchTensor batch_ndim, shape property, and _tc/_T_tc views."""

import numpy as np
import torch

import quadrants as qd
from quadrants.lang._py_tensor import MyTorchTensor, create_tensor


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


def test_tensor_iter_yields_scalars():
    """Iterating a 1-D MyTorchTensor should yield Python scalars, not 0-d tensors."""
    qd.init(qd.python)
    t = MyTorchTensor(torch.tensor([10, 20, 30]))
    items = list(t)
    assert all(isinstance(x, int) for x in items)
    assert items == [10, 20, 30]

    t2 = MyTorchTensor(torch.tensor([1.5, 2.5]))
    items2 = list(t2)
    assert all(isinstance(x, float) for x in items2)
    assert items2 == [1.5, 2.5]


def test_tensor_star_unpack_in_tuple():
    """*J unpacking in tuple should yield ints for numpy indexing."""
    qd.init(qd.python)
    import numpy as np

    arr = np.zeros((3, 4, 5))
    arr[1, 2, 3] = 42.0
    J = MyTorchTensor(torch.tensor([2, 3]))
    assert arr[(1, *J)] == 42.0


def test_grouped_vector_index():
    """MyTorchTensor used as a vector index should unpack to multi-dim indexing."""
    qd.init(qd.python)
    f = qd.field(qd.math.vec3, shape=(3, 2))
    f[1, 0] = torch.tensor([10.0, 20.0, 30.0])
    idx = MyTorchTensor(torch.tensor([1, 0]))
    result = f[idx]
    assert result[0].item() == 10.0
    assert result[1].item() == 20.0
    f[idx] = torch.tensor([99.0, 88.0, 77.0])
    assert f[1, 0, 0].item() == 99.0


def test_isnan():
    qd.init(qd.python)

    @qd.kernel
    def check_nan(a: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(a.shape[0]):
            out[i] = 1 if qd.math.isnan(a[i]) else 0

    a = qd.ndarray(qd.f32, shape=(3,))
    out = qd.ndarray(qd.i32, shape=(3,))
    a[0] = 1.0
    a[1] = float("nan")
    a[2] = float("inf")
    check_nan(a, out)
    assert out[0] == 0
    assert out[1] == 1
    assert out[2] == 0


def test_isinf():
    qd.init(qd.python)

    @qd.kernel
    def check_inf(a: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(a.shape[0]):
            out[i] = 1 if qd.math.isinf(a[i]) else 0

    a = qd.ndarray(qd.f32, shape=(3,))
    out = qd.ndarray(qd.i32, shape=(3,))
    a[0] = 1.0
    a[1] = float("nan")
    a[2] = float("inf")
    check_inf(a, out)
    assert out[0] == 0
    assert out[1] == 0
    assert out[2] == 1


def test_cast():
    qd.init(qd.python)

    @qd.kernel
    def do_cast(a: qd.types.ndarray(dtype=qd.f32, ndim=1), out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(a.shape[0]):
            out[i] = qd.cast(a[i], qd.i32)

    a = qd.ndarray(qd.f32, shape=(3,))
    out = qd.ndarray(qd.i32, shape=(3,))
    a[0] = 1.7
    a[1] = 2.3
    a[2] = -0.9
    do_cast(a, out)
    assert out[0] == 1
    assert out[1] == 2
    assert out[2] == 0


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


def test_dtype_call_f32():
    """qd.f32(0.0) should work in the python backend (DataTypeCxx.__call__)."""
    qd.init(qd.python)

    @qd.kernel
    def use_dtype_call(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        x = qd.f32(0.0)
        y = qd.f32(3)
        a[0] = x
        a[1] = y

    a = qd.ndarray(qd.f32, shape=(2,))
    use_dtype_call(a)
    assert a[0] == 0.0
    assert a[1] == 3.0


def test_dtype_call_i32():
    """qd.i32(3.7) should truncate to 3 in the python backend."""
    qd.init(qd.python)

    @qd.kernel
    def use_i32(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        a[0] = qd.i32(3.7)

    a = qd.ndarray(qd.i32, shape=(1,))
    use_i32(a)
    assert a[0] == 3


def test_ndrange_in_kernel():
    """qd.ndrange() should work inside a python backend kernel."""
    qd.init(qd.python)

    @qd.kernel
    def fill_2d(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        idx = 0
        for i, j in qd.ndrange(3, 4):
            a[idx] = i * 10 + j
            idx += 1

    a = qd.ndarray(qd.i32, shape=(12,))
    fill_2d(a)
    assert a[0] == 0
    assert a[1] == 1
    assert a[4] == 10
    assert a[11] == 23


def test_ndrange_with_tensor_bounds():
    """ndrange should accept 0-d tensor bounds (from .shape[0])."""
    qd.init(qd.python)

    @qd.kernel
    def sum_2d(
        a: qd.types.ndarray(dtype=qd.f32, ndim=2),
        out: qd.types.ndarray(dtype=qd.f32, ndim=1),
    ):
        total = qd.f32(0.0)
        for i, j in qd.ndrange(a.shape[0], a.shape[1]):
            total += a[i, j]
        out[0] = total

    a = qd.ndarray(qd.f32, shape=(3, 4))
    a.fill(1.0)
    out = qd.ndarray(qd.f32, shape=(1,))
    sum_2d(a, out)
    assert out[0] == 12.0


def test_transpose_no_args():
    """MyTorchTensor.transpose() with no args should transpose a 2D matrix."""
    qd.init(qd.python)
    t = MyTorchTensor([[1, 2, 3], [4, 5, 6]])
    tt = t.transpose()
    assert tt.size() == torch.Size([3, 2])
    assert tt[0, 0] == 1
    assert tt[0, 1] == 4
    assert tt[2, 1] == 6


def test_torch_function_with_foreign_subclass():
    """__torch_function__ should unwrap MyTorchTensor when mixed with foreign subclasses."""
    qd.init(qd.python)

    class ForeignTensor(torch.Tensor):
        pass

    a = MyTorchTensor([1.0, 2.0, 3.0])
    b = ForeignTensor([10.0, 20.0, 30.0])
    result = a + b
    assert not isinstance(result, MyTorchTensor)
    assert torch.equal(result, torch.tensor([11.0, 22.0, 33.0]))


def test_getattr_delegation_norm():
    """MyTorchTensor should delegate .norm() to matrix_ops."""
    qd.init(qd.python)
    v = MyTorchTensor([3.0, 4.0])
    n = v.norm()
    assert abs(n - 5.0) < 1e-5


def test_getattr_delegation_dot():
    """MyTorchTensor should delegate .dot() to matrix_ops."""
    qd.init(qd.python)
    a = MyTorchTensor([1.0, 2.0, 3.0])
    b = MyTorchTensor([4.0, 5.0, 6.0])
    d = a.dot(b)
    assert abs(d - 32.0) < 1e-5


def test_sync_noop():
    """qd.sync() should not crash on the python backend."""
    qd.init(qd.python)
    a = qd.field(qd.f32, shape=(4,))
    a[0] = 1.0
    qd.sync()
    assert a[0] == 1.0
