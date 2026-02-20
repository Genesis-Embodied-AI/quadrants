"""
To test our new `qd.field` API is functional (#1500)
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants._test_tools.load_kernel_string import load_kernel_from_string
from quadrants.lang import impl
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

data_types = [qd.i32, qd.f32, qd.i64, qd.f64]
field_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalar_field(dtype, shape):
    x = qd.field(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)

    assert x.dtype == dtype


@pytest.mark.parametrize("n", vector_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_vector_field(n, dtype, shape):
    vec_type = qd.types.vector(n, dtype)
    x = qd.field(vec_type, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == 1


@pytest.mark.parametrize("n,m", matrix_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_matrix_field(n, m, dtype, shape):
    mat_type = qd.types.matrix(n, m, dtype)
    x = qd.field(dtype=mat_type, shape=shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalr_field_from_numpy(dtype, shape):
    import numpy as np

    x = qd.field(dtype, shape)
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        qd.i32: np.int32,
        qd.f32: np.float32,
        qd.f64: np.float64,
        qd.i64: np.int64,
    }
    arr = np.empty(shape, dtype=numpy_dtypes[dtype])
    x.from_numpy(arr)


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize(
    "shape, offset",
    [
        ((), None),
        ((), ()),
        (8, None),
        (8, 0),
        (8, 8),
        (8, -4),
        ((6, 12), None),
        ((6, 12), (0, 0)),
        ((6, 12), (-4, -4)),
        ((6, 12), (-4, 4)),
        ((6, 12), (4, -4)),
        ((6, 12), (8, 8)),
    ],
)
@test_utils.test(arch=get_host_arch_list())
def test_scalr_field_from_numpy_with_offset(dtype, shape, offset):
    import numpy as np

    x = qd.field(dtype=dtype, shape=shape, offset=offset)
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        qd.i32: np.int32,
        qd.f32: np.float32,
        qd.f64: np.float64,
        qd.i64: np.int64,
    }
    arr = np.ones(shape, dtype=numpy_dtypes[dtype])
    x.from_numpy(arr)

    def mat_equal(A, B, tol=1e-6):
        return np.max(np.abs(A - B)) < tol

    tol = 1e-5 if dtype == qd.f32 else 1e-12
    assert mat_equal(x.to_numpy(), arr, tol=tol)


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", field_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalr_field_from_numpy_with_mismatch_shape(dtype, shape):
    import numpy as np

    x = qd.field(dtype, shape)
    numpy_dtypes = {
        qd.i32: np.int32,
        qd.f32: np.float32,
        qd.f64: np.float64,
        qd.i64: np.int64,
    }
    # compose the mismatch shape for every qd.field.
    # set the shape to (2, 3) by default, if the qd.field shape is a tuple, set it to 1.
    mismatch_shape = (2, 3)
    if isinstance(shape, tuple):
        mismatch_shape = 1
    arr = np.empty(mismatch_shape, dtype=numpy_dtypes[dtype])
    with pytest.raises(ValueError):
        x.from_numpy(arr)


@test_utils.test(arch=get_host_arch_list())
def test_field_needs_grad():
    # Just make sure the usage doesn't crash, see #1545
    n = 8
    m1 = qd.field(dtype=qd.f32, shape=n, needs_grad=True)
    m2 = qd.field(dtype=qd.f32, shape=n, needs_grad=True)
    gr = qd.field(dtype=qd.f32, shape=n)

    @qd.kernel
    def func():
        for i in range(n):
            gr[i] = m1.grad[i] + m2.grad[i]

    func()


@test_utils.test()
def test_field_needs_grad_dtype():
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        a = qd.field(int, shape=1, needs_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        b = qd.field(qd.math.ivec3, shape=1, needs_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        mat_type = qd.types.matrix(2, 3, int)
        c = qd.field(dtype=mat_type, shape=1, needs_grad=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        d = qd.Struct.field(
            {
                "pos": qd.types.vector(3, int),
                "vel": qd.types.vector(3, float),
                "acc": qd.types.vector(3, float),
                "mass": qd.f32,
            },
            shape=1,
            needs_grad=True,
        )


@test_utils.test()
def test_field_needs_dual_dtype():
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        a = qd.field(int, shape=1, needs_dual=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        b = qd.field(qd.math.ivec3, shape=1, needs_dual=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        mat_type = qd.types.matrix(2, 3, int)
        c = qd.field(mat_type, shape=1, needs_dual=True)
    with pytest.raises(
        RuntimeError,
        match=r".* is not supported for field with `needs_grad=True` or `needs_dual=True`.",
    ):
        d = qd.Struct.field(
            {
                "pos": qd.types.vector(3, int),
                "vel": qd.types.vector(3, float),
                "acc": qd.types.vector(3, float),
                "mass": qd.f32,
            },
            shape=1,
            needs_dual=True,
        )


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
def test_default_fp(dtype):
    qd.init(default_fp=dtype)
    vec_type = qd.types.vector(3, dtype)

    x = qd.field(vec_type, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64])
def test_default_ip(dtype):
    qd.init(default_ip=dtype)

    x = qd.field(qd.math.ivec2, ())

    assert x.dtype == impl.get_runtime().default_ip


@test_utils.test()
def test_field_name():
    a = qd.field(dtype=qd.f32, shape=(2, 3), name="a")
    b = qd.field(qd.math.vec3, shape=(2, 3), name="b")
    c = qd.field(qd.math.mat3, shape=(5, 4), name="c")
    assert a._name == "a"
    assert b._name == "b"
    assert c._name == "c"
    assert b.snode._name == "b"
    d = []
    for i in range(10):
        d.append(qd.field(dtype=qd.f32, shape=(2, 3), name=f"d{i}"))
        assert d[i]._name == f"d{i}"


@test_utils.test()
@pytest.mark.parametrize("shape", field_shapes)
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32])
def test_field_copy_from(shape, dtype):
    x = qd.field(dtype=qd.f32, shape=shape)
    other = qd.field(dtype=dtype, shape=shape)
    other.fill(1)
    x.copy_from(other)
    convert = lambda arr: arr[0] if len(arr) == 1 else arr
    assert convert(x.shape) == shape
    assert x.dtype == qd.f32
    assert (x.to_numpy() == 1).all()


@test_utils.test()
def test_field_copy_from_with_mismatch_shape():
    x = qd.field(dtype=qd.f32, shape=(2, 3))
    for other_shape in [(2,), (2, 2), (2, 3, 4)]:
        other = qd.field(dtype=qd.f16, shape=other_shape)
        with pytest.raises(ValueError):
            x.copy_from(other)


@test_utils.test()
@pytest.mark.parametrize(
    "shape, x_offset, other_offset",
    [
        ((), (), ()),
        (8, 4, 0),
        (8, 0, -4),
        (8, -4, -4),
        (8, 8, -4),
        ((6, 12), (0, 0), (-6, -6)),
        ((6, 12), (-6, -6), (0, 0)),
        ((6, 12), (-6, -6), (-6, -6)),
    ],
)
@pytest.mark.parametrize("dtype", [qd.i32, qd.f32])
def test_field_copy_from_with_offset(shape, dtype, x_offset, other_offset):
    x = qd.field(dtype=qd.f32, shape=shape, offset=x_offset)
    other = qd.field(dtype=dtype, shape=shape, offset=other_offset)
    other.fill(1)
    x.copy_from(other)
    convert = lambda arr: arr[0] if len(arr) == 1 else arr
    assert convert(x.shape) == shape
    assert x.dtype == qd.f32
    assert (x.to_numpy() == 1).all()


@test_utils.test()
def test_field_copy_from_with_non_filed_object():
    import numpy as np

    x = qd.field(dtype=qd.f32, shape=(2, 3))
    other = np.zeros((2, 3))
    with pytest.raises(TypeError):
        x.copy_from(other)


@test_utils.test()
def test_field_shape_0():
    with pytest.raises(
        qd._lib.core.QuadrantsRuntimeError,
        match="Every dimension of a Quadrants field should be positive",
    ):
        x = qd.field(dtype=qd.f32, shape=0)


@test_utils.test()
def test_index_mismatch():
    with pytest.raises(AssertionError, match="Slicing is not supported on qd.field"):
        val = qd.field(qd.i32, shape=(1, 2, 3))
        val[0, 0] = 1


@test_utils.test()
def test_invalid_slicing():
    with pytest.raises(
        TypeError,
        match="Detected illegal element of type: .*?\. Please be aware that slicing a qd.field is not supported so far.",
    ):
        val = qd.field(qd.i32, shape=(2, 2))
        val[0, :]


@test_utils.test()
def test_indexing_with_np_int():
    val = qd.field(qd.i32, shape=(2))
    idx = np.int32(0)
    val[idx]


@test_utils.test()
def test_indexing_vec_field_with_np_int():
    val = qd.field(qd.math.ivec2, shape=(2))
    idx = np.int32(0)
    val[idx][idx]


@test_utils.test()
def test_indexing_mat_field_with_np_int():
    mat_type = qd.types.matrix(2, 2, int)
    val = qd.field(mat_type, shape=(2))
    idx = np.int32(0)
    val[idx][idx, idx]


@test_utils.test()
def test_python_for_in():
    x = qd.field(int, shape=3)
    with pytest.raises(NotImplementedError, match="Struct for is only available in Quadrants scope"):
        for i in x:
            pass


@test_utils.test()
def test_matrix_mult_field():
    x = qd.field(int, shape=())
    with pytest.raises(qd.QuadrantsTypeError, match="unsupported operand type"):

        @qd.kernel
        def foo():
            a = qd.Vector([1, 1, 1])
            b = a * x

        foo()


@test_utils.test(exclude=[qd.x64, qd.arm64, qd.cuda])
def test_sparse_not_supported():
    with pytest.raises(qd.QuadrantsRuntimeError, match="Pointer SNode is not supported on this backend."):
        qd.root.pointer(qd.i, 10)

    with pytest.raises(qd.QuadrantsRuntimeError, match="Pointer SNode is not supported on this backend."):
        a = qd.root.dense(qd.i, 10)
        a.pointer(qd.j, 10)

    with pytest.raises(qd.QuadrantsRuntimeError, match="Dynamic SNode is not supported on this backend."):
        qd.root.dynamic(qd.i, 10)

    with pytest.raises(qd.QuadrantsRuntimeError, match="Dynamic SNode is not supported on this backend."):
        a = qd.root.dense(qd.i, 10)
        a.dynamic(qd.j, 10)

    with pytest.raises(qd.QuadrantsRuntimeError, match="Bitmasked SNode is not supported on this backend."):
        qd.root.bitmasked(qd.i, 10)

    with pytest.raises(qd.QuadrantsRuntimeError, match="Bitmasked SNode is not supported on this backend."):
        a = qd.root.dense(qd.i, 10)
        a.bitmasked(qd.j, 10)


@test_utils.test(require=qd.extension.data64)
def test_write_u64():
    x = qd.field(qd.u64, shape=())
    x[None] = 2**64 - 1
    assert x[None] == 2**64 - 1


@test_utils.test(require=qd.extension.data64)
def test_field_with_dynamic_index():
    vel = qd.Vector.field(2, dtype=qd.f64, shape=(100, 100))

    @qd.func
    def foo(i, j, l):
        tmp = 1.0 / vel[i, j][l]
        return tmp

    @qd.kernel
    def collide():
        tmp0 = foo(0, 0, 0)
        print(tmp0)

    collide()


@test_utils.test()
def test_field_max_num_args() -> None:
    num_args = 512
    kernel_templ = """
import quadrants as qd
@qd.kernel
def my_kernel({args}) -> None:
{arg_uses}
"""
    args_l = []
    arg_uses_l = []
    arg_objs_l = []
    for i in range(num_args):
        args_l.append(f"a{i}: qd.Template")
        arg_uses_l.append(f"    a{i}[0] += {i + 1}")
        arg_objs_l.append(qd.field(qd.i32, (10,)))
    args_str = ", ".join(args_l)
    arg_uses_str = "\n".join(arg_uses_l)
    kernel_str = kernel_templ.format(args=args_str, arg_uses=arg_uses_str)
    with load_kernel_from_string(kernel_str, "my_kernel") as my_kernel:
        my_kernel(*arg_objs_l)
    for i in range(num_args):
        assert arg_objs_l[i][0] == i + 1
