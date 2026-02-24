import copy
import platform
import sys

import numpy as np
import pytest

import quadrants as qd
from quadrants._test_tools.load_kernel_string import load_kernel_from_string
from quadrants.lang import impl
from quadrants.lang.exception import (
    QuadrantsIndexError,
    QuadrantsRuntimeError,
    QuadrantsTypeError,
)
from quadrants.lang.misc import get_host_arch_list
from quadrants.lang.util import has_pytorch
from quadrants.math import ivec3, vec3

from tests import test_utils

if has_pytorch():
    import torch  # noqa: F401

# properties

data_types = [qd.i32, qd.f32, qd.i64, qd.f64]
ndarray_shapes = [(), 8, (6, 12)]
vector_dims = [3]
matrix_dims = [(1, 2), (2, 3)]
supported_archs_quadrants_ndarray = [
    qd.cpu,
    qd.cuda,
    qd.vulkan,
    qd.metal,
    qd.amdgpu,
]


def _test_scalar_ndarray(dtype, shape):
    x = qd.ndarray(dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == ()

    assert x.dtype == dtype


@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_scalar_ndarray(dtype, shape):
    _test_scalar_ndarray(dtype, shape)


def _test_vector_ndarray(n, dtype, shape):
    x = qd.Vector.ndarray(n, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == (n,)

    assert x.dtype == dtype
    assert x.n == n


@pytest.mark.parametrize("n", vector_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_vector_ndarray(n, dtype, shape):
    _test_vector_ndarray(n, dtype, shape)


def _test_matrix_ndarray(n, m, dtype, shape):
    x = qd.Matrix.ndarray(n, m, dtype, shape)

    if isinstance(shape, tuple):
        assert x.shape == shape
    else:
        assert x.shape == (shape,)
    assert x.element_shape == (n, m)

    assert x.dtype == dtype
    assert x.n == n
    assert x.m == m


@pytest.mark.parametrize("n,m", matrix_dims)
@pytest.mark.parametrize("dtype", data_types)
@pytest.mark.parametrize("shape", ndarray_shapes)
@test_utils.test(arch=get_host_arch_list())
def test_matrix_ndarray(n, m, dtype, shape):
    _test_matrix_ndarray(n, m, dtype, shape)


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_default_fp_ndarray(dtype):
    arch = qd.lang.impl.current_cfg().arch
    qd.reset()
    qd.init(arch=arch, default_fp=dtype)

    x = qd.Vector.ndarray(2, float, ())

    assert x.dtype == impl.get_runtime().default_fp


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64])
@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_default_ip_ndarray(dtype):
    arch = qd.lang.impl.current_cfg().arch
    qd.reset()
    qd.init(arch=arch, default_ip=dtype)

    x = qd.Vector.ndarray(2, int, ())

    assert x.dtype == impl.get_runtime().default_ip


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_1d():
    n = 4

    @qd.kernel
    def run(x: qd.types.NDArray[qd.i32, 1], y: qd.types.NDArray[qd.i32, 1]):
        for i in range(n):
            x[i] += i + y[i]

    a = qd.ndarray(qd.i32, shape=(n,))
    for i in range(n):
        a[i] = i * i
    b = np.ones((n,), dtype=np.int32)
    run(a, b)
    for i in range(n):
        assert a[i] == i * i + i + 1
    run(b, a)
    for i in range(n):
        assert b[i] == i * i + (i + 1) * 2


def _test_ndarray_2d():
    n = 4
    m = 7

    @qd.kernel
    def run(x: qd.types.NDArray[qd.i32, 2], y: qd.types.NDArray[qd.i32, 2]):
        for i in range(n):
            for j in range(m):
                x[i, j] += i + j + y[i, j]

    a = qd.ndarray(qd.i32, shape=(n, m))
    for i in range(n):
        for j in range(m):
            a[i, j] = i * j
    b = np.ones((n, m), dtype=np.int32)
    run(a, b)
    for i in range(n):
        for j in range(m):
            assert a[i, j] == i * j + i + j + 1
    run(b, a)
    for i in range(n):
        for j in range(m):
            assert b[i, j] == i * j + (i + j + 1) * 2


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_2d():
    _test_ndarray_2d()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_compound_element():
    n = 10
    a = qd.ndarray(qd.i32, shape=(n,))

    vec3 = qd.types.vector(3, qd.i32)
    b = qd.ndarray(vec3, shape=(n, n))
    assert isinstance(b, qd.VectorNdarray)
    assert b.shape == (n, n)
    assert b.element_type.element_type() == qd.i32
    assert b.element_type.shape() == [3]

    matrix34 = qd.types.matrix(3, 4, float)
    c = qd.ndarray(matrix34, shape=(n, n + 1))
    assert isinstance(c, qd.MatrixNdarray)
    assert c.shape == (n, n + 1)
    assert c.element_type.element_type() == qd.f32
    assert c.element_type.shape() == [3, 4]


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_copy_from_ndarray():
    n = 16
    a = qd.ndarray(qd.i32, shape=n)
    b = qd.ndarray(qd.i32, shape=n)
    a[0] = 1
    a[4] = 2
    b[0] = 4
    b[4] = 5

    a.copy_from(b)

    assert a[0] == 4
    assert a[4] == 5

    x = qd.Vector.ndarray(10, qd.i32, 5)
    y = qd.Vector.ndarray(10, qd.i32, 5)
    x[1][0] = 1
    x[2][4] = 2
    y[1][0] = 4
    y[2][4] = 5

    x.copy_from(y)

    assert x[1][0] == 4
    assert x[2][4] == 5

    x = qd.Matrix.ndarray(2, 2, qd.i32, 5)
    y = qd.Matrix.ndarray(2, 2, qd.i32, 5)
    x[0][0, 0] = 1
    x[4][1, 0] = 3
    y[0][0, 0] = 4
    y[4][1, 0] = 6

    x.copy_from(y)

    assert x[0][0, 0] == 4
    assert x[4][1, 0] == 6


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_deepcopy():
    n = 16
    x = qd.ndarray(qd.i32, shape=n)
    x[0] = 1
    x[4] = 2

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y[0] == 1
    assert y[4] == 2
    x[0] = 4
    x[4] = 5
    assert y[0] == 1
    assert y[4] == 2

    x = qd.Vector.ndarray(10, qd.i32, 5)
    x[1][0] = 4
    x[2][4] = 5

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.n == x.n
    assert y[1][0] == 4
    assert y[2][4] == 5
    x[1][0] = 1
    x[2][4] = 2
    assert y[1][0] == 4
    assert y[2][4] == 5

    x = qd.Matrix.ndarray(2, 2, qd.i32, 5)
    x[0][0, 0] = 7
    x[4][1, 0] = 9

    y = copy.deepcopy(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.m == x.m
    assert y.n == x.n
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9
    x[0][0, 0] = 3
    x[4][1, 0] = 5
    assert y[0][0, 0] == 7
    assert y[4][1, 0] == 9


@test_utils.test(arch=[qd.cuda])
def test_ndarray_caching_allocator():
    n = 8
    a = qd.ndarray(qd.i32, shape=(n))
    a.fill(2)
    a = 1
    b = qd.ndarray(qd.i32, shape=(n))
    b.fill(2)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_fill():
    n = 8
    a = qd.ndarray(qd.i32, shape=(n))
    anp = np.ones((n,), dtype=np.int32)
    a.fill(2)
    anp.fill(2)
    assert (a.to_numpy() == anp).all()

    b = qd.Vector.ndarray(4, qd.f32, shape=(n))
    bnp = np.ones(shape=b.arr.total_shape(), dtype=np.float32)
    b.fill(2.5)
    bnp.fill(2.5)
    assert (b.to_numpy() == bnp).all()

    c = qd.Matrix.ndarray(4, 4, qd.f32, shape=(n))
    cnp = np.ones(shape=c.arr.total_shape(), dtype=np.float32)
    c.fill(1.5)
    cnp.fill(1.5)
    assert (c.to_numpy() == cnp).all()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_rw_cache():
    a = qd.Vector.ndarray(3, qd.f32, ())
    b = qd.Vector.ndarray(3, qd.f32, 12)

    n = 100
    for i in range(n):
        c_a = copy.deepcopy(a)
        c_b = copy.deepcopy(b)
        c_a[None] = c_b[10]


def _test_ndarray_numpy_io():
    n = 7
    m = 4
    a = qd.ndarray(qd.i32, shape=(n, m))
    a.fill(2)
    b = qd.ndarray(qd.i32, shape=(n, m))
    b.from_numpy(np.ones((n, m), dtype=np.int32) * 2)
    assert (a.to_numpy() == b.to_numpy()).all()

    d = 2
    p = 4
    x = qd.Vector.ndarray(d, qd.f32, p)
    x.fill(2)
    y = qd.Vector.ndarray(d, qd.f32, p)
    y.from_numpy(np.ones((p, d), dtype=np.int32) * 2)
    assert (x.to_numpy() == y.to_numpy()).all()

    c = 2
    d = 2
    p = 4
    x = qd.Matrix.ndarray(c, d, qd.f32, p)
    x.fill(2)
    y = qd.Matrix.ndarray(c, d, qd.f32, p)
    y.from_numpy(np.ones((p, c, d), dtype=np.int32) * 2)
    assert (x.to_numpy() == y.to_numpy()).all()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_numpy_io():
    _test_ndarray_numpy_io()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_matrix_numpy_io():
    n = 5
    m = 2

    x = qd.Vector.ndarray(n, qd.i32, (m,))
    x_np = 1 + np.arange(n * m).reshape(m, n).astype(np.int32)
    x.from_numpy(x_np)
    assert (x_np.flatten() == x.to_numpy().flatten()).all()

    k = 2
    x = qd.Matrix.ndarray(m, k, qd.i32, n)
    x_np = 1 + np.arange(m * k * n).reshape(n, m, k).astype(np.int32)
    x.from_numpy(x_np)
    assert (x_np.flatten() == x.to_numpy().flatten()).all()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_matrix_ndarray_python_scope():
    a = qd.Matrix.ndarray(2, 2, qd.i32, 5)
    for i in range(5):
        for j, k in qd.ndrange(2, 2):
            a[i][j, k] = j * j + k * k
    assert a[0][0, 0] == 0
    assert a[1][0, 1] == 1
    assert a[2][1, 0] == 1
    assert a[3][1, 1] == 2
    assert a[4][0, 1] == 1


def _test_matrix_ndarray_quadrants_scope():
    @qd.kernel
    def func(a: qd.types.NDArray[qd.types.matrix(2, 2, qd.i32), 1]):
        for i in range(5):
            for j, k in qd.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = qd.Matrix.ndarray(2, 2, qd.i32, 5)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_matrix_ndarray_quadrants_scope():
    _test_matrix_ndarray_quadrants_scope()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_matrix_ndarray_quadrants_scope_real_matrix():
    _test_matrix_ndarray_quadrants_scope()


def _test_matrix_ndarray_quadrants_scope_struct_for():
    @qd.kernel
    def func(a: qd.types.NDArray[qd.types.matrix(2, 2, qd.i32), 1]):
        for i in a:
            for j, k in qd.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = qd.Matrix.ndarray(2, 2, qd.i32, 5)
    func(m)
    assert m[0][0, 0] == 0
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_matrix_ndarray_quadrants_scope_struct_for():
    _test_matrix_ndarray_quadrants_scope_struct_for()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_matrix_ndarray_quadrants_scope_struct_for_real_matrix():
    _test_matrix_ndarray_quadrants_scope_struct_for()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_vector_ndarray_python_scope():
    a = qd.Vector.ndarray(10, qd.i32, 5)
    for i in range(5):
        for j in range(4):
            a[i][j * j] = j * j
    assert a[0][9] == 9
    assert a[1][0] == 0
    assert a[2][1] == 1
    assert a[3][4] == 4
    assert a[4][9] == 9


def _test_vector_ndarray_quadrants_scope():
    @qd.kernel
    def func(a: qd.types.NDArray[qd.types.vector(10, qd.i32), 1]):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = qd.Vector.ndarray(10, qd.i32, 5)
    func(v)
    assert v[0][9] == 9
    assert v[1][0] == 0
    assert v[2][1] == 1
    assert v[3][4] == 4
    assert v[4][9] == 9


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_vector_ndarray_quadrants_scope():
    _test_vector_ndarray_quadrants_scope()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_vector_ndarray_quadrants_scope_real_matrix():
    _test_vector_ndarray_quadrants_scope()


# number of compiled functions
def _test_compiled_functions():
    @qd.kernel
    def func(a: qd.types.NDArray[qd.types.vector(n=10, dtype=qd.i32), 1]):
        for i in range(5):
            for j in range(4):
                a[i][j * j] = j * j

    v = qd.Vector.ndarray(10, qd.i32, 5)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1
    v = np.zeros((6, 10), dtype=np.int32)
    func(v)
    assert impl.get_runtime().get_num_compiled_functions() == 1


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_compiled_functions():
    _test_compiled_functions()


# annotation compatibility


def _test_arg_not_match():
    @qd.kernel
    def func1(a: qd.types.NDArray[qd.types.vector(2, qd.i32), 2]):
        pass

    x = qd.Matrix.ndarray(2, 3, qd.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: VectorType\[2, i32\], but .* is provided",
    ):
        func1(x)

    x = qd.Matrix.ndarray(2, 1, qd.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: VectorType\[2, i32\], but .* is provided",
    ):
        func1(x)

    @qd.kernel
    def func2(a: qd.types.NDArray[qd.types.matrix(2, 2, qd.i32), 2]):
        pass

    x = qd.Vector.ndarray(2, qd.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: MatrixType\[2,2, i32\], but .* is provided",
    ):
        func2(x)

    @qd.kernel
    def func3(a: qd.types.NDArray[qd.types.matrix(2, 1, qd.i32), 2]):
        pass

    x = qd.Vector.ndarray(2, qd.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type: MatrixType\[2,1, i32\], but .* is provided",
    ):
        func3(x)

    @qd.kernel
    def func5(a: qd.types.NDArray[qd.types.matrix(2, 3, dtype=qd.i32), 2]):
        pass

    x = qd.Vector.ndarray(2, qd.i32, shape=(4, 7))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required element type",
    ):
        func5(x)

    @qd.kernel
    def func7(a: qd.types.NDArray[qd.i32, 2]):
        pass

    x = qd.ndarray(qd.i32, shape=(3,))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument a - required ndim",
    ):
        func7(x)

    @qd.kernel
    def func8(x: qd.types.NDArray[qd.f32, 2]):
        pass

    x = qd.ndarray(dtype=qd.i32, shape=(16, 16))
    with pytest.raises(TypeError, match=r"Expect element type .* for argument x, but get .*"):
        func8(x)


@test_utils.test(arch=get_host_arch_list())
def test_arg_not_match():
    _test_arg_not_match()


def _test_size_in_bytes():
    a = qd.ndarray(qd.i32, 8)
    assert a._get_element_size() == 4
    assert a._get_nelement() == 8

    b = qd.Vector.ndarray(10, qd.f64, 5)
    assert b._get_element_size() == 80
    assert b._get_nelement() == 5


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_size_in_bytes():
    _test_size_in_bytes()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_different_shape():
    n1 = 4
    x = qd.ndarray(dtype=qd.f32, shape=(n1, n1))

    @qd.kernel
    def init(d: qd.i32, arr: qd.types.NDArray):
        for i, j in arr:
            arr[i, j] = d

    init(2, x)
    assert (x.to_numpy() == (np.ones(shape=(n1, n1)) * 2)).all()
    n2 = 8
    y = qd.ndarray(dtype=qd.f32, shape=(n2, n2))
    init(3, y)
    assert (y.to_numpy() == (np.ones(shape=(n2, n2)) * 3)).all()


def _test_ndarray_grouped():
    @qd.kernel
    def func(a: qd.types.NDArray):
        for i in qd.grouped(a):
            for j, k in qd.ndrange(2, 2):
                a[i][j, k] = j * j

    a1 = qd.Matrix.ndarray(2, 2, qd.i32, shape=5)
    func(a1)
    for i in range(5):
        for j in range(2):
            for k in range(2):
                assert a1[i][j, k] == j * j

    a2 = qd.Matrix.ndarray(2, 2, qd.i32, shape=(3, 3))
    func(a2)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                for p in range(2):
                    assert a2[i, j][k, p] == k * k


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_grouped():
    _test_ndarray_grouped()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_ndarray_grouped_real_matrix():
    _test_ndarray_grouped()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_as_template():
    @qd.kernel
    def func(arr_src: qd.Template, arr_dst: qd.Template):
        for i, j in qd.ndrange(*arr_src.shape):
            arr_dst[i, j] = arr_src[i, j]

    arr_0 = qd.ndarray(qd.f32, shape=(5, 10))
    arr_1 = qd.ndarray(qd.f32, shape=(5, 10))
    with pytest.raises(qd.QuadrantsRuntimeTypeError, match=r"Ndarray shouldn't be passed in via"):
        func(arr_0, arr_1)


@pytest.mark.parametrize("shape", [2**31, 1.5, 0, (1, 0), (1, 0.5), (1, 2**31)])
@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_shape_invalid(shape):
    with pytest.raises(QuadrantsRuntimeError, match=r"is not a valid shape for ndarray"):
        x = qd.ndarray(dtype=int, shape=shape)


@pytest.mark.parametrize("shape", [1, np.int32(1), (1, np.int32(1), 4096)])
@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_shape_valid(shape):
    x = qd.ndarray(dtype=int, shape=shape)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_gaussian_kernel():
    M_PI = 3.14159265358979323846

    @qd.func
    def gaussian(x, sigma):
        return qd.exp(-0.5 * qd.pow(x / sigma, 2)) / (sigma * qd.sqrt(2.0 * M_PI))

    @qd.kernel
    def fill_gaussian_kernel(ker: qd.types.NDArray[qd.f32, 1], N: qd.i32):
        sum = 0.0
        for i in range(2 * N + 1):
            ker[i] = gaussian(i - N, qd.sqrt(N))
            sum += ker[i]
        for i in range(2 * N + 1):
            ker[i] = ker[i] / sum

    N = 4
    arr = qd.ndarray(dtype=qd.f32, shape=(20))
    fill_gaussian_kernel(arr, N)
    res = arr.to_numpy()

    np_arr = np.zeros(20, dtype=np.float32)
    fill_gaussian_kernel(np_arr, N)

    assert test_utils.allclose(res, np_arr)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_numpy_matrix():
    boundary_box_np = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    boundary_box = qd.Vector.ndarray(3, qd.f32, shape=2)
    boundary_box.from_numpy(boundary_box_np)
    ref_numpy = boundary_box.to_numpy()

    assert (boundary_box_np == ref_numpy).all()


@pytest.mark.parametrize("dtype", [qd.i64, qd.u64, qd.f64])
@test_utils.test(arch=supported_archs_quadrants_ndarray, require=qd.extension.data64)
def test_ndarray_python_scope_read_64bit(dtype):
    @qd.kernel
    def run(x: qd.types.NDArray[dtype, 1]):
        for i in x:
            x[i] = i + qd.i64(2**40)

    n = 4
    a = qd.ndarray(dtype, shape=(n,))
    run(a)
    for i in range(n):
        assert a[i] == i + 2**40


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_init_as_zero():
    a = qd.ndarray(dtype=qd.f32, shape=(6, 10))
    v = np.zeros((6, 10), dtype=np.float32)
    assert test_utils.allclose(a.to_numpy(), v)

    b = qd.ndarray(dtype=qd.math.vec2, shape=(6, 4))
    k = np.zeros((6, 4, 2), dtype=np.float32)
    assert test_utils.allclose(b.to_numpy(), k)

    c = qd.ndarray(dtype=qd.math.mat2, shape=(6, 4))
    m = np.zeros((6, 4, 2, 2), dtype=np.float32)
    assert test_utils.allclose(c.to_numpy(), m)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_zero_fill():
    dt = qd.types.vector(n=2, dtype=qd.f32)
    arr = qd.ndarray(dtype=dt, shape=(3, 4))

    arr.fill(1.0)

    arr.to_numpy()
    no = qd.ndarray(dtype=dt, shape=(3, 5))
    assert no[0, 0][0] == 0.0


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_reset():
    n = 8
    c = qd.Matrix.ndarray(4, 4, qd.f32, shape=(n))
    del c
    d = qd.Matrix.ndarray(4, 4, qd.f32, shape=(n))
    qd.reset()


@pytest.mark.run_in_serial
@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_in_python_func():
    def test():
        z = qd.ndarray(float, (8192, 8192))

    for i in range(300):
        test()


@test_utils.test(arch=[qd.cpu, qd.cuda], exclude=[qd.amdgpu])
def test_ndarray_with_fp16():
    half2 = qd.types.vector(n=2, dtype=qd.f16)

    @qd.kernel
    def init(x: qd.types.NDArray[half2, 1]):
        for i in x:
            x[i] = half2(2.0)

    @qd.kernel
    def test(table: qd.types.NDArray[half2, 1]):
        tmp = qd.Vector([qd.f16(0.0), qd.f16(0.0)])
        for i in qd.static(range(2)):
            tmp = tmp + 4.0 * table[i]

        table[0] = tmp

    acc = qd.ndarray(dtype=half2, shape=(40))
    table = qd.ndarray(dtype=half2, shape=(40))

    init(table)
    test(table)

    assert (table.to_numpy()[0] == 16.0).all()


@test_utils.test(
    arch=supported_archs_quadrants_ndarray,
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_scalar_ndarray_oob():
    @qd.kernel
    def access_arr(input: qd.types.NDArray, x: qd.i32) -> qd.f32:
        return input[x]

    input = np.random.randn(4)

    # Works
    access_arr(input, 1)

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 4)

    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, -1)


# SOA layout for ndarray is deprecated so no need to test
@test_utils.test(
    arch=supported_archs_quadrants_ndarray,
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
# TODO: investigate why this crashes sometimes on Windows
@pytest.mark.skipif(sys.platform == "win32", reason="Crashes frequently on windows")
def test_matrix_ndarray_oob():
    @qd.kernel
    def access_arr(input: qd.types.NDArray[qd.math.mat2, 2], p: qd.i32, q: qd.i32, x: qd.i32, y: qd.i32) -> qd.f32:
        return input[p, q][x, y]

    @qd.kernel
    def valid_access(indices: qd.types.NDArray[ivec3, 1], dummy: qd.types.NDArray[ivec3, 1]):
        for i in indices:
            index_vec = qd.Vector([0, 0, 0])
            for j in qd.static(range(3)):
                index = indices[i][j]
                index_vec[j] = index
            dummy[i] = index_vec

    input = qd.ndarray(dtype=qd.math.mat2, shape=(4, 5))

    indices = qd.ndarray(dtype=ivec3, shape=(10))
    dummy = qd.ndarray(dtype=ivec3, shape=(10))

    # Works
    access_arr(input, 2, 3, 0, 1)
    valid_access(indices, dummy)

    # element_shape
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 2, 3, 2, 1)
    # field_shape[0]
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 4, 4, 0, 1)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, -3, 4, 1, 1)
    # field_shape[1]
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 3, 5, 0, 1)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_arr(input, 2, -10, 1, 1)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_mismatched_index_python_scope():
    x = qd.ndarray(dtype=qd.f32, shape=(4, 4))
    with pytest.raises(QuadrantsIndexError, match=r"2d ndarray indexed with 1d indices"):
        x[0]

    with pytest.raises(QuadrantsIndexError, match=r"2d ndarray indexed with 3d indices"):
        x[0, 0, 0]


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_0dim_ndarray_read_write_python_scope():
    x = qd.ndarray(dtype=qd.f32, shape=())

    x[()] = 1.0
    assert x[None] == 1.0

    y = qd.ndarray(dtype=qd.math.vec2, shape=())
    y[()] = [1.0, 2.0]
    assert y[None] == [1.0, 2.0]


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_0dim_ndarray_read_write_quadrants_scope():
    x = qd.ndarray(dtype=qd.f32, shape=())

    @qd.kernel
    def write(x: qd.types.NDArray):
        a = x[()] + 1
        x[None] = 2 * a

    write(x)
    assert x[None] == 2.0

    y = qd.ndarray(dtype=qd.math.vec2, shape=())
    write(y)
    assert y[None] == [2.0, 2.0]


@test_utils.test(arch=supported_archs_quadrants_ndarray, require=qd.extension.data64)
def test_read_write_f64_python_scope():
    x = qd.ndarray(dtype=qd.f64, shape=2)

    x[0] = 1.0
    assert x[0] == 1.0

    y = qd.ndarray(dtype=qd.math.vec2, shape=2)
    y[0] = [1.0, 2.0]
    assert y[0] == [1.0, 2.0]


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_fill():
    vec2 = qd.types.vector(2, qd.f32)
    x_vec = qd.ndarray(vec2, (512, 512))
    x_vec.fill(1.0)
    assert (x_vec[2, 2] == [1.0, 1.0]).all()

    x_vec.fill(vec2(2.0, 4.0))
    assert (x_vec[3, 3] == [2.0, 4.0]).all()

    mat2x2 = qd.types.matrix(2, 2, qd.f32)
    x_mat = qd.ndarray(mat2x2, (512, 512))
    x_mat.fill(2.0)
    assert (x_mat[2, 2] == [[2.0, 2.0], [2.0, 2.0]]).all()

    x_mat.fill(mat2x2([[2.0, 4.0], [1.0, 3.0]]))
    assert (x_mat[3, 3] == [[2.0, 4.0], [1.0, 3.0]]).all()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_wrong_dtype():
    @qd.kernel
    def test2(arr: qd.types.NDArray[qd.f32, 2]):
        for I in qd.grouped(arr):
            arr[I] = 2.0

    tp_ivec3 = qd.types.vector(3, qd.i32)

    y = qd.ndarray(tp_ivec3, shape=(12, 4))
    with pytest.raises(TypeError, match=r"get \[Tensor \(3\) i32\]"):
        test2(y)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_bad_assign():
    tp_ivec3 = qd.types.vector(3, qd.i32)

    @qd.kernel
    def test4(arr: qd.types.NDArray[tp_ivec3, 2]):
        for I in qd.grouped(arr):
            arr[I] = [1, 2]

    y = qd.ndarray(tp_ivec3, shape=(12, 4))
    with pytest.raises(QuadrantsTypeError, match=r"cannot assign '\[Tensor \(2\) i32\]' to '\[Tensor \(3\) i32\]'"):
        test4(y)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_bad_ndim():
    x = qd.ndarray(qd.f32, shape=(12, 13))

    @qd.kernel
    def test5(arr: qd.types.NDArray[qd.f32, 1]):
        for i, j in arr:
            arr[i, j] = 0

    with pytest.raises(ValueError, match=r"required ndim=1, but 2d ndarray with shape \(12, 13\) is provided"):
        test5(x)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_type_hint_matrix():
    @qd.kernel
    def test(x: qd.types.NDArray[qd.types.matrix(), 1]):
        for I in qd.grouped(x):
            x[I] = 1.0

    x = qd.ndarray(qd.math.mat2, (3))
    test(x)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    y = qd.ndarray(qd.math.mat3, (3))
    test(y)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    z = qd.ndarray(qd.math.vec2, (3))
    with pytest.raises(ValueError, match=r"Invalid value for argument x"):
        test(z)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_type_hint_vector():
    @qd.kernel
    def test(x: qd.types.NDArray[qd.types.vector(), 1]):
        for I in qd.grouped(x):
            x[I] = 1.0

    x = qd.ndarray(qd.math.vec3, (3))
    test(x)
    assert impl.get_runtime().get_num_compiled_functions() == 1

    y = qd.ndarray(qd.math.vec2, (3))
    test(y)
    assert impl.get_runtime().get_num_compiled_functions() == 2

    z = qd.ndarray(qd.math.mat2, (3))
    with pytest.raises(ValueError, match=r"Invalid value for argument x"):
        test(z)


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_pass_ndarray_to_func():
    @qd.func
    def bar(weight: qd.types.NDArray[qd.f32, 3]) -> qd.f32:
        return weight[1, 1, 1]

    @qd.kernel
    def foo(weight: qd.types.NDArray[qd.f32, 3]) -> qd.f32:
        return bar(weight)

    weight = qd.ndarray(dtype=qd.f32, shape=(2, 2, 2))
    weight.fill(42.0)
    assert foo(weight) == 42.0


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_pass_ndarray_to_real_func():
    @qd.real_func
    def bar(weight: qd.types.NDArray[qd.f32, 3]) -> qd.f32:
        return weight[1, 1, 1]

    @qd.kernel
    def foo(weight: qd.types.NDArray[qd.f32, 3]) -> qd.f32:
        return bar(weight)

    weight = qd.ndarray(dtype=qd.f32, shape=(2, 2, 2))
    weight.fill(42.0)
    assert foo(weight) == 42.0


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_pass_ndarray_outside_kernel_to_real_func():
    weight = qd.ndarray(dtype=qd.f32, shape=(2, 2, 2))

    @qd.real_func
    def bar(weight: qd.types.NDArray[qd.f32, 3]) -> qd.f32:
        return weight[1, 1, 1]

    @qd.kernel
    def foo() -> qd.f32:
        return bar(weight)

    weight.fill(42.0)
    with pytest.raises(qd.QuadrantsTypeError, match=r"Expected ndarray in the kernel argument for argument weight"):
        foo()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_oob_clamp():
    @qd.kernel
    def test(x: qd.types.ndarray(boundary="clamp"), y: qd.i32) -> qd.f32:
        return x[y]

    x = qd.ndarray(qd.f32, shape=(3))
    for i in range(3):
        x[i] = i

    assert test(x, -1) == 0
    assert test(x, -2) == 0
    assert test(x, 3) == 2
    assert test(x, 4) == 2

    @qd.kernel
    def test_vec_arr(x: qd.types.ndarray(boundary="clamp"), y: qd.i32) -> qd.f32:
        return x[1, 2][y]

    x2 = qd.ndarray(qd.math.vec2, shape=(3, 3))
    for i in range(3):
        for j in range(3):
            x2[i, j] = [i, j]
    assert test_vec_arr(x2, -1) == 1
    assert test_vec_arr(x2, 2) == 2

    @qd.kernel
    def test_mat_arr(x: qd.types.ndarray(boundary="clamp"), i: qd.i32, j: qd.i32) -> qd.f32:
        return x[1, 2][i, j]

    x3 = qd.ndarray(qd.math.mat2, shape=(3, 3))
    for i in range(3):
        for j in range(3):
            x3[i, j] = [[i, j], [i + 1, j + 1]]
    assert test_mat_arr(x3, -1, 0) == 1
    assert test_mat_arr(x3, 1, -1) == 2
    assert test_mat_arr(x3, 2, 0) == 3
    assert test_mat_arr(x3, 1, 2) == 3


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_clamp_verify():
    height = 3
    width = 3

    @qd.kernel
    def test(ao: qd.types.ndarray(dtype=qd.f32, ndim=2, boundary="clamp")):
        for y, x in qd.ndrange(height, width):
            vis = 0.0
            ao[y, x] = vis

    ao = qd.ndarray(qd.f32, shape=(height, width))
    test(ao)
    assert (ao.to_numpy() == np.zeros((height, width))).all()


@test_utils.test(arch=supported_archs_quadrants_ndarray)
def test_ndarray_arg_builtin_float_type():
    @qd.kernel
    def foo(x: qd.types.NDArray[float, 0]) -> qd.f32:
        return x[None]

    x = qd.ndarray(qd.f32, shape=())
    x[None] = 42
    assert foo(x) == 42


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_vector_ndarray_arg():
    @qd.real_func
    def foo(x: qd.types.NDArray[vec3, 1]) -> vec3:
        return x[0]

    @qd.kernel
    def test(x: qd.types.NDArray[vec3, 1]) -> vec3:
        return foo(x)

    x = qd.Vector.ndarray(3, qd.f32, shape=(1))
    x[0] = vec3(1, 2, 3)
    assert (test(x) == vec3(1, 2, 3)).all()


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_real_func_write_ndarray_cfg():
    @qd.real_func
    def bar(a: qd.types.NDArray[qd.types.vector(3, float), 1]):
        a[0] = vec3(1)

    @qd.kernel
    def foo(
        a: qd.types.NDArray[qd.types.vector(3, float), 1],
    ):
        a[0] = vec3(3)
        bar(a)
        a[0] = vec3(3)

    a = qd.Vector.ndarray(3, float, shape=(2,))
    foo(a)
    assert (a[0] == vec3(3)).all()


# exclude metal, because metal limited to < 30 parametrs AFAIK
@test_utils.test(exclude=[qd.metal])
def test_ndarray_max_num_args() -> None:
    if platform.system() == "Darwin" and qd.lang.impl.current_cfg().arch == qd.vulkan:
        pytest.skip(reason="Mac doesn't support so many arguments, on Vulkan")

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
        args_l.append(f"a{i}: qd.types.NDArray[qd.i32, 1]")
        arg_uses_l.append(f"    a{i}[0] += {i + 1}")
        arg_objs_l.append(qd.ndarray(qd.i32, (10,)))
    args_str = ", ".join(args_l)
    arg_uses_str = "\n".join(arg_uses_l)
    kernel_str = kernel_templ.format(args=args_str, arg_uses=arg_uses_str)
    with load_kernel_from_string(kernel_str, "my_kernel") as my_kernel:
        my_kernel(*arg_objs_l)
    for i in range(num_args):
        assert arg_objs_l[i][0] == i + 1


@pytest.mark.parametrize("dtype", [qd.i32, qd.types.vector(3, qd.f32), qd.types.matrix(2, 2, qd.f32)])
@test_utils.test()
def test_ndarray_del(dtype) -> None:
    def foo():
        nd = qd.ndarray(dtype, (1000,))
        assert qd.lang.impl.get_runtime().prog._get_num_ndarrays() == 1

    foo()
    assert qd.lang.impl.get_runtime().prog._get_num_ndarrays() == 0
