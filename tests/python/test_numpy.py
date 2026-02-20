import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


def with_data_type(dt):
    val = qd.field(qd.i32)

    n = 4

    qd.root.dense(qd.i, n).place(val)

    @qd.kernel
    def test_numpy(arr: qd.types.ndarray()):
        for i in range(n):
            arr[i] = arr[i] ** 2

    a = np.array([4, 8, 1, 24], dtype=dt)

    for i in range(n):
        a[i] = i * 2

    test_numpy(a)

    for i in range(n):
        assert a[i] == i * i * 4


@test_utils.test()
def test_numpy_f32():
    with_data_type(np.float32)


@test_utils.test(require=qd.extension.data64)
def test_numpy_f64():
    with_data_type(np.float64)


@test_utils.test(arch=qd.metal)
def test_np_i64_metal():
    @qd.kernel
    def arange(x: qd.types.ndarray(qd.i64, ndim=1)):
        for i in x:
            x[i] = i

    xx = np.array([1, 2, 3, 4, 5])  # by default it's int64
    arange(xx)


@test_utils.test()
def test_numpy_i32():
    with_data_type(np.int32)


@test_utils.test(require=qd.extension.data64)
def test_numpy_i64():
    with_data_type(np.int64)


@test_utils.test()
def test_numpy_2d():
    val = qd.field(qd.i32)

    n = 4
    m = 7

    qd.root.dense(qd.i, n).dense(qd.j, m).place(val)

    @qd.kernel
    def test_numpy(arr: qd.types.ndarray()):
        for i in range(n):
            for j in range(m):
                arr[i, j] += i + j

    a = np.empty(shape=(n, m), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            a[i, j] = i * j

    test_numpy(a)

    for i in range(n):
        for j in range(m):
            assert a[i, j] == i * j + i + j


@test_utils.test()
def test_numpy_2d_transpose():
    val = qd.field(qd.i32)

    n = 8
    m = 8

    qd.root.dense(qd.ij, (n, m)).place(val)

    @qd.kernel
    def test_numpy(arr: qd.types.ndarray()):
        for i in qd.grouped(val):
            val[i] = arr[i]
            arr[i] = 1

    a = np.empty(shape=(n, m), dtype=np.int32)
    b = a.transpose()

    for i in range(n):
        for j in range(m):
            a[i, j] = i * j + i * 4

    test_numpy(b)

    for i in range(n):
        for j in range(m):
            assert val[i, j] == i * j + j * 4
            assert a[i][j] == 1


@test_utils.test()
def test_numpy_3d():
    val = qd.field(qd.i32)

    n = 4
    m = 7
    p = 11

    qd.root.dense(qd.i, n).dense(qd.j, m).dense(qd.k, p).place(val)

    @qd.kernel
    def test_numpy(arr: qd.types.ndarray()):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    arr[i, j, k] += i + j + k * 2

    a = np.empty(shape=(n, m, p), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                a[i, j, k] = i * j * (k + 1)

    test_numpy(a)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                assert a[i, j, k] == i * j * (k + 1) + i + j + k * 2


@test_utils.test(print_full_traceback=False)
def test_numpy_3d_error():
    val = qd.field(qd.i32)

    n = 4
    m = 7
    p = 11

    qd.root.dense(qd.i, n).dense(qd.j, m).dense(qd.k, p).place(val)

    @qd.kernel
    def test_numpy(arr: qd.types.ndarray()):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    arr[i, j] += i + j + k * 2

    a = np.empty(shape=(n, m, p), dtype=np.int32)

    with pytest.raises(qd.QuadrantsCompilationError):
        test_numpy(a)


@test_utils.test()
def test_numpy_multiple_external_arrays():
    n = 4

    @qd.kernel
    def test_numpy(a: qd.types.ndarray(), b: qd.types.ndarray()):
        for i in range(n):
            a[i] = a[i] * b[i]
            b[i] = a[i] + b[i]

    a = np.array([4, 8, 1, 24], dtype=np.int32)
    b = np.array([5, 6, 12, 3], dtype=np.int32)
    c = a * b
    d = c + b

    test_numpy(a, b)
    for i in range(n):
        assert a[i] == c[i]
        assert b[i] == d[i]


@test_utils.test()
def test_numpy_zero():
    @qd.kernel
    def test_numpy(arr: qd.types.ndarray()):
        pass

    test_numpy(np.empty(shape=(0), dtype=np.int32))
    test_numpy(np.empty(shape=(0, 5), dtype=np.int32))
    test_numpy(np.empty(shape=(5, 0), dtype=np.int32))


@test_utils.test()
def test_numpy_struct_for():
    @qd.kernel
    def func1(a: qd.types.ndarray()):
        for i, j in a:
            a[i, j] = i + j

    m = np.zeros((123, 456), dtype=np.int32)
    func1(m)
    for i in range(123):
        for j in range(456):
            assert m[i, j] == i + j

    @qd.kernel
    def func2(a: qd.types.ndarray()):
        for I in qd.grouped(a):
            a[I] = I.sum()

    n = np.zeros((98, 76, 54), dtype=np.int32)
    func2(n)
    for i, j, k in qd.ndrange(98, 76, 54):
        assert n[i, j, k] == i + j + k


@test_utils.test(debug=True)
def test_numpy_op_with_matrix():
    scalar = np.cos(0)
    vec = qd.Vector([1, 2])
    assert isinstance(scalar + vec, qd.Matrix) and isinstance(vec + scalar, qd.Matrix)

    @qd.kernel
    def test():
        x = scalar + vec
        assert all(x == [2.0, 3.0])
        x = vec + scalar
        assert all(x == [2.0, 3.0])
        y = scalar / vec
        assert all(y == [1.0, 0.5])
        y = vec / scalar
        assert all(y == [1.0, 2.0])

    test()


@test_utils.test(debug=True)
def test_numpy_with_matrix():
    x = qd.Vector.field(3, dtype=qd.f32, shape=())
    a = np.array([1, 2, 3], dtype=np.float32)
    b = qd.Vector([0, a[2], 0], dt=qd.f32)

    @qd.kernel
    def bar():
        foo()

    @qd.func
    def foo():
        x[None] = qd.max(x[None], b)

    bar()
    assert (x.to_numpy() == [0.0, 3.0, 0.0]).all()


@test_utils.test()
def test_numpy_view():
    @qd.kernel
    def fill(img: qd.types.ndarray()):
        img[0] = 1

    a = np.zeros(shape=(2, 2))[:, 0]
    with pytest.raises(ValueError, match="Non contiguous numpy arrays are not supported"):
        fill(a)


@test_utils.test()
def test_numpy_ndarray_dim_check():
    @qd.kernel
    def add_one_mat(arr: qd.types.ndarray(dtype=qd.math.mat3, ndim=2)):
        for i in qd.grouped(arr):
            arr[i] = arr[i] + 1.0

    @qd.kernel
    def add_one_scalar(arr: qd.types.ndarray(dtype=qd.f32, ndim=2)):
        for i in qd.grouped(arr):
            arr[i] = arr[i] + 1.0

    a = np.zeros(shape=(2, 2, 3, 3), dtype=np.float32)
    b = np.zeros(shape=(2, 2, 2, 3), dtype=np.float32)
    c = np.zeros(shape=(2, 2, 3), dtype=np.float32)
    d = np.zeros(shape=(2, 2), dtype=np.float32)
    add_one_mat(a)
    add_one_scalar(d)
    np.testing.assert_allclose(a, np.ones(shape=(2, 2, 3, 3), dtype=np.float32))
    np.testing.assert_allclose(d, np.ones(shape=(2, 2), dtype=np.float32))
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument arr - required element_shape=\(.*\), array with element shape of \(.*\)",
    ):
        add_one_mat(b)
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument arr - required array has ndim=2 element_dim=2, array with 3 dimensions is provided",
    ):
        add_one_mat(c)
    with pytest.raises(
        ValueError,
        match=r"Invalid value for argument arr - required array has ndim=2, array with 4 dimensions is provided",
    ):
        add_one_scalar(a)


@test_utils.test()
def test_numpy_dtype_mismatch():
    @qd.kernel
    def arange(x: qd.types.ndarray(qd.i32, ndim=1)):
        for i in x:
            x[i] = i

    xx = np.array([1, 2, 3, 4, 5], dtype=np.int64)  # by default it's int64
    with pytest.raises(ValueError, match=r"Invalid value for argument x - required array has dtype="):
        arange(xx)
