import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_to_numpy_2d():
    val = qd.field(qd.i32)

    n = 4
    m = 7

    qd.root.dense(qd.ij, (n, m)).place(val)

    for i in range(n):
        for j in range(m):
            val[i, j] = i + j * 3

    arr = val.to_numpy()

    assert arr.shape == (4, 7)
    for i in range(n):
        for j in range(m):
            assert arr[i, j] == i + j * 3


@test_utils.test()
def test_from_numpy_2d():
    val = qd.field(qd.i32)

    n = 4
    m = 7

    qd.root.dense(qd.ij, (n, m)).place(val)

    arr = np.empty(shape=(n, m), dtype=np.int32)

    for i in range(n):
        for j in range(m):
            arr[i, j] = i + j * 3

    val.from_numpy(arr)

    for i in range(n):
        for j in range(m):
            assert val[i, j] == i + j * 3


@test_utils.test()
def test_to_numpy_struct():
    n = 16
    f = qd.Struct.field({"a": qd.i32, "b": qd.f32}, shape=(n,))

    for i in range(n):
        f[i].a = i
        f[i].b = f[i].a * 2

    arr_dict = f.to_numpy()

    for i in range(n):
        assert arr_dict["a"][i] == i
        assert arr_dict["b"][i] == i * 2


@test_utils.test()
def test_from_numpy_struct():
    n = 16
    f = qd.Struct.field({"a": qd.i32, "b": qd.f32}, shape=(n,))

    arr_dict = {
        "a": np.arange(n, dtype=np.int32),
        "b": np.arange(n, dtype=np.int32) * 2,
    }

    f.from_numpy(arr_dict)

    for i in range(n):
        assert f[i].a == i
        assert f[i].b == i * 2


@test_utils.test(require=qd.extension.data64)
def test_f64():
    val = qd.field(qd.f64)

    n = 4
    m = 7

    qd.root.dense(qd.ij, (n, m)).place(val)

    for i in range(n):
        for j in range(m):
            val[i, j] = (i + j * 3) * 1e100

    val.from_numpy(val.to_numpy() * 2)

    for i in range(n):
        for j in range(m):
            assert val[i, j] == (i + j * 3) * 2e100


@test_utils.test()
def test_matrix():
    n = 4
    m = 7
    val = qd.Matrix.field(2, 3, qd.f32, shape=(n, m))

    nparr = np.empty(shape=(n, m, 2, 3), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            for k in range(2):
                for l in range(3):
                    nparr[i, j, k, l] = i + j * 2 - k - l * 3

    val.from_numpy(nparr)
    new_nparr = val.to_numpy()
    assert (nparr == new_nparr).all()


@test_utils.test()
def test_numpy_io_example():
    n = 4
    m = 7

    # Quadrants tensors
    val = qd.field(qd.i32, shape=(n, m))
    vec = qd.Vector.field(3, dtype=qd.i32, shape=(n, m))
    mat = qd.Matrix.field(3, 4, dtype=qd.i32, shape=(n, m))

    # Scalar
    arr = np.ones(shape=(n, m), dtype=np.int32)
    val.from_numpy(arr)
    arr = val.to_numpy()

    # Vector
    arr = np.ones(shape=(n, m, 3), dtype=np.int32)
    vec.from_numpy(arr)

    arr = np.ones(shape=(n, m, 3, 1), dtype=np.int32)
    vec.from_numpy(arr)

    arr = np.ones(shape=(n, m, 1, 3), dtype=np.int32)
    vec.from_numpy(arr)

    arr = vec.to_numpy()
    assert arr.shape == (n, m, 3)

    arr = vec.to_numpy(keep_dims=True)
    assert arr.shape == (n, m, 3, 1)

    # Matrix
    arr = np.ones(shape=(n, m, 3, 4), dtype=np.int32)
    mat.from_numpy(arr)

    arr = mat.to_numpy()
    assert arr.shape == (n, m, 3, 4)

    arr = mat.to_numpy(keep_dims=True)
    assert arr.shape == (n, m, 3, 4)

    # For PyTorch tensors, use to_torch/from_torch instead


@test_utils.test()
def test_from_numpy_non_contiguous():
    n = 9
    m = 7
    p = 4
    arr = np.ones(shape=(n, m, p, p), dtype=np.int32)

    val = qd.field(qd.i32, shape=(2, 2))
    val.from_numpy(arr[0:6:3, 0:6:3, 0, 0])

    vec = qd.Vector.field(3, dtype=qd.i32, shape=(2, 2))
    vec.from_numpy(arr[0:6:3, 0:6:3, 0:3, 0])

    mat = qd.Matrix.field(3, 4, dtype=qd.i32, shape=(2, 2))
    mat.from_numpy(arr[0:6:3, 0:6:3, 0:3, 0:4])
