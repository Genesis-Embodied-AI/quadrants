import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

_QD_DTYPE_TO_NP_DTYPE = {
    qd.u1: np.bool_,
    qd.u8: np.uint8,
    qd.u16: np.uint16,
    qd.u32: np.uint32,
    qd.u64: np.uint64,
    qd.i8: np.int8,
    qd.i16: np.int16,
    qd.i32: np.int32,
    qd.i64: np.int64,
}


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray])
@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@test_utils.test()
def test_tensor_consistency_kernel_write_to_numpy_consistency(tensor_type, dtype) -> None:
    """
    write from kernel => to_numpy => check
    """
    poses = [0, 2, 5, 11]
    a = tensor_type(dtype, (16,))

    TensorType = qd.types.NDArray if tensor_type == qd.ndarray else qd.Template

    @qd.kernel
    def k1(a: TensorType) -> None:
        for b_ in range(1):
            for pos in qd.static(poses):
                a[pos] = 1

    k1(a)

    a_np = a.to_numpy()

    for i in range(16):
        assert a_np[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray])
@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@test_utils.test()
def test_tensor_consistency_kernel_from_numpy_to_numpy_consistency(tensor_type, dtype) -> None:
    """
    write to numpy => from_numpy => to_numpy => check
    """
    poses = [0, 2, 5, 11]

    np_dtype = _QD_DTYPE_TO_NP_DTYPE[dtype]

    a_np = np.zeros(dtype=np_dtype, shape=(16,))

    for pos in poses:
        a_np[pos] = 1

    a = tensor_type(dtype, (16,))
    a.from_numpy(a_np)

    b_np = a.to_numpy()

    for i in range(16):
        assert b_np[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray])
@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@test_utils.test()
def test_tensor_consistency_np_read_write_np_consistency(tensor_type, dtype) -> None:
    """
    write to numpy => read from kernel => write from kernel => numpy => check
    check consistency
    """
    poses = [0, 2, 5, 11]

    np_dtype = _QD_DTYPE_TO_NP_DTYPE[dtype]

    a_np = np.zeros(dtype=np_dtype, shape=(16,))
    a = tensor_type(dtype, (16,))
    b = tensor_type(dtype, (16,))

    for pos in poses:
        a_np[pos] = 1
    a.from_numpy(a_np)

    TensorType = qd.types.NDArray if tensor_type == qd.ndarray else qd.Template

    @qd.kernel
    def k1(a: TensorType, b: TensorType) -> None:
        for b_ in range(1):
            for pos in qd.static(poses):
                b[pos] = a[pos]

    k1(a, b)

    b_np = b.to_numpy()

    for i in range(16):
        assert b_np[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray])
@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@test_utils.test()
def test_tensor_consistency_from_numpy_accessor_read_consistency(tensor_type, dtype) -> None:
    """
    write to numpy => from_numpy => accessor read => check
    check consistency
    """
    poses = [0, 2, 5, 11]

    np_dtype = _QD_DTYPE_TO_NP_DTYPE[dtype]
    a_np = np.zeros(dtype=np_dtype, shape=(16,))
    a = tensor_type(dtype, (16,))

    for pos in poses:
        a_np[pos] = 1
    a.from_numpy(a_np)

    for i in range(16):
        assert a[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray])
@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@test_utils.test()
def test_tensor_consistency_accessor_write_to_numpy_consistency(tensor_type, dtype) -> None:
    """
    accessor write => to_numpy => check
    """
    poses = [0, 2, 5, 11]

    a = tensor_type(dtype, (16,))
    for pos in poses:
        a[pos] = 1

    a_np = a.to_numpy()

    for i in range(16):
        assert a_np[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray])
@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@pytest.mark.parametrize("std_dtype", [qd.i8, qd.i32])
@test_utils.test()
def test_tensor_consistency_from_numpy_kern_read(tensor_type, dtype, std_dtype) -> None:
    """
    write numpy => from_numpy => kernel read => kernel write to standard type => to_numpy => check
    """
    poses = [0, 2, 5, 11]
    N = 16

    np_dtype = _QD_DTYPE_TO_NP_DTYPE[dtype]
    a_np = np.zeros(dtype=np_dtype, shape=(N,))
    a = tensor_type(dtype, (N,))
    b = tensor_type(std_dtype, (N,))

    for pos in poses:
        a_np[pos] = 1
    a.from_numpy(a_np)

    TensorType = qd.types.NDArray if tensor_type == qd.ndarray else qd.Template

    @qd.kernel
    def k1(a: TensorType, b: TensorType) -> None:
        for b_ in range(1):
            for i in range(N):
                b[i] = a[i]

    k1(a, b)

    b_np = b.to_numpy()

    for i in range(N):
        assert b_np[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("tensor_type", [qd.field, qd.ndarray])
@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@pytest.mark.parametrize("std_dtype", [qd.i8, qd.i32])
@test_utils.test()
def test_tensor_consistency_kern_write_to_numpy(tensor_type, dtype, std_dtype) -> None:
    """
    write to std type numpy => from_numpy => std type kernel read => kernel write => to_numpy => check
    """
    poses = [0, 2, 5, 11]
    N = 16

    np_dtype = _QD_DTYPE_TO_NP_DTYPE[dtype]
    a_np = np.zeros(dtype=np_dtype, shape=(N,))
    a = tensor_type(std_dtype, (N,))
    b = tensor_type(dtype, (N,))

    for pos in poses:
        a_np[pos] = 1
    a.from_numpy(a_np)

    TensorType = qd.types.NDArray if tensor_type == qd.ndarray else qd.Template

    @qd.kernel
    def k1(a: TensorType, b: TensorType) -> None:
        for b_ in range(1):
            for i in range(N):
                b[i] = a[i]

    k1(a, b)

    b_np = b.to_numpy()

    for i in range(N):
        assert b_np[i] == (1 if i in poses else 0)


@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@test_utils.test()
def test_tensor_consistency_ext_to_kern(dtype) -> None:
    """
    write to numpy => pass directly to kernel => test in kern
    """
    poses_l = [0, 2, 5, 11]
    N = 16

    np_dtype = _QD_DTYPE_TO_NP_DTYPE[dtype]
    a_np = np.zeros(dtype=np_dtype, shape=(N,))
    anti_poses_l = list(range(N))  # i.e positions with zeros
    for pos in poses_l:
        a_np[pos] = 1
        anti_poses_l.remove(pos)

    result = qd.ndarray(qd.i32, ())
    result[()] = 1

    @qd.kernel
    def k1(a: qd.types.NDArray, result: qd.types.NDArray) -> None:
        for b_ in range(1):
            for pos in qd.static(poses_l):
                if a[pos] != 1:
                    result[()] = 0
            for pos in qd.static(anti_poses_l):
                if a[pos] != 0:
                    result[()] = 0

    k1(a_np, result)
    qd.sync()
    assert result[()] == 1


@pytest.mark.parametrize("dtype", [qd.u1, qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i32, qd.i16, qd.i64])
@test_utils.test()
def test_tensor_consistency_kern_to_ext(dtype) -> None:
    """
    write directly to numpy array in kernel => check in numpy
    """

    poses_l = [0, 2, 5, 11]
    N = 16

    np_dtype = _QD_DTYPE_TO_NP_DTYPE[dtype]
    a_np = np.zeros(dtype=np_dtype, shape=(N,))

    @qd.kernel
    def k1(a: qd.types.NDArray) -> None:
        for b_ in range(1):
            for pos in qd.static(poses_l):
                a[pos] = 1

    k1(a_np)

    qd.sync()

    for i in range(N):
        assert a_np[i] == (1 if i in poses_l else 0)
