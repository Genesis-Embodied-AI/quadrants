import quadrants as qd

from tests import test_utils


@qd.kernel
def some_kernel(a: qd.types.NDArray[qd.i32, 2], b: qd.types.NDArray[qd.i32, 2]) -> None:
    for i, j in b:
        a[i, j] = b[i, j] + 2


@qd.kernel
def some_kernel_single_arg(a: qd.types.NDArray[qd.i32], b: qd.types.NDArray[qd.i32]) -> None:
    for i, j in b:
        a[i, j] = b[i, j] + 2


@test_utils.test()
def test_ndarray_typing_square_brackets():
    a = qd.ndarray(dtype=int, shape=(2, 3))
    b = qd.ndarray(dtype=int, shape=(2, 3))
    b[1, 1] = 5
    some_kernel(a, b)
    assert a[1, 1] == 5 + 2


def test_ndarray_typing_single_arg():
    t = qd.types.NDArray[qd.i32]
    assert t.dtype == qd.i32
    assert t.ndim is None


@test_utils.test()
def test_ndarray_typing_single_arg_kernel():
    a = qd.ndarray(dtype=int, shape=(2, 3))
    b = qd.ndarray(dtype=int, shape=(2, 3))
    b[1, 1] = 5
    some_kernel_single_arg(a, b)
    assert a[1, 1] == 5 + 2
