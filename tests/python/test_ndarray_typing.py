import quadrants as ti

from tests import test_utils


@ti.kernel
def some_kernel(a: ti.types.NDArray[ti.i32, 2], b: ti.types.NDArray[ti.i32, 2]) -> None:
    for i, j in b:
        a[i, j] = b[i, j] + 2


@test_utils.test()
def test_ndarray_typing_square_brackets():
    a = ti.ndarray(dtype=int, shape=(2, 3))
    b = ti.ndarray(dtype=int, shape=(2, 3))
    b[1, 1] = 5
    some_kernel(a, b)
    assert a[1, 1] == 5 + 2
