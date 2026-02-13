# This is a test file. It just has to exist, to check that pyright works with it.

import quadrants as ti

from tests import test_utils

ti.init(arch=ti.cpu)


@ti.kernel
def k1(a: ti.types.ndarray(), b: ti.types.NDArray, c: ti.types.NDArray[ti.i32, 1]) -> None: ...


@ti.kernel()
def k2(a: ti.types.ndarray(), b: ti.types.NDArray, c: ti.types.NDArray[ti.i32, 1]) -> None: ...


@ti.data_oriented
class SomeClass:
    @ti.kernel
    def k1(self, a: ti.types.ndarray(), b: ti.types.NDArray, c: ti.types.NDArray[ti.i32, 1]) -> None: ...

    @ti.kernel()
    def k2(self, a: ti.types.ndarray(), b: ti.types.NDArray, c: ti.types.NDArray[ti.i32, 1]) -> None: ...


@test_utils.test()
def test_ndarray_type():
    a = ti.ndarray(ti.i32, (10,))
    k1(a, a, a)
    k2(a, a, a)

    some_class = SomeClass()
    some_class.k1(a, a, a)
    some_class.k2(a, a, a)
