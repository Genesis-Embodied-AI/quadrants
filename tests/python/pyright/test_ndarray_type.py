# This is a test file. It just has to exist, to check that pyright works with it.

import quadrants as qd

from tests import test_utils

qd.init(arch=qd.cpu)


@qd.kernel
def k1(a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...


@qd.kernel()
def k2(a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...


@qd.data_oriented
class SomeClass:
    @qd.kernel
    def k1(self, a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...

    @qd.kernel()
    def k2(self, a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...


@test_utils.test()
def test_ndarray_type():
    a = qd.ndarray(qd.i32, (10,))
    k1(a, a, a)
    k2(a, a, a)

    some_class = SomeClass()
    some_class.k1(a, a, a)
    some_class.k2(a, a, a)
