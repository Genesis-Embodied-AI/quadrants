# Pyright test: NDArray annotations accepted by the type checker and functional at runtime.

import quadrants as qd

from tests import test_utils

qd.init(arch=qd.cpu)


# Legacy call syntax (still works, pyright warns about call expressions in type positions)
@qd.kernel
def k1(a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...  # type: ignore[reportInvalidTypeForm]


@qd.kernel()
def k2(a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...  # type: ignore[reportInvalidTypeForm]


# New subscript syntax (preferred, no pyright warnings)
@qd.kernel
def k3(a: qd.types.NDArray[qd.i32, 1], b: qd.types.NDArray[qd.i32], c: qd.types.NDArray) -> None: ...


@qd.data_oriented
class SomeClass:
    @qd.kernel
    def k1(self, a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...  # type: ignore[reportInvalidTypeForm]

    @qd.kernel()
    def k2(self, a: qd.types.ndarray(), b: qd.types.NDArray, c: qd.types.NDArray[qd.i32, 1]) -> None: ...  # type: ignore[reportInvalidTypeForm]

    @qd.kernel
    def k3(self, a: qd.types.NDArray[qd.i32, 1], b: qd.types.NDArray[qd.i32], c: qd.types.NDArray) -> None: ...


@test_utils.test()
def test_ndarray_type():
    a = qd.ndarray(qd.i32, (10,))
    k1(a, a, a)
    k2(a, a, a)
    k3(a, a, a)

    some_class = SomeClass()
    some_class.k1(a, a, a)
    some_class.k2(a, a, a)
    some_class.k3(a, a, a)
