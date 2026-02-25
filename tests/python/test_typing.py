import test_utils

import quadrants as qd


@test_utils.test()
def test_typing_kernel_return_none():
    x = qd.field(qd.i32, shape=())

    @qd.kernel
    def some_kernel() -> None:
        x[None] += 1

    some_kernel()


@test_utils.test()
def test_typing_func_return_none():
    x = qd.field(qd.i32, shape=())

    @qd.func
    def some_func() -> None:
        x[None] += 1

    @qd.kernel
    def some_kernel() -> None:
        some_func()

    some_kernel()
