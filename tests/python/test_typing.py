import test_utils

import quadrants as ti


@test_utils.test()
def test_typing_kernel_return_none():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def some_kernel() -> None:
        x[None] += 1

    some_kernel()


@test_utils.test()
def test_typing_func_return_none():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def some_func() -> None:
        x[None] += 1

    @ti.kernel
    def some_kernel() -> None:
        some_func()

    some_kernel()
