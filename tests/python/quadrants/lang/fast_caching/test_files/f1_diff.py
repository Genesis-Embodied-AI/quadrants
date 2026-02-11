import quadrants as ti


# diff
@ti.func
def f3() -> int:
    return 123


# diff


# diff
@ti.kernel
def f1() -> None:
    f3()


# diff
