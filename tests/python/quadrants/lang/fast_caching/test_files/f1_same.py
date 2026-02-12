import quadrants as ti


# same
@ti.func
def f2() -> None:
    pass


# base


# same
@ti.kernel
def f1() -> None:
    f2()


# same
