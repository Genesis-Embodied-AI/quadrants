import quadrants as ti


# same
@ti.func
def f2() -> None:
    pass


# same


# same
@ti.kernel
def f1() -> None:
    f2()


# same
