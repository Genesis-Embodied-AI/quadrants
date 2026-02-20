import quadrants as qd


# same
@qd.func
def f2() -> None:
    pass


# base


# same
@qd.kernel
def f1() -> None:
    f2()


# same
