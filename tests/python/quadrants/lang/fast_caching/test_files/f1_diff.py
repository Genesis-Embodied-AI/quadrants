import quadrants as qd


# diff
@qd.func
def f3() -> int:
    return 123


# diff


# diff
@qd.kernel
def f1() -> None:
    f3()


# diff
