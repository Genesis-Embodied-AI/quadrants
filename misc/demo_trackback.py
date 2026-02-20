import quadrants as qd

qd.init()


@qd.func
def func3():
    qd.static_assert(1 + 1 == 3)


@qd.func
def func2():
    func3()


@qd.func
def func1():
    func2()


@qd.kernel
def func0():
    func1()


func0()
