import quadrants as qd

from tests import test_utils


@test_utils.test(debug=True, short_circuit_operators=True)
def test_and_shorted():
    a = qd.field(qd.i32, shape=10)

    @qd.func
    def explode() -> qd.u1:
        return qd.u1(a[-1])

    @qd.kernel
    def func() -> qd.u1:
        return False and explode()

    assert func() == False


@test_utils.test(debug=True, short_circuit_operators=True)
def test_and_not_shorted():
    @qd.kernel
    def func() -> qd.i32:
        return True and False

    assert func() == 0


@test_utils.test(debug=True, short_circuit_operators=True)
def test_or_shorted():
    a = qd.field(qd.i32, shape=10)

    @qd.func
    def explode() -> qd.u1:
        return qd.u1(a[-1])

    @qd.kernel
    def func() -> qd.i32:
        return True or explode()

    assert func() == 1


@test_utils.test(debug=True, short_circuit_operators=True)
def test_or_not_shorted():
    @qd.kernel
    def func() -> qd.u1:
        return False or True

    assert func() == 1


@test_utils.test(debug=True)
def test_static_or():
    @qd.kernel
    def func() -> qd.i32:
        return qd.static(0 or 3 or 5)

    assert func() == 3


@test_utils.test(debug=True)
def test_static_and():
    @qd.kernel
    def func() -> qd.i32:
        return qd.static(5 and 2 and 0)

    assert func() == 0


@test_utils.test(require=qd.extension.data64, default_ip=qd.i64)
def test_condition_type():
    @qd.kernel
    def func() -> int:
        x = False
        result = 0
        if x:
            result = 1
        else:
            result = 2
        return result

    assert func() == 2


@test_utils.test(require=qd.extension.data64, default_ip=qd.i64)
def test_u1_bool():
    @qd.kernel
    def func() -> qd.u1:
        return True

    assert func() == 1


@test_utils.test()
def test_bool_parameter():
    @qd.kernel
    def func(x: qd.u1) -> qd.u1:
        return not x

    assert func(False) == True


@test_utils.test()
def test_if():
    @qd.kernel
    def func(x: qd.u1) -> qd.u1:
        y = False
        if x:
            y = True
        return y

    assert func(2 == 2) == True
    assert func(2 == 3) == False


@test_utils.test()
def test_ternary():
    @qd.kernel
    def func(x: qd.i32) -> qd.u1:
        return True if x == 114514 else False

    assert func(114514) == True
    assert func(1919810) == False
