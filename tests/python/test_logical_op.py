import quadrants as qd

from tests import test_utils


@test_utils.test(debug=True)
def test_logical_and_i32():
    @qd.kernel
    def func(x: qd.i32, y: qd.i32) -> qd.i32:
        return x and y

    assert func(1, 2) == 2
    assert func(2, 1) == 1
    assert func(0, 1) == 0
    assert func(1, 0) == 0


@test_utils.test(debug=True)
def test_logical_or_i32():
    @qd.kernel
    def func(x: qd.i32, y: qd.i32) -> qd.i32:
        return x or y

    assert func(1, 2) == 1
    assert func(2, 1) == 2
    assert func(1, 0) == 1
    assert func(0, 1) == 1


@test_utils.test(debug=True)
def test_logical_vec_i32():
    vec4d = qd.types.vector(4, qd.i32)

    @qd.kernel
    def p() -> vec4d:
        a = qd.Vector([2, 2, 0, 0])
        b = qd.Vector([1, 0, 1, 0])
        z = a or b
        return z

    @qd.kernel
    def q() -> vec4d:
        a = qd.Vector([2, 2, 0, 0])
        b = qd.Vector([1, 0, 1, 0])
        z = a and b
        return z

    x = p()
    y = q()

    assert x[0] == 1
    assert x[1] == 1
    assert x[2] == 1
    assert x[3] == 0
    assert y[0] == 1
    assert y[1] == 0
    assert y[2] == 0
    assert y[3] == 0


# FIXME: bool vectors not supported on spir-v
@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True)
def test_logical_vec_bool():
    vec4d = qd.types.vector(4, qd.u1)

    @qd.kernel
    def p() -> vec4d:
        a = qd.Vector([True, True, False, False])
        b = qd.Vector([True, False, True, False])
        z = a or b
        return z

    @qd.kernel
    def q() -> vec4d:
        a = qd.Vector([True, True, False, False])
        b = qd.Vector([True, False, True, False])
        z = a and b
        return z

    x = p()
    y = q()

    assert x[0] == True
    assert x[1] == True
    assert x[2] == True
    assert x[3] == False
    assert y[0] == True
    assert y[1] == False
    assert y[2] == False
    assert y[3] == False
