import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_simple():
    n = 128
    x = qd.field(qd.i32, shape=n)

    @qd.kernel
    def func():
        x[7] = 120

    func()

    for i in range(n):
        if i == 7:
            assert x[i] == 120
        else:
            assert x[i] == 0


@test_utils.test()
def test_range_loops():
    n = 128
    x = qd.field(qd.i32, shape=n)

    @qd.kernel
    def func():
        for i in range(n):
            x[i] = i + 123

    func()

    for i in range(n):
        assert x[i] == i + 123


@test_utils.test()
def test_python_access():
    n = 128
    x = qd.field(qd.i32, shape=n)

    x[3] = 123
    x[4] = 456
    assert x[3] == 123
    assert x[4] == 456


@test_utils.test()
def test_if():
    x = qd.field(qd.f32, shape=16)

    @qd.kernel
    def if_test():
        for i in x:
            if i < 100:
                x[i] = 100
            else:
                x[i] = i

    if_test()

    for i in range(16):
        assert x[i] == 100

    @qd.kernel
    def if_test2():
        for i in x:
            if i < 100:
                x[i] = i
            else:
                x[i] = 100

    if_test2()

    for i in range(16):
        assert x[i] == i


@test_utils.test()
def test_if_global_load():
    x = qd.field(qd.i32, shape=16)

    @qd.kernel
    def fill():
        for i in x:
            if x[i]:
                x[i] = i

    for i in range(16):
        x[i] = i % 2

    fill()

    for i in range(16):
        if i % 2 == 0:
            assert x[i] == 0
        else:
            assert x[i] == i


@test_utils.test()
def test_while_global_load():
    x = qd.field(qd.i32, shape=16)
    y = qd.field(qd.i32, shape=())

    @qd.kernel
    def run():
        while x[3]:
            x[3] -= 1
            y[None] += 1

    for i in range(16):
        x[i] = i

    run()

    assert y[None] == 3


@test_utils.test()
def test_datatype_string():
    for ty in [qd.u8, qd.u16, qd.u32, qd.u64, qd.i8, qd.i16, qd.i32, qd.f32, qd.f64]:
        assert ty.to_string() == str(ty)


@test_utils.test()
def test_nested_for_with_atomic():
    x = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def nested_loops():
        for i in range(2):
            x[None] += 1
            for j in range(1):
                x[None] += 2

    nested_loops()
