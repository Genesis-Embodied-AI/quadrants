import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_ptr_scalar():
    a = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def func(t: qd.f32):
        b = qd.static(a)
        c = qd.static(b)
        b[None] = b[None] * t
        c[None] = a[None] + t

    for x, y in zip(range(-5, 5), range(-4, 4)):
        a[None] = x
        func(y)
        assert a[None] == x * y + y


@test_utils.test()
def test_ptr_matrix():
    a = qd.Matrix.field(2, 2, dtype=qd.f32, shape=())

    @qd.kernel
    def func(t: qd.f32):
        a[None] = [[2, 3], [4, 5]]
        b = qd.static(a)
        b[None][1, 0] = t

    for x in range(-5, 5):
        func(x)
        assert a[None][1, 0] == x


@test_utils.test()
def test_ptr_field():
    a = qd.field(dtype=qd.f32, shape=(3, 4))

    @qd.kernel
    def func(t: qd.f32):
        b = qd.static(a)
        b[1, 3] = b[1, 2] * t
        b[2, 0] = b[2, 1] + t

    for x, y in zip(range(-5, 5), range(-4, 4)):
        a[1, 2] = x
        a[2, 1] = x
        func(y)
        assert a[1, 3] == x * y
        assert a[2, 0] == x + y


@test_utils.test()
def test_pythonish_tuple_assign():
    a = qd.field(dtype=qd.f32, shape=())
    b = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def func(x: qd.f32, y: qd.f32):
        u, v = qd.static(b, a)
        u[None] = x
        v[None] = y

    for x, y in zip(range(-5, 5), range(-4, 4)):
        func(x, y)
        assert a[None] == y
        assert b[None] == x


@test_utils.test()
def test_ptr_func():
    a = qd.field(dtype=qd.f32, shape=(3,))

    def add2numbers_py(x, y):
        return x + y

    @qd.func
    def add2numbers_func(x, y):
        return x + y

    @qd.kernel
    def func():
        add_py = qd.static(add2numbers_py)
        add_func = qd.static(add2numbers_func)
        a[0] = add_py(2, 3)
        a[1] = add_func(3, 7)

    func()
    assert a[0] == 5.0
    assert a[1] == 10.0


@test_utils.test()
def test_ptr_class_func():
    @qd.data_oriented
    class MyClass:
        def __init__(self):
            self.a = qd.field(dtype=qd.f32, shape=(3))

        def add2numbers_py(self, x, y):
            return x + y

        @qd.func
        def add2numbers_func(self, x, y):
            return x + y

        @qd.kernel
        def func(self):
            a, add_py, add_func = qd.static(self.a, self.add2numbers_py, self.add2numbers_func)
            a[0] = add_py(2, 3)
            a[1] = add_func(3, 7)

    obj = MyClass()
    obj.func()
    assert obj.a[0] == 5.0
    assert obj.a[1] == 10.0
