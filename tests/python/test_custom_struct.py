import numpy as np
from pytest import approx

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test()
def test_struct_member_access():
    n = 32

    x = qd.Struct.field({"a": qd.f32, "b": qd.f32}, shape=(n,))
    y = qd.Struct.field({"a": qd.f32, "b": qd.f32})

    qd.root.dense(qd.i, n // 4).dense(qd.i, 4).place(y)

    @qd.kernel
    def init():
        for i in x:
            x[i].a = i
            y[i].a = i

    @qd.kernel
    def run_quadrants_scope():
        for i in x:
            x[i].b = x[i].a

    def run_python_scope():
        for i in range(n):
            y[i].b = y[i].a * 2 + 1

    init()
    run_quadrants_scope()
    for i in range(n):
        assert x[i].b == i
    run_python_scope()
    for i in range(n):
        assert y[i].b == i * 2 + 1


@test_utils.test()
def test_struct_whole_access():
    n = 32

    # also tests implicit cast
    x = qd.Struct.field({"a": qd.i32, "b": qd.f32}, shape=(n,))
    y = qd.Struct.field({"a": qd.f32, "b": qd.i32})

    qd.root.dense(qd.i, n // 4).dense(qd.i, 4).place(y)

    @qd.kernel
    def init():
        for i in x:
            x[i] = qd.Struct(a=2 * i, b=1.01 * i)

    @qd.kernel
    def run_quadrants_scope():
        for i in x:
            y[i].a = x[i].a * 2 + 1
            y[i].b = x[i].b * 2 + 1

    def run_python_scope():
        for i in range(n):
            y[i] = qd.Struct(a=x[i].a, b=int(x[i].b))

    init()
    for i in range(n):
        assert x[i].a == 2 * i
        assert x[i].b == approx(1.01 * i, rel=1e-4)
    run_quadrants_scope()
    for i in range(n):
        assert y[i].a == 4 * i + 1
        assert y[i].b == int((1.01 * i) * 2 + 1)
    run_python_scope()
    for i in range(n):
        assert y[i].a == 2 * i
        assert y[i].b == int(1.01 * i)


@test_utils.test()
def test_struct_fill():
    n = 32

    # also tests implicit cast
    x = qd.Struct.field({"a": qd.f32, "b": qd.types.vector(3, qd.i32)}, shape=(n,))

    def fill_each():
        x.a.fill(1.0)
        x.b.fill(1.5)

    def fill_all():
        x.fill(2.5)

    @qd.kernel
    def fill_elements():
        for i in x:
            x[i].a = i + 0.5
            x[i].b.fill(i + 0.5)

    fill_each()
    for i in range(n):
        assert x[i].a == 1.0
        assert x[i].b[0] == 1 and x[i].b[1] == 1 and x[i].b[2] == 1
    fill_all()
    for i in range(n):
        assert x[i].a == 2.5
        assert x[i].b[0] == 2 and x[i].b[1] == 2 and x[i].b[2] == 2
    fill_elements()
    for i in range(n):
        assert x[i].a == i + 0.5
        assert np.allclose(x[i].b.to_numpy(), int(x[i].a))


@test_utils.test()
def test_matrix_type():
    n = 32
    vec2f = qd.types.vector(2, qd.f32)
    vec3i = qd.types.vector(3, qd.i32)
    x = vec3i.field()
    qd.root.dense(qd.i, n).place(x)

    @qd.kernel
    def run_quadrants_scope():
        for i in x:
            v = vec2f(i + 0.2)
            # also tests implicit cast
            x[i] = vec3i(v, i + 1.2)

    def run_python_scope():
        for i in range(n):
            v = vec2f(i + 0.2)
            x[i] = vec3i(i + 1.8, v)

    run_quadrants_scope()
    for i in range(n):
        assert np.allclose(x[i].to_numpy(), np.array([i, i, i + 1]))
    run_python_scope()
    for i in range(n):
        assert np.allclose(x[i].to_numpy(), np.array([i + 1, i, i]))


@test_utils.test()
def test_struct_type():
    n = 32
    vec3f = qd.types.vector(3, float)
    line3f = qd.types.struct(linedir=vec3f, length=float)
    mystruct = qd.types.struct(line=line3f, idx=int)
    x = mystruct.field(shape=(n,))

    @qd.kernel
    def init_quadrants_scope():
        for i in x:
            x[i] = mystruct(1)

    def init_python_scope():
        for i in range(n):
            x[i] = mystruct(3)

    @qd.kernel
    def run_quadrants_scope():
        for i in x:
            v = vec3f(1)
            line = line3f(linedir=v, length=i + 0.5)
            x[i] = mystruct(line=line, idx=i)

    def run_python_scope():
        for i in range(n):
            v = vec3f(1)
            x[i] = qd.Struct({"line": {"linedir": v, "length": i + 0.5}, "idx": i})

    init_quadrants_scope()
    for i in range(n):
        assert x[i].idx == 0
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == 0.0
    run_quadrants_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5
    init_python_scope()
    for i in range(n):
        assert x[i].idx == 0
        assert np.allclose(x[i].line.linedir.to_numpy(), 3.0)
        assert x[i].line.length == 0.0
    run_python_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5
    x.fill(5)
    for i in range(n):
        assert x[i].idx == 5
        assert np.allclose(x[i].line.linedir.to_numpy(), 5.0)
        assert x[i].line.length == 5.0


@test_utils.test()
def test_dataclass():
    # example struct class type
    vec3f = qd.types.vector(3, float)

    @qd.dataclass
    class Sphere:
        center: vec3f
        radius: qd.f32

        @qd.func
        def area(self):
            return 4 * 3.14 * self.radius * self.radius

        def py_scope_area(self):
            return 4 * 3.14 * self.radius * self.radius

    # test function usage from python scope
    assert np.isclose(Sphere(center=vec3f(0.0), radius=2.0).py_scope_area(), 4.0 * 3.14 * 4.0)

    # test function usage from quadrants scope
    @qd.kernel
    def get_area() -> qd.f32:
        sphere = Sphere(center=vec3f(0.0), radius=2.0)
        return sphere.area()

    assert np.isclose(get_area(), 4.0 * 3.14 * 4.0)

    # test function usage from quadrants scope with field
    struct_field = Sphere.field(shape=(4,))
    struct_field[3] = Sphere(center=vec3f(0.0), radius=2.0)

    @qd.kernel
    def get_area_field() -> qd.f32:
        return struct_field[3].area()

    assert np.isclose(get_area_field(), 4.0 * 3.14 * 4.0)


@test_utils.test()
def test_struct_assign():
    n = 32
    vec3f = qd.types.vector(3, float)
    line3f = qd.types.struct(linedir=vec3f, length=float)
    mystruct = qd.types.struct(line=line3f, idx=int)
    x = mystruct.field(shape=(n,))
    y = line3f.field(shape=(n,))

    @qd.kernel
    def init():
        for i in y:
            y[i] = line3f(linedir=vec3f(1), length=i + 0.5)

    @qd.kernel
    def run_quadrants_scope():
        for i in x:
            x[i].idx = i
            x[i].line = y[i]

    def run_python_scope():
        for i in range(n):
            x[i].idx = i
            x[i].line = y[i]

    init()
    run_quadrants_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5
    x.fill(5)
    run_python_scope()
    for i in range(n):
        assert x[i].idx == i
        assert np.allclose(x[i].line.linedir.to_numpy(), 1.0)
        assert x[i].line.length == i + 0.5


@test_utils.test()
def test_compound_type_implicit_cast():
    vec2i = qd.types.vector(2, int)
    vec2f = qd.types.vector(2, float)
    structi = qd.types.struct(a=int, b=vec2i)
    structf = qd.types.struct(a=float, b=vec2f)

    @qd.kernel
    def f2i_quadrants_scope() -> int:
        s = structi(2.5, (2.5, 2.5))
        return s.a + s.b[0] + s.b[1]

    def f2i_python_scope():
        s = structi(2.5, (2.5, 2.5))
        return s.a + s.b[0] + s.b[1]

    @qd.kernel
    def i2f_quadrants_scope() -> float:
        s = structf(2, (2, 2))
        return s.a + s.b[0] + s.b[1]

    def i2f_python_scope():
        s = structf(2, (2, 2))
        return s.a + s.b[0] + s.b[1]

    int_value = f2i_quadrants_scope()
    assert isinstance(int_value, (int, np.integer)) and int_value == 6
    int_value = f2i_python_scope()
    assert isinstance(int_value, (int, np.integer)) and int_value == 6
    float_value = i2f_quadrants_scope()
    assert isinstance(float_value, (float, np.floating)) and float_value == approx(6.0, rel=1e-4)
    float_value = i2f_python_scope()
    assert isinstance(float_value, (float, np.floating)) and float_value == approx(6.0, rel=1e-4)


@test_utils.test()
def test_local_struct_assign():
    n = 32
    vec3f = qd.types.vector(3, float)
    line3f = qd.types.struct(linedir=vec3f, length=float)
    mystruct = qd.types.struct(line=line3f, idx=int)

    @qd.kernel
    def run_quadrants_scope():
        y = line3f(0)
        x = mystruct(0)
        x.idx = 0
        x.line = y

    def run_python_scope():
        y = line3f(0)
        x = mystruct(0)
        x.idx = 0
        x.line = y

    run_quadrants_scope()
    run_python_scope()


@test_utils.test(debug=True)
def test_copy_python_scope_struct_to_quadrants_scope():
    a = qd.Struct({"a": 2, "b": 3})

    @qd.kernel
    def test():
        b = a
        assert b.a == 2
        assert b.b == 3
        b = qd.Struct({"a": 3, "b": 4})
        assert b.a == 3
        assert b.b == 4

    test()


@test_utils.test(debug=True)
def test_copy_struct_field_element_to_quadrants_scope():
    a = qd.Struct.field({"a": qd.i32, "b": qd.i32}, shape=())
    a[None].a = 2
    a[None].b = 3

    @qd.kernel
    def test():
        b = a[None]
        assert b.a == 2
        assert b.b == 3
        b.a = 5
        b.b = 9
        assert b.a == 5
        assert b.b == 9
        assert a[None].a == 2
        assert a[None].b == 3

    test()


@test_utils.test(debug=True)
def test_copy_struct_in_quadrants_scope():
    @qd.kernel
    def test():
        a = qd.Struct({"a": 2, "b": 3})
        b = a
        assert b.a == 2
        assert b.b == 3
        b.a = 5
        b.b = 9
        assert b.a == 5
        assert b.b == 9
        assert a.a == 2
        assert a.b == 3

    test()


@test_utils.test(debug=True)
def test_dataclass():
    vec3 = qd.types.vector(3, float)

    @qd.dataclass
    class Foo:
        pos: vec3
        vel: vec3
        mass: float

    @qd.kernel
    def test():
        A = Foo((1, 1, 1), mass=2)
        assert all(A.pos == [1.0, 1.0, 1.0])
        assert all(A.vel == [0.0, 0.0, 0.0])
        assert A.mass == 2.0

    test()


@test_utils.test(arch=get_host_arch_list())
def test_name_collision():
    # https://github.com/taichi-dev/quadrants/issues/6652
    @qd.dataclass
    class Foo:
        zoo: qd.f32

    @qd.dataclass
    class Bar:
        @qd.func
        def zoo(self):
            return 0

    Foo()  # instantiate struct with zoo as member first
    Bar()  # then instantiate struct with zoo as method


@test_utils.test(debug=True)
def test_dataclass_as_member():
    # https://github.com/taichi-dev/quadrants/issues/6884
    @qd.dataclass
    class A:
        i: int
        j: float

    @qd.dataclass
    class B:
        a1: A
        a2: A

    a = A(1, 2.0)
    b = B(a)
    assert b.a1.i == 1 and b.a1.j == 2.0
