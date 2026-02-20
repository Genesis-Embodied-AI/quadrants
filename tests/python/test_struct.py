import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("round", range(10))
@test_utils.test()
def test_linear(round):
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.dense(qd.i, n).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test()
def test_linear_nested():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.i, n // 16).dense(qd.i, 16).place(x)
    qd.root.dense(qd.i, n // 16).dense(qd.i, 16).place(y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test()
def test_linear_nested_aos():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.i, n // 16).dense(qd.i, 16).place(x, y)

    for i in range(n):
        x[i] = i
        y[i] = i + 123

    for i in range(n):
        assert x[i] == i
        assert y[i] == i + 123


@test_utils.test(exclude=[qd.vulkan])
def test_2d_nested():
    x = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.ij, n // 16).dense(qd.ij, (32, 16)).place(x)

    for i in range(n * 2):
        for j in range(n):
            x[i, j] = i + j * 10

    for i in range(n * 2):
        for j in range(n):
            assert x[i, j] == i + j * 10


@test_utils.test()
def test_func_of_data_class_as_kernel_arg():
    @qd.dataclass
    class Foo:
        x: qd.f32
        y: qd.f32

        @qd.func
        def add(self, other: qd.template()):
            return Foo(self.x + other.x, self.y + other.y)

    @qd.kernel
    def foo_x(x: Foo) -> qd.f32:
        return x.add(x).x

    assert foo_x(Foo(1, 2)) == 2

    @qd.kernel
    def foo_y(x: Foo) -> qd.f32:
        return x.add(x).y

    assert foo_y(Foo(1, 2)) == 4


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_func_of_data_class_as_kernel_return():
    # TODO: enable this test in SPIR-V based backends after SPIR-V based backends can return structs.
    @qd.dataclass
    class Foo:
        x: qd.f32
        y: qd.f32

        @qd.func
        def add(self, other: qd.template()):
            return Foo(self.x + other.x, self.y + other.y)

        def add_python(self, other):
            return Foo(self.x + other.x, self.y + other.y)

    @qd.kernel
    def foo(x: Foo) -> Foo:
        return x.add(x)

    b = foo(Foo(1, 2))
    assert b.x == 2
    assert b.y == 4

    c = b.add_python(b)
    assert c.x == 4
    assert c.y == 8


@test_utils.test()
def test_nested_data_class_func():
    @qd.dataclass
    class Foo:
        a: int

        @qd.func
        def foo(self):
            return self.a

    @qd.dataclass
    class Nested:
        f: Foo

        @qd.func
        def testme(self) -> int:
            return self.f.foo()

    @qd.kernel
    def k() -> int:
        x = Nested(Foo(42))
        return x.testme()

    assert k() == 42


@test_utils.test()
def test_nested_data_class_func():
    with pytest.raises(qd.QuadrantsSyntaxError, match="Default value in @dataclass is not supported."):

        @qd.dataclass
        class Foo:
            a: int
            b: float = 3.14

        foo = Foo()
        print(foo)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_struct_field_with_bool():
    @qd.dataclass
    class S:
        a: qd.i16
        b: bool
        c: qd.i16

    sf = S.field(shape=(10, 1))
    sf[0, 0].b = False
    sf[0, 0].a = 0xFFFF
    sf[0, 0].c = 0xFFFF

    def foo() -> S:
        return sf[0, 0]

    assert foo().a == -1
    assert foo().c == -1
    assert foo().b == False

    sf[1, 0].a = 0x0000
    sf[1, 0].c = 0x0000
    sf[1, 0].b = True

    def bar() -> S:
        return sf[1, 0]

    assert bar().a == 0
    assert bar().c == 0
    assert bar().b == True


@test_utils.test()
def test_struct_special_element_name():
    @qd.dataclass
    class Foo:
        entries: int
        keys: int
        items: int
        methods: int

    @qd.kernel
    def foo() -> int:
        x = Foo(42, 21, 23, 11)
        return x.entries + x.keys + x.items + x.methods

    assert foo() == 42 + 21 + 23 + 11


@test_utils.test()
def test_struct_with_matrix():
    @qd.dataclass
    class TestStruct:
        p1: qd.math.vec3
        p2: qd.math.vec3

        @qd.func
        def get_vec(self, struct2, additional):
            self.p1 = (self.p1 + average(struct2)) / 2 + additional.p1

    global_struct = TestStruct(p1=[0, 2, 4], p2=[-2, -4, -6])

    @qd.func
    def average(struct) -> qd.math.vec3:
        return (struct.p1 + struct.p2) / 2

    @qd.kernel
    def process_struct(field1: qd.template(), field2: qd.template()):
        for i in field1:
            field1[i].get_vec(field2[i], global_struct)

    field1 = TestStruct.field()
    field2 = TestStruct.field()

    qd.root.dense(qd.i, 64).place(field1, field2)

    for i in range(64):
        field1[i] = TestStruct(p1=[1, 2, 3], p2=[4, 5, 6])
        field2[i] = TestStruct(p1=[4, 5, 6], p2=[1, 2, 3])

    process_struct(field1, field2)
