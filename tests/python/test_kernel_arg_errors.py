import platform

import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=qd.cpu)
def test_pass_float_as_i32():
    @qd.kernel
    def foo(a: qd.i32):
        pass

    with pytest.raises(
        qd.QuadrantsRuntimeTypeError,
        match=r"Argument \(0,\) \(type=<class 'float'>\) cannot be converted into required type i32",
    ) as e:
        foo(1.2)


@test_utils.test(arch=qd.cpu)
def test_pass_float_as_ndarray():
    @qd.kernel
    def foo(a: qd.types.ndarray()):
        pass

    with pytest.raises(
        qd.QuadrantsRuntimeTypeError,
        match=r"Invalid type for argument a, got 1.2",
    ):
        foo(1.2)


@test_utils.test(arch=qd.cpu)
def test_random_python_class_as_ndarray():
    @qd.kernel
    def foo(a: qd.types.ndarray()):
        pass

    class Bla:
        pass

    with pytest.raises(
        qd.QuadrantsRuntimeTypeError,
        match=r"Invalid type for argument a, got",
    ):
        b = Bla()
        foo(b)


@test_utils.test(exclude=[qd.metal])
def test_pass_u64():
    if qd.lang.impl.current_cfg().arch == qd.vulkan and platform.system() == "Darwin":
        return

    @qd.kernel
    def foo(a: qd.u64):
        pass

    foo(2**64 - 1)


@test_utils.test()
def test_argument_redefinition():
    @qd.kernel
    def foo(a: qd.i32):
        a = 1

    with pytest.raises(qd.QuadrantsSyntaxError, match='Kernel argument "a" is immutable in the kernel') as e:
        foo(5)


@test_utils.test()
def test_argument_augassign():
    @qd.kernel
    def foo(a: qd.i32):
        a += 1

    with pytest.raises(qd.QuadrantsSyntaxError, match='Kernel argument "a" is immutable in the kernel') as e:
        foo(5)


@test_utils.test()
def test_argument_annassign():
    @qd.kernel
    def foo(a: qd.i32):
        a: qd.i32 = 1

    with pytest.raises(qd.QuadrantsSyntaxError, match='Kernel argument "a" is immutable in the kernel') as e:
        foo(5)


@test_utils.test()
def test_pass_struct_mismatch():
    sphere_type = qd.types.struct(center=qd.math.vec3, radius=float)
    circle_type = qd.types.struct(center=qd.math.vec2, radius=float)

    @qd.kernel
    def foo(sphere: sphere_type):
        pass

    with pytest.raises(
        qd.QuadrantsRuntimeTypeError,
        match=r"Argument <class 'quadrants.lang.struct.Struct.* cannot be converted into required type <qd"
        r".StructType center=<quadrants.lang.matrix.VectorType object at .*>, radius=f32, struct_methods={}>",
    ) as e:
        foo(circle_type(center=qd.math.vec2([1, 2]), radius=2.33))
