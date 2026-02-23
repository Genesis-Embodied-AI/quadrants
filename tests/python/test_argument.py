import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(debug=True)
def test_kernel_position_only_args():
    @qd.kernel
    def foo(a: qd.i32, b: qd.i32):
        assert a == 1
        assert b == 2

    foo(1, 2)


@test_utils.test(debug=True)
def test_kernel_keyword_args():
    @qd.kernel
    def foo(a: qd.i32, b: qd.i32):
        assert a == 1
        assert b == 2

    foo(1, b=2)


@test_utils.test(debug=True)
def test_kernel_args_missing():
    @qd.kernel
    def foo(a: qd.i32, b: qd.i32):
        assert a == 1
        assert b == 2

    with pytest.raises(qd.QuadrantsSyntaxError, match="Missing argument 'b'"):
        foo(2)


@test_utils.test(debug=True)
def test_kernel_keyword_args_missing():
    @qd.kernel
    def foo(a: qd.i32, b: qd.i32):
        assert a == 1
        assert b == 2

    with pytest.raises(qd.QuadrantsSyntaxError, match="Missing argument 'a'"):
        foo(b=2)


@test_utils.test(debug=True)
def test_kernel_keyword_args_not_found():
    @qd.kernel
    def foo(a: qd.i32, b: qd.i32):
        assert a == 1
        assert b == 2

    with pytest.raises(qd.QuadrantsSyntaxError, match="Unexpected argument 'c'"):
        foo(1, 2, c=2)


@test_utils.test(debug=True)
def test_kernel_too_many():
    @qd.kernel
    def foo(a: qd.i32, b: qd.i32):
        assert a == 1
        assert b == 2

    with pytest.raises(qd.QuadrantsSyntaxError, match="Too many arguments"):
        foo(1, 2, 3)


@test_utils.test(debug=True)
def test_function_keyword_args():
    @qd.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @qd.func
    def bar(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 4

    @qd.func
    def all_default(a=1, b=2, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @qd.func
    def do_nothing():
        pass

    @qd.kernel
    def baz():
        foo(1, b=2)
        bar(b=2, a=1, c=4)
        all_default()
        do_nothing()

    baz()


@test_utils.test(debug=True)
def test_function_keyword_args_missing():
    @qd.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @qd.kernel
    def missing():
        foo(1, c=3)

    with pytest.raises(qd.QuadrantsSyntaxError, match="Missing argument 'b'"):
        missing()


@test_utils.test(debug=True)
def test_function_keyword_args_not_found():
    @qd.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @qd.kernel
    def not_found():
        foo(1, 2, 3, d=3)

    with pytest.raises(qd.QuadrantsSyntaxError, match="Unexpected argument 'd'"):
        not_found()


@test_utils.test(debug=True)
def test_function_too_many():
    @qd.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @qd.kernel
    def many():
        foo(1, 2, 3, 4)

    with pytest.raises(qd.QuadrantsSyntaxError, match="Too many arguments"):
        many()


@test_utils.test(debug=True)
def test_function_keyword_args_duplicate():
    @qd.func
    def foo(a, b, c=3):
        assert a == 1
        assert b == 2
        assert c == 3

    @qd.kernel
    def duplicate():
        foo(1, a=3, b=3)

    with pytest.raises(qd.QuadrantsSyntaxError, match="Multiple values for argument 'a'"):
        duplicate()


@test_utils.test()
def test_args_with_many_ndarrays():
    particle_num = 0
    cluster_num = 0
    permu_num = 0

    particlePosition = qd.Vector.ndarray(3, qd.f32, shape=10)
    outClusterPosition = qd.Vector.ndarray(3, qd.f32, shape=10)
    outClusterOffsets = qd.ndarray(qd.i32, shape=10)
    outClusterSizes = qd.ndarray(qd.i32, shape=10)
    outClusterIndices = qd.ndarray(qd.i32, shape=10)

    particle_pos = qd.Vector.ndarray(3, qd.f32, shape=20)
    particle_prev_pos = qd.Vector.ndarray(3, qd.f32, shape=20)
    particle_rest_pos = qd.Vector.ndarray(3, qd.f32, shape=20)
    particle_index = qd.ndarray(qd.i32, shape=20)

    cluster_rest_mass_center = qd.Vector.ndarray(3, qd.f32, shape=20)
    cluster_begin = qd.ndarray(qd.i32, shape=20)

    @qd.kernel
    def ti_import_cluster_data(
        center: qd.types.vector(3, qd.f32),
        particle_num: int,
        cluster_num: int,
        permu_num: int,
        particlePosition: qd.types.ndarray(ndim=1),
        outClusterPosition: qd.types.ndarray(ndim=1),
        outClusterOffsets: qd.types.ndarray(ndim=1),
        outClusterSizes: qd.types.ndarray(ndim=1),
        outClusterIndices: qd.types.ndarray(ndim=1),
        particle_pos: qd.types.ndarray(ndim=1),
        particle_prev_pos: qd.types.ndarray(ndim=1),
        particle_rest_pos: qd.types.ndarray(ndim=1),
        cluster_rest_mass_center: qd.types.ndarray(ndim=1),
        cluster_begin: qd.types.ndarray(ndim=1),
        particle_index: qd.types.ndarray(ndim=1),
    ):
        added_permu_num = outClusterIndices.shape[0]

        for i in range(added_permu_num):
            particle_index[i] = 1.0

    center = qd.math.vec3(0, 0, 0)
    ti_import_cluster_data(
        center,
        particle_num,
        cluster_num,
        permu_num,
        particlePosition,
        outClusterPosition,
        outClusterOffsets,
        outClusterSizes,
        outClusterIndices,
        particle_pos,
        particle_prev_pos,
        particle_rest_pos,
        cluster_rest_mass_center,
        cluster_begin,
        particle_index,
    )


@test_utils.test()
def test_struct_arg():
    s0 = qd.types.struct(a=qd.i16, b=qd.f32)
    s1 = qd.types.struct(a=qd.f32, b=s0)

    @qd.kernel
    def foo(a: s1) -> qd.f32:
        return a.a + a.b.a + a.b.b

    ret = foo(s1(a=1, b=s0(a=65537, b=123)))
    assert ret == pytest.approx(125)


@test_utils.test()
def test_struct_arg_with_matrix():
    mat = qd.types.matrix(3, 2, qd.f32)
    s0 = qd.types.struct(a=mat, b=qd.f32)
    s1 = qd.types.struct(a=qd.i32, b=s0)

    @qd.kernel
    def foo(a: s1) -> qd.i32:
        ret = a.a + a.b.b
        for i in range(3):
            for j in range(2):
                ret += a.b.a[i, j] * (i + 1) * (j + 2)
        return ret

    arg = s1(a=1, b=s0(a=mat(1, 2, 3, 4, 5, 6), b=123))
    ret_std = 1 + 123

    for i in range(3):
        for j in range(2):
            ret_std += (i + 1) * (j + 2) * (i * 2 + j + 1)

    ret = foo(arg)
    assert ret == ret_std


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_struct_arg_with_matrix_real_func():
    mat = qd.types.matrix(3, 2, qd.f32)
    s0 = qd.types.struct(a=mat, b=qd.f32)
    s1 = qd.types.struct(a=qd.i32, b=s0)

    @qd.real_func
    def foo(a: s1) -> qd.i32:
        ret = a.a + a.b.b
        for i in range(3):
            for j in range(2):
                ret += a.b.a[i, j] * (i + 1) * (j + 2)
        return ret

    @qd.kernel
    def bar(a: s1) -> qd.i32:
        return foo(a)

    arg = s1(a=1, b=s0(a=mat(1, 2, 3, 4, 5, 6), b=123))
    ret_std = 1 + 123

    for i in range(3):
        for j in range(2):
            ret_std += (i + 1) * (j + 2) * (i * 2 + j + 1)

    ret = bar(arg)
    assert ret == ret_std


@test_utils.test()
def test_func_scalar_arg_cast():
    @qd.func
    def bar(a: qd.i32) -> qd.f32:
        return a

    @qd.kernel
    def foo(a: qd.f32) -> qd.f32:
        return bar(a)

    assert foo(1.5) == 1.0


@test_utils.test(exclude=[qd.amdgpu])
def test_arg_4k():
    vec1024 = qd.types.vector(1024, qd.i32)

    @qd.kernel
    def bar(a: vec1024) -> qd.i32:
        ret = 0
        for i in range(1024):
            ret += a[i]

        return ret

    a = vec1024([i for i in range(1024)])
    assert bar(a) == 523776
