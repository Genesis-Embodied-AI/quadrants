import math
import operator
import platform
import sys

import numpy as np
import pytest
from pytest import approx

import quadrants as qd
from quadrants.lang import impl
from quadrants.lang.exception import QuadrantsCompilationError
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

matrix_operation_types = [operator.add, operator.sub, operator.matmul]
test_matrix_arrays = [
    np.array([[1, 2], [3, 4]]),
    np.array([[5, 6], [7, 8]]),
    np.array([[2, 8], [-1, 3]]),
]

vector_operation_types = [operator.add, operator.sub]
test_vector_arrays = [np.array([42, 42]), np.array([24, 24]), np.array([83, 12])]

u = platform.uname()


@test_utils.test(arch=get_host_arch_list())
def test_python_scope_vector_operations():
    for ops in vector_operation_types:
        a, b = test_vector_arrays[:2]
        m1, m2 = qd.Vector(a), qd.Vector(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


@test_utils.test(arch=get_host_arch_list())
def test_python_scope_matrix_operations():
    for ops in matrix_operation_types:
        a, b = test_matrix_arrays[:2]
        m1, m2 = qd.Matrix(a), qd.Matrix(b)
        c = ops(m1, m2)
        assert np.allclose(c.to_numpy(), ops(a, b))


# TODO: Loops inside the function will cause AssertionError:
# No new variables can be declared after kernel invocations
# or Python-scope field accesses.
# ideally we should use pytest.fixture to parameterize the tests
# over explicit loops
@pytest.mark.parametrize("ops", vector_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_python_scope_vector_field(ops):
    t1 = qd.Vector.field(2, dtype=qd.i32, shape=())
    t2 = qd.Vector.field(2, dtype=qd.i32, shape=())
    a, b = test_vector_arrays[:2]
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None], t2[None])
    assert np.allclose(c.to_numpy(), ops(a, b))


@pytest.mark.parametrize("ops", matrix_operation_types)
@test_utils.test(arch=get_host_arch_list())
def test_python_scope_matrix_field(ops):
    t1 = qd.Matrix.field(2, 2, dtype=qd.i32, shape=())
    t2 = qd.Matrix.field(2, 2, dtype=qd.i32, shape=())
    a, b = test_matrix_arrays[:2]
    # ndarray not supported here
    t1[None], t2[None] = a.tolist(), b.tolist()

    c = ops(t1[None], t2[None])
    print(c)

    assert np.allclose(c.to_numpy(), ops(a, b))


@test_utils.test(arch=get_host_arch_list())
def test_constant_matrices():
    assert qd.cos(math.pi / 3) == test_utils.approx(0.5)
    assert np.allclose((-qd.Vector([2, 3])).to_numpy(), np.array([-2, -3]))
    assert qd.cos(qd.Vector([2, 3])).to_numpy() == test_utils.approx(np.cos(np.array([2, 3])))
    assert qd.max(2, 3) == 3
    res = qd.max(4, qd.Vector([3, 4, 5]))
    assert np.allclose(res.to_numpy(), np.array([4, 4, 5]))
    res = qd.Vector([2, 3]) + qd.Vector([3, 4])
    assert np.allclose(res.to_numpy(), np.array([5, 7]))
    res = qd.atan2(qd.Vector([2, 3]), qd.Vector([3, 4]))
    assert res.to_numpy() == test_utils.approx(np.arctan2(np.array([2, 3]), np.array([3, 4])))
    res = qd.Matrix([[2, 3], [4, 5]]) @ qd.Vector([2, 3])
    assert np.allclose(res.to_numpy(), np.array([13, 23]))
    v = qd.Vector([3, 4])
    w = qd.Vector([5, -12])
    r = qd.Vector([1, 2, 3, 4])
    s = qd.Matrix([[1, 2], [3, 4]])
    assert v.normalized().to_numpy() == test_utils.approx(np.array([0.6, 0.8]))
    assert v.cross(w) == test_utils.approx(-12 * 3 - 4 * 5)
    w.y = v.x * w[0]
    r.x = r.y
    r.y = r.z
    r.z = r.w
    r.w = r.x
    assert np.allclose(w.to_numpy(), np.array([5, 15]))
    assert qd.select(qd.Vector([1, 0]), qd.Vector([2, 3]), qd.Vector([4, 5])) == qd.Vector([2, 5])
    s[0, 1] = 2
    assert s[0, 1] == 2

    @qd.kernel
    def func(t: qd.i32):
        m = qd.Matrix([[2, 3], [4, t]])
        print(m @ qd.Vector([2, 3]))
        m += qd.Matrix([[3, 4], [5, t]])
        print(m @ v)
        print(r.x, r.y, r.z, r.w)
        s = w @ m
        print(s)
        print(m)

    func(5)


@test_utils.test(arch=get_host_arch_list())
def test_quadrants_scope_vector_operations_with_global_vectors():
    for ops in vector_operation_types:
        a, b, c = test_vector_arrays[:3]
        m1, m2 = qd.Vector(a), qd.Vector(b)
        r1 = qd.Vector.field(2, dtype=qd.i32, shape=())
        r2 = qd.Vector.field(2, dtype=qd.i32, shape=())
        m3 = qd.Vector.field(2, dtype=qd.i32, shape=())
        m3.from_numpy(c)

        @qd.kernel
        def run():
            r1[None] = ops(m1, m2)
            r2[None] = ops(m1, m3[None])

        run()

        assert np.allclose(r1[None].to_numpy(), ops(a, b))
        assert np.allclose(r2[None].to_numpy(), ops(a, c))


@test_utils.test(arch=get_host_arch_list())
def test_quadrants_scope_matrix_operations_with_global_matrices():
    for ops in matrix_operation_types:
        a, b, c = test_matrix_arrays[:3]
        m1, m2 = qd.Matrix(a), qd.Matrix(b)
        r1 = qd.Matrix.field(2, 2, dtype=qd.i32, shape=())
        r2 = qd.Matrix.field(2, 2, dtype=qd.i32, shape=())
        m3 = qd.Matrix.field(2, 2, dtype=qd.i32, shape=())
        m3.from_numpy(c)

        @qd.kernel
        def run():
            r1[None] = ops(m1, m2)
            r2[None] = ops(m1, m3[None])

        run()

        assert np.allclose(r1[None].to_numpy(), ops(a, b))
        assert np.allclose(r2[None].to_numpy(), ops(a, c))


def _test_local_matrix_non_constant_index():
    @qd.kernel
    def func1() -> qd.types.vector(3, qd.i32):
        tmp = qd.Vector([1, 2, 3])
        for i in range(3):
            vec = qd.Vector([4, 5, 6])
            for j in range(3):
                vec[tmp[i] % 3] += vec[j]
            tmp[i] = vec[tmp[i] % 3]
        return tmp

    assert (func1() == qd.Vector([24, 30, 19])).all()

    @qd.kernel
    def func2(i: qd.i32, j: qd.i32, k: qd.i32) -> qd.i32:
        tmp = qd.Matrix([[k, k * 2], [k * 2, k * 3]])
        return tmp[i, j]

    for i in range(2):
        for j in range(2):
            assert func2(i, j, 10) == 10 * (i + j + 1)


@test_utils.test()
def test_local_matrix_non_constant_index():
    _test_local_matrix_non_constant_index()


@test_utils.test(arch=[qd.cuda, qd.cpu], real_matrix_scalarize=False)
def test_local_matrix_non_constant_index_real_matrix():
    _test_local_matrix_non_constant_index()


@test_utils.test()
def test_matrix_ndarray_non_constant_index():
    @qd.kernel
    def func1(a: qd.types.ndarray(dtype=qd.math.mat2)):
        for i in range(5):
            for j, k in qd.ndrange(2, 2):
                a[i][j, k] = j * j + k * k

    m = np.empty((5, 2, 2), dtype=np.int32)
    func1(m)
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1

    @qd.kernel
    def func2(b: qd.types.ndarray(dtype=qd.types.vector(n=10, dtype=qd.i32))):
        for i in range(5):
            for j in range(4):
                b[i][j * j] = j * j

    v = np.empty((5, 10), dtype=np.int32)
    func2(v)
    assert v[0][0] == 0
    assert v[1][1] == 1
    assert v[2][4] == 4
    assert v[3][9] == 9


@test_utils.test()
def test_matrix_field_non_constant_index():
    m = qd.Matrix.field(2, 2, qd.i32, 5)
    v = qd.Vector.field(10, qd.i32, 5)

    @qd.kernel
    def func1():
        for i in range(5):
            for j, k in qd.ndrange(2, 2):
                m[i][j, k] = j * j + k * k

    func1()
    assert m[1][0, 1] == 1
    assert m[2][1, 0] == 1
    assert m[3][1, 1] == 2
    assert m[4][0, 1] == 1

    @qd.kernel
    def func2():
        for i in range(5):
            for j in range(4):
                v[i][j * j] = j * j

    func2()
    assert v[1][0] == 0
    assert v[1][1] == 1
    assert v[1][4] == 4
    assert v[1][9] == 9


@test_utils.test()
def test_matrix_field_constant_index():
    m = qd.Matrix.field(2, 2, qd.i32, 5)

    @qd.kernel
    def func():
        for i in range(5):
            for j, k in qd.static(qd.ndrange(2, 2)):
                m[i][j, k] = 12

    func()

    assert np.allclose(m.to_numpy(), np.ones((5, 2, 2), np.int32) * 12)


@test_utils.test(arch=qd.cpu)
def test_vector_to_list():
    a = qd.Vector.field(2, float, ())

    data = [2, 3]
    b = qd.Vector(data)
    assert list(b) == data
    assert len(b) == len(data)

    a[None] = b
    assert all(a[None] == qd.Vector(data))


@test_utils.test(arch=qd.cpu)
def test_matrix_to_list():
    a = qd.Matrix.field(2, 3, float, ())

    data = [[2, 3, 4], [5, 6, 7]]
    b = qd.Matrix(data)
    assert list(b) == data
    assert len(b) == len(data)

    a[None] = b
    assert all(a[None] == qd.Matrix(data))


@test_utils.test()
def test_matrix_needs_grad():
    # Just make sure the usage doesn't crash, see https://github.com/taichi-dev/quadrants/pull/1545
    n = 8
    m1 = qd.Matrix.field(2, 2, qd.f32, n, needs_grad=True)
    m2 = qd.Matrix.field(2, 2, qd.f32, n, needs_grad=True)
    gr = qd.Matrix.field(2, 2, qd.f32, n)

    @qd.kernel
    def func():
        for i in range(n):
            gr[i] = m1.grad[i] + m2.grad[i]

    func()


@test_utils.test(debug=True)
def test_copy_python_scope_matrix_to_quadrants_scope():
    a = qd.Vector([1, 2, 3])

    @qd.kernel
    def test():
        b = a
        assert b[0] == 1
        assert b[1] == 2
        assert b[2] == 3
        b = qd.Vector([4, 5, 6])
        assert b[0] == 4
        assert b[1] == 5
        assert b[2] == 6

    test()


@test_utils.test(debug=True)
def test_copy_matrix_field_element_to_quadrants_scope():
    a = qd.Vector.field(3, qd.i32, shape=())
    a[None] = qd.Vector([1, 2, 3])

    @qd.kernel
    def test():
        b = a[None]
        assert b[0] == 1
        assert b[1] == 2
        assert b[2] == 3
        b[0] = 5
        b[1] = 9
        b[2] = 7
        assert b[0] == 5
        assert b[1] == 9
        assert b[2] == 7
        assert a[None][0] == 1
        assert a[None][1] == 2
        assert a[None][2] == 3

    test()


@test_utils.test(debug=True)
def test_copy_matrix_in_quadrants_scope():
    @qd.kernel
    def test():
        a = qd.Vector([1, 2, 3])
        b = a
        assert b[0] == 1
        assert b[1] == 2
        assert b[2] == 3
        b[0] = 5
        b[1] = 9
        b[2] = 7
        assert b[0] == 5
        assert b[1] == 9
        assert b[2] == 7
        assert a[0] == 1
        assert a[1] == 2
        assert a[2] == 3

    test()


@test_utils.test(arch=[qd.cpu, qd.cuda], debug=True)
def test_matrix_field_dynamic_index_stride():
    # placeholders
    temp_a = qd.field(qd.f32)
    temp_b = qd.field(qd.f32)
    temp_c = qd.field(qd.f32)
    # target
    v = qd.Vector.field(3, qd.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)
    z = v.get_scalar_field(2)

    S0 = qd.root
    S1 = S0.pointer(qd.i, 4)
    S2 = S1.dense(qd.i, 2)
    S3 = S2.pointer(qd.i, 8)
    S3.place(temp_a)
    S4 = S2.dense(qd.i, 16)
    S4.place(x)
    S5 = S1.dense(qd.i, 2)
    S6 = S5.pointer(qd.i, 8)
    S6.place(temp_b)
    S7 = S5.dense(qd.i, 16)
    S7.place(y)
    S8 = S1.dense(qd.i, 2)
    S9 = S8.dense(qd.i, 32)
    S9.place(temp_c)
    S10 = S8.dense(qd.i, 16)
    S10.place(z)

    @qd.kernel
    def check_stride():
        for i in range(128):
            assert qd.get_addr(y, i) - qd.get_addr(x, i) == v._get_dynamic_index_stride()
            assert qd.get_addr(z, i) - qd.get_addr(y, i) == v._get_dynamic_index_stride()

    check_stride()

    @qd.kernel
    def run():
        for i in range(128):
            for j in range(3):
                v[i][j] = i * j

    run()
    for i in range(128):
        for j in range(3):
            assert v[i][j] == i * j


@test_utils.test()
def test_matrix_field_dynamic_index_different_path_length():
    v = qd.Vector.field(2, qd.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    qd.root.dense(qd.i, 8).place(x)
    qd.root.dense(qd.i, 2).dense(qd.i, 4).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_matrix_field_dynamic_index_not_pure_dense():
    v = qd.Vector.field(2, qd.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    qd.root.dense(qd.i, 2).pointer(qd.i, 4).place(x)
    qd.root.dense(qd.i, 2).dense(qd.i, 4).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test()
def test_matrix_field_dynamic_index_different_cell_size_bytes():
    temp = qd.field(qd.f32)

    v = qd.Vector.field(2, qd.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    qd.root.dense(qd.i, 8).place(x, temp)
    qd.root.dense(qd.i, 8).place(y)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test()
def test_matrix_field_dynamic_index_different_offset_bytes_in_parent_cell():
    temp_a = qd.field(qd.f32)
    temp_b = qd.field(qd.f32)

    v = qd.Vector.field(2, qd.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)

    qd.root.dense(qd.i, 8).place(temp_a, x)
    qd.root.dense(qd.i, 8).place(y, temp_b)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test()
def test_matrix_field_dynamic_index_different_stride():
    temp = qd.field(qd.f32)

    v = qd.Vector.field(3, qd.i32)
    x = v.get_scalar_field(0)
    y = v.get_scalar_field(1)
    z = v.get_scalar_field(2)

    qd.root.dense(qd.i, 8).place(x, y, temp, z)

    impl.get_runtime().materialize()
    assert v._get_dynamic_index_stride() is None


@test_utils.test()
def test_matrix_field_dynamic_index_multiple_materialize():
    @qd.kernel
    def empty():
        pass

    empty()

    n = 5
    a = qd.Vector.field(3, dtype=qd.i32, shape=n)

    @qd.kernel
    def func():
        for i in a:
            a[i][i % 3] = i

    func()
    for i in range(n):
        for j in range(3):
            assert a[i][j] == (i if j == i % 3 else 0)


@test_utils.test(debug=True)
def test_local_vector_initialized_in_a_loop():
    @qd.kernel
    def foo():
        for c in range(10):
            p = qd.Vector([c, c * 2])
            for i in range(2):
                assert p[i] == c * (i + 1)

    foo()


@test_utils.test(debug=True)
def test_vector_dtype():
    @qd.kernel
    def foo():
        a = qd.Vector([1, 2, 3], qd.f32)
        a /= 2
        assert all(abs(a - (0.5, 1.0, 1.5)) < 1e-6)
        b = qd.Vector([1.5, 2.5, 3.5], qd.i32)
        assert all(b == (1, 2, 3))

    foo()


@test_utils.test(debug=True)
def test_matrix_dtype():
    @qd.kernel
    def foo():
        a = qd.Matrix([[1, 2], [3, 4]], qd.f32)
        a /= 2
        assert all(abs(a - ((0.5, 1.0), (1.5, 2.0))) < 1e-6)
        b = qd.Matrix([[1.5, 2.5], [3.5, 4.5]], qd.i32)
        assert all(b == ((1, 2), (3, 4)))

    foo()


inplace_operation_types = [
    operator.iadd,
    operator.isub,
    operator.imul,
    operator.ifloordiv,
    operator.imod,
    operator.ilshift,
    operator.irshift,
    operator.ior,
    operator.ixor,
    operator.iand,
]


@test_utils.test()
def test_python_scope_inplace_operator():
    for ops in inplace_operation_types:
        a, b = test_matrix_arrays[:2]
        m1, m2 = qd.Matrix(a), qd.Matrix(b)
        m1 = ops(m1, m2)
        assert np.allclose(m1.to_numpy(), ops(a, b))


@test_utils.test(print_full_traceback=False)
def test_indexing():
    @qd.kernel
    def foo():
        m = qd.Matrix([[0.0, 0.0, 0.0, 0.0] for _ in range(4)])
        print(m[0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 2 indices, got 1"):
        foo()

    @qd.kernel
    def bar():
        vec = qd.Vector([1, 2, 3, 4])
        print(vec[0, 0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 1 indices, got 2"):
        bar()


@test_utils.test(print_full_traceback=False)
def test_indexing_in_fields():
    f = qd.Matrix.field(3, 3, qd.f32, shape=())

    @qd.kernel
    def foo():
        f[None][0, 0] = 1.0
        print(f[None][0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 2 indices, got 1"):
        foo()

    g = qd.Vector.field(3, qd.f32, shape=())

    @qd.kernel
    def bar():
        g[None][0] = 1.0
        print(g[None][0, 0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 1 indices, got 2"):
        bar()


@test_utils.test(print_full_traceback=False)
def test_indexing_in_struct():
    @qd.kernel
    def foo():
        s = qd.Struct(a=qd.Vector([0, 0, 0]), b=2)
        print(s.a[0, 0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 1 indices, got 2"):
        foo()

    @qd.kernel
    def bar():
        s = qd.Struct(m=qd.Matrix([[0, 0, 0], [0, 0, 0]]), n=2)
        print(s.m[0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 2 indices, got 1"):
        bar()


@test_utils.test(print_full_traceback=False)
def test_indexing_in_struct_field():
    s = qd.Struct.field({"v": qd.types.vector(3, qd.f32), "m": qd.types.matrix(3, 3, qd.f32)}, shape=())

    @qd.kernel
    def foo():
        print(s[None].v[0, 0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 1 indices, got 2"):
        foo()

    @qd.kernel
    def bar():
        print(s[None].m[0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 2 indices, got 1"):
        bar()


@test_utils.test(arch=get_host_arch_list(), debug=True)
def test_matrix_vector_multiplication():
    mat = qd.math.mat3(1)
    vec = qd.math.vec3(3)
    r = mat @ vec
    for i in range(3):
        assert r[i] == 9

    @qd.kernel
    def foo():
        mat = qd.math.mat3(1)
        vec = qd.math.vec3(3)
        r = mat @ vec
        assert r[0] == r[1] == r[2] == 9

    foo()


@test_utils.test(arch=[qd.cuda, qd.cpu], real_matrix_scalarize=False)
def test_local_matrix_read():
    s = qd.field(qd.i32, shape=())

    @qd.kernel
    def get_index(i: qd.i32, j: qd.i32):
        mat = qd.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        s[None] = mat[i, j]

    for i in range(3):
        for j in range(3):
            get_index(i, j)
            assert s[None] == i * 3 + j


@test_utils.test(arch=[qd.cuda, qd.cpu], real_matrix_scalarize=False)
def test_local_matrix_read_without_assign():
    @qd.kernel
    def local_vector_read(i: qd.i32) -> qd.i32:
        return qd.Vector([0, 1, 2])[i]

    for i in range(3):
        assert local_vector_read(i) == i


@test_utils.test(arch=[qd.cuda, qd.cpu], real_matrix_scalarize=False)
def test_local_matrix_indexing_in_loop():
    s = qd.field(qd.i32, shape=(3, 3))

    @qd.kernel
    def test():
        mat = qd.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        for i in range(3):
            for j in range(3):
                s[i, j] = mat[i, j] + 1

    test()
    for i in range(3):
        for j in range(3):
            assert s[i, j] == i * 3 + j + 1


@test_utils.test(arch=[qd.cuda, qd.cpu], real_matrix_scalarize=False)
def test_local_matrix_indexing_ops():
    @qd.kernel
    def element_write() -> qd.i32:
        mat = qd.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        s = 0
        for i in range(3):
            for j in range(3):
                mat[i, j] = 10
                s += mat[i, j]
        return s

    f = qd.field(qd.i32, shape=(3, 3))

    @qd.kernel
    def assign_from_index():
        mat = qd.Matrix([[x * 3 + y for y in range(3)] for x in range(3)])
        result = qd.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # TODO: fix parallelization
        qd.loop_config(serialize=True)
        for i in range(3):
            for j in range(3):
                result[i, j] = mat[j, i]
        for i in range(3):
            for j in range(3):
                f[i, j] = result[i, j]

    assert element_write() == 90
    assign_from_index()
    xs = [[x * 3 + y for y in range(3)] for x in range(3)]
    for i in range(3):
        for j in range(3):
            assert f[i, j] == xs[j][i]


@test_utils.test(print_full_traceback=False)
def test_local_matrix_index_check():
    @qd.kernel
    def foo():
        mat = qd.Matrix([[1, 2, 3], [4, 5, 6]])
        print(mat[0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 2 indices, got 1"):
        foo()

    @qd.kernel
    def bar():
        vec = qd.Vector([1, 2, 3, 4])
        print(vec[0, 0])

    with pytest.raises(QuadrantsCompilationError, match=r"Expected 1 indices, got 2"):
        bar()


@test_utils.test(arch=[qd.cuda, qd.cpu], real_matrix_scalarize=False, debug=True)
def test_elementwise_ops():
    @qd.kernel
    def test():
        # TODO: fix parallelization
        x = qd.Matrix([[1, 2], [3, 4]])
        # Unify rhs
        t1 = x + 10
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t1[i, j] == x[i, j] + 10
        t2 = x * 2
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t2[i, j] == x[i, j] * 2
        # elementwise-add
        t3 = t1 + t2
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t3[i, j] == t1[i, j] + t2[i, j]
        # Unify lhs
        t4 = 1 / t1
        # these should be *exactly* equals
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t4[i, j] == 1 / t1[i, j]
        t5 = 1 << x
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t5[i, j] == 1 << x[i, j]
        t6 = 1 + (x // 2)
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert t6[i, j] == 1 + (x[i, j] // 2)

        # test floordiv
        y = qd.Matrix([[1, 2], [3, 4]], dt=qd.i32)
        z = y * 2
        factors = z // y
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert factors[i, j] == 2

        y1 = qd.Matrix([[1, 2], [3, 4]], dt=qd.f32)
        z1 = y1 * 2
        factors1 = z1 // y1
        qd.loop_config(serialize=True)
        for i in range(2):
            for j in range(2):
                assert factors1[i, j] == 2

    test()


@test_utils.test(debug=True)
def test_local_matrix_scalarize():
    @qd.kernel
    def func():
        x = qd.Matrix([[1, 2], [3, 4]], qd.f32)

        # Store
        x[0, 0] = 100.0

        # Load + Store
        x[0, 1] = x[0, 0]

        # Binary
        x[1, 0] = x[0, 1] + x[0, 1]

        # Unary
        x[1, 1] = qd.sqrt(x[1, 0])

        assert x[0, 0] == 100.0
        assert x[0, 1] == 100.0
        assert x[1, 0] == 200.0
        assert x[1, 1] < 14.14214
        assert x[1, 1] > 14.14213

    func()


@test_utils.test()
def test_vector_vector_t():
    @qd.kernel
    def foo() -> qd.f32:
        a = qd.Vector([1.0, 2.0])
        b = qd.Vector([1.0, 2.0])
        return a @ b

    assert foo() == test_utils.approx(5.0)


def _test_field_and_ndarray(field, ndarray, func, verify):
    @qd.kernel
    def kern_field(a: qd.template()):
        func(a)

    @qd.kernel
    def kern_ndarray(a: qd.types.ndarray()):
        func(a)

    kern_field(field)
    verify(field)
    kern_ndarray(ndarray)
    verify(ndarray)


@test_utils.test()
def test_store_scalarize():
    @qd.func
    def func(a: qd.template()):
        for i in range(5):
            a[i] = [[i, i + 1], [i + 2, i + 3]]

    def verify(x):
        assert (x[0] == [[0, 1], [2, 3]]).all()
        assert (x[1] == [[1, 2], [3, 4]]).all()
        assert (x[2] == [[2, 3], [4, 5]]).all()
        assert (x[3] == [[3, 4], [5, 6]]).all()
        assert (x[4] == [[4, 5], [6, 7]]).all()

    field = qd.Matrix.field(2, 2, qd.i32, shape=5)
    ndarray = qd.Matrix.ndarray(2, 2, qd.i32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_load_store_scalarize():
    @qd.func
    def func(a: qd.template()):
        for i in range(3):
            a[i] = [[i, i + 1], [i + 2, i + 3]]

        a[3] = a[1]
        a[4] = a[2]

    def verify(x):
        assert (x[3] == [[1, 2], [3, 4]]).all()
        assert (x[4] == [[2, 3], [4, 5]]).all()

    field = qd.Matrix.field(2, 2, qd.i32, shape=5)
    ndarray = qd.Matrix.ndarray(2, 2, qd.i32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_load_broadcast():
    @qd.func
    def func(a: qd.template()):
        for i in qd.grouped(a):
            a[i] = 42

    def verify(x):
        for i in range(5):
            assert (x[i] == [[42, 42], [42, 42]]).all()

    field = qd.Matrix.field(2, 2, qd.i32, shape=5)
    ndarray = qd.Matrix.ndarray(2, 2, qd.i32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_unary_op_scalarize():
    @qd.func
    def func(a: qd.template()):
        a[0] = [[0, 1], [2, 3]]
        a[1] = [[3, 4], [5, 6]]
        a[2] = -a[0]
        a[3] = qd.exp(a[1])
        a[4] = qd.sqrt(a[3])

    def verify(x):
        assert (x[0] == [[0.0, 1.0], [2.0, 3.0]]).all()
        assert (x[1] == [[3.0, 4.0], [5.0, 6.0]]).all()
        assert (x[2] == [[-0.0, -1.0], [-2.0, -3.0]]).all()
        assert (x[3] < [[20.086, 54.60], [148.42, 403.43]]).all()
        assert (x[3] > [[20.085, 54.59], [148.41, 403.42]]).all()
        assert (x[4] < [[4.49, 7.39], [12.19, 20.09]]).all()
        assert (x[4] > [[4.48, 7.38], [12.18, 20.08]]).all()

    field = qd.Matrix.field(2, 2, qd.f32, shape=5)
    ndarray = qd.Matrix.ndarray(2, 2, qd.f32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test()
def test_binary_op_scalarize():
    @qd.func
    def func(a: qd.template()):
        a[0] = [[0.0, 1.0], [2.0, 3.0]]
        a[1] = [[3.0, 4.0], [5.0, 6.0]]
        a[2] = a[0] + a[0]
        a[3] = a[1] * a[1]
        a[4] = qd.max(a[2], a[3])

    def verify(x):
        assert (x[2] == [[0.0, 2.0], [4.0, 6.0]]).all()
        assert (x[3] == [[9.0, 16.0], [25.0, 36.0]]).all()
        assert (x[4] == [[9.0, 16.0], [25.0, 36.0]]).all()

    field = qd.Matrix.field(2, 2, qd.f32, shape=5)
    ndarray = qd.Matrix.ndarray(2, 2, qd.f32, shape=5)
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test(print_full_traceback=False)
def test_trace_op():
    @qd.kernel
    def test_fun() -> qd.f32:
        x = qd.Matrix([[0.1, 3.0], [5.0, 7.0]])
        return x.trace()

    assert np.abs(test_fun() - 7.1) < 1e-6

    x = qd.Matrix([[0.1, 3.0], [5.0, 7.0]])
    assert np.abs(x.trace() - 7.1) < 1e-6

    with pytest.raises(QuadrantsCompilationError, match=r"expected a square matrix, got shape \(3, 2\)"):
        x = qd.Matrix([[0.1, 3.0], [5.0, 7.0], [1.0, 2.0]])
        print(x.trace())

    @qd.kernel
    def failed_func():
        x = qd.Matrix([[0.1, 3.0], [5.0, 7.0], [1.0, 2.0]])
        print(x.trace())

    with pytest.raises(QuadrantsCompilationError, match=r"expected a square matrix, got shape \(3, 2\)"):
        failed_func()


@test_utils.test(debug=True)
def test_ternary_op_scalarize():
    @qd.kernel
    def test():
        cond = qd.Vector([1, 0, 1])
        x = qd.Vector([3, 3, 3])
        y = qd.Vector([5, 5, 5])

        z = qd.select(cond, x, y)

        assert z[0] == 3
        assert z[1] == 5
        assert z[2] == 3

    test()


@test_utils.test(debug=True)
def test_ternary_op_cond_is_scalar():
    @qd.kernel
    def test():
        x = qd.Vector([3, 3, 3])
        y = qd.Vector([5, 5, 5])

        for i in range(10):
            z = qd.select(i % 2, x, y)
            if i % 2 == 1:
                assert z[0] == x[0] and z[1] == x[1] and z[2] == x[2]
            else:
                assert z[0] == y[0] and z[1] == y[1] and z[2] == y[2]

    test()


@test_utils.test(debug=True)
def test_fill_op():
    @qd.kernel
    def test_fun():
        x = qd.Matrix([[0.0 for _ in range(4)] for _ in range(5)])
        x.fill(1.14)
        for i in qd.static(range(5)):
            for j in qd.static(range(4)):
                assert x[i, j] == 1.14

    test_fun()


@test_utils.test(debug=True)
def test_atomic_op_scalarize():
    @qd.func
    def func(x: qd.template()):
        x[0] = [1.0, 2.0, 3.0]
        tmp = qd.Vector([3, 2, 1])
        z = qd.atomic_sub(x[0], tmp)
        assert z[0] == 1.0
        assert z[1] == 2.0
        assert z[2] == 3.0

        # Broadcasting
        x[1] = [1.0, 1.0, 1.0]
        g = qd.atomic_add(x[1], 2)
        assert g[0] == 1.0
        assert g[1] == 1.0
        assert g[2] == 1.0

    def verify(x):
        assert (x[0] == [-2.0, 0.0, 2.0]).all()
        assert (x[1] == [3.0, 3.0, 3.0]).all()

    field = qd.Vector.field(n=3, dtype=qd.f32, shape=10)
    ndarray = qd.Vector.ndarray(n=3, dtype=qd.f32, shape=(10))
    _test_field_and_ndarray(field, ndarray, func, verify)


@test_utils.test(print_full_traceback=False)
def test_vector_transpose():
    @qd.kernel
    def foo():
        x = qd.Vector([1, 2])
        y = qd.Vector([3, 4])
        z = x @ y.transpose()

    with pytest.raises(
        QuadrantsCompilationError,
        match=r"`transpose\(\)` cannot apply to a vector. If you want something like `a @ b.transpose\(\)`, write `a.outer_product\(b\)` instead.",
    ):
        foo()


@test_utils.test(debug=True)
def test_cross_scope_matrix_binary_ops():
    n = 128
    x = qd.Vector.field(3, dtype=int, shape=(n, n))
    spring_offsets = [qd.Vector([1, 2]), qd.Vector([2, 3])]

    @qd.kernel
    def test():
        vec = qd.Vector([4, 5])
        ind0 = vec + qd.static(spring_offsets)[0]
        ind1 = qd.lang.ops.add(vec, qd.static(spring_offsets)[1])

        x[ind0] = [100, 10, 1]
        x[ind1] = [1, 10, 100]

    test()

    assert (x[5, 7] == [100, 10, 1]).all()
    assert (x[6, 8] == [1, 10, 100]).all()


@test_utils.test(debug=True)
def test_cross_scope_matrix_ternary_ops():
    n = 128
    x = qd.Vector.field(3, dtype=int, shape=(n, n))
    spring_offsets = [qd.Vector([1, 2]), qd.Vector([2, 3])]

    @qd.kernel
    def test():
        vec = qd.Vector([0, 1])
        ind0 = qd.select(vec, vec, qd.static(spring_offsets)[0])
        x[ind0] = [100, 10, 1]

    test()

    assert (x[1, 1] == [100, 10, 1]).all()


@test_utils.test(debug=True)
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "segfaults on Mac with multiprocess. Runs ok with -t 1 "
        "SHOULD PASS. Created "
        "https://linear.app/genesis-ai-company/issue/CMP-31/"
        "fix-failing-test-cross-scope-matrix-atomic-ops-on-mac-in-multiprocess"
        " to track"
    ),
)
def test_cross_scope_matrix_atomic_ops():
    n = 128
    x = qd.Vector.field(3, dtype=int, shape=(n, n))
    spring_offsets = [qd.Vector([1, 2]), qd.Vector([2, 3])]

    @qd.kernel
    def test():
        vec = qd.Vector([0, 1])
        vec += qd.static(spring_offsets)[0]
        x[vec] = [100, 10, 1]

    test()

    assert (x[1, 3] == [100, 10, 1]).all()


@test_utils.test(debug=True)
def test_global_tmp_overwrite():
    # https://github.com/taichi-dev/quadrants/issues/6663
    @qd.kernel
    def foo() -> qd.i32:
        p = qd.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        loop = 1
        sig = qd.Vector([0, 0, 0, 0])
        assert p[0, 0] == 1
        while loop == 1:
            assert p[0, 0] == 1
            loop = 0
            p[0, 0] = -1
        for i in range(1):
            sig[i] = 2
        return sig.sum() + p.sum()

    assert foo() == 4


@test_utils.test(debug=True)
def test_matrix_len():
    @qd.kernel
    def test():
        x = qd.Vector([1, 0])
        y = qd.Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

        assert len(x) == 2
        assert len(y) == 3

    test()


@test_utils.test()
def test_cross_scope_matrix():
    a = qd.Matrix([[1, 2], [3, 4]])

    @qd.kernel
    def foo() -> qd.types.vector(4, qd.i32):
        return qd.Vector([a[0, 0], a[0, 1], a[1, 0], a[1, 1]])

    assert (foo() == [1, 2, 3, 4]).all()


@test_utils.test(debug=True)
def test_matrix_type_inference():
    @qd.kernel
    def foo():
        a = qd.Vector([1, 2.5])[1]  # should be f32 instead of i32
        assert a == 2.5

    foo()


@test_utils.test(arch=[qd.cpu, qd.cuda], real_matrix_scalarize=False)
def test_matrix_arithmatics():
    f = qd.ndarray(qd.math.vec4, 4)

    @qd.kernel
    def fill(arr: qd.types.ndarray()):
        v0 = qd.math.vec4([0.0, 1.0, 2.0, 3.0])
        v1 = qd.math.vec4([1.0, 2.0, 3.0, 4.0])
        v2 = qd.math.vec4([2.0, 3.0, 4.0, 5.0])
        v3 = qd.math.vec4([4.0, 5.0, 6.0, 7.0])
        arr[0] = v0
        arr[1] = v1
        arr[2] = v2
        arr[3] = v3

    @qd.kernel
    def vec_test(arr: qd.types.ndarray()):
        v0 = arr[0]
        v1 = arr[1]
        v2 = arr[2]
        v3 = arr[3]

        arr[0] = v0 * v1 + v2
        arr[1] = v1 * v2 + v3
        arr[2] = v0 * v2 + v3

    fill(f)
    vec_test(f)

    assert (
        f.to_numpy()
        == np.array(
            [
                [2.0, 5.0, 10.0, 17.0],
                [6.0, 11.0, 18.0, 27.0],
                [4.0, 8.0, 14.0, 22.0],
                [4.0, 5.0, 6.0, 7.0],
            ]
        )
    ).all()


@pytest.mark.skipif(u.system == "linux" and u.machine in ("arm64", "aarch64"), reason="crashes on linux aarch64")
@test_utils.test(
    require=qd.extension.assertion,
    debug=True,
    check_out_of_bound=True,
    gdb_trigger=False,
)
def test_matrix_oob():
    @qd.kernel
    def access_vec(i: qd.i32):
        x = qd.Vector([1, 0])
        x[i] = 42

        # To keep x
        assert x[i] == 42

    @qd.kernel
    def access_mat(i: qd.i32, j: qd.i32):
        y = qd.Matrix([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        y[i, j] = 42

        # To keep y
        assert y[i, j] == 42

    # works
    access_vec(1)
    access_mat(2, 2)

    # vector overflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_vec(2)
    # vector underflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_vec(-1)

    # matrix overflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_mat(2, 3)
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_mat(3, 0)
    # matrix underflow
    with pytest.raises(AssertionError, match=r"Out of bound access"):
        access_mat(-1, 0)

    # TODO: As offset information per dimension is lacking, only the accumulated index is checked. These tests will not raise even if the individual indices are incorrect.
    # with pytest.raises(AssertionError, match=r"Out of bound access"):
    #    access_mat(0, 8)
    # with pytest.raises(AssertionError, match=r"Out of bound access"):
    #    access_mat(-1, 10)
    # with pytest.raises(AssertionError, match=r"Out of bound access"):
    #    access_mat(3, -1)


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.i64, qd.f64])
@pytest.mark.parametrize("shape", [(8,), (6, 12)])
@pytest.mark.parametrize("offset", [None, 0, -4, 4])
@pytest.mark.parametrize("m, n", [(3, 4)])
@test_utils.test(arch=get_host_arch_list(), debug=True)
def test_matrix_from_numpy_with_offset(dtype, shape, offset, m, n):
    import numpy as np

    x = qd.Matrix.field(
        dtype=dtype, m=m, n=n, shape=shape, offset=[offset] * len(shape) if offset is not None else None
    )

    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        qd.i32: np.int32,
        qd.f32: np.float32,
        qd.f64: np.float64,
        qd.i64: np.int64,
    }
    numpy_shape = ((shape,) if isinstance(shape, int) else shape) + (n, m)
    arr = np.ones(numpy_shape, dtype=numpy_dtypes[dtype])
    x.from_numpy(arr)

    @qd.kernel
    def func():
        for I in qd.grouped(x):
            pass
            # TODO: figure out the the purpose this assert (or of this test)
            # with debug off (as before) the failing assert is ignored
            # with debug on, it fires on all platforms
            # I suspect debug should be on
            # but this assert is clearly broken
            # assert all(abs(I - 1.0) < 1e-6)

    func()


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.i64, qd.f64])
@pytest.mark.parametrize("shape", [(8,), (6, 12)])
@pytest.mark.parametrize("offset", [0, -4, 4])
@pytest.mark.parametrize("m, n", [(3, 4)])
@test_utils.test(arch=get_host_arch_list())
def test_matrix_to_numpy_with_offset(dtype, shape, offset, m, n):
    import numpy as np

    x = qd.Matrix.field(dtype=dtype, m=m, n=n, shape=shape, offset=[offset] * len(shape))
    x.fill(1.0)
    # use the corresponding dtype for the numpy array.
    numpy_dtypes = {
        qd.i32: np.int32,
        qd.f32: np.float32,
        qd.f64: np.float64,
        qd.i64: np.int64,
    }
    numpy_shape = ((shape,) if isinstance(shape, int) else shape) + (n, m)
    arr = x.to_numpy()

    assert np.allclose(arr, np.ones(numpy_shape, dtype=numpy_dtypes[dtype]))


@test_utils.test()
def test_matrix_dtype():
    a = qd.types.vector(3, dtype=qd.f32)([0, 1, 2])
    assert a.entries.dtype == np.float32

    b = qd.types.matrix(2, 2, dtype=qd.i32)([[0, 1], [2, 3]])
    assert b.entries.dtype == np.int32


@test_utils.test()
def test_matrix_and_func():
    vec4d = qd.types.vector(4, float)
    v = vec4d(1, 2, 3, 4)

    @qd.func
    def length(w: vec4d):
        return w.norm()

    @qd.kernel
    def test() -> qd.f32:
        return length(v)

    approx(test(), 5.477226)


@test_utils.test()
def test_matrix_loop_unique():
    F_x = qd.Vector.field(3, dtype=qd.f32, shape=10)

    @qd.kernel
    def init():
        for u in F_x:
            F_x[u][1] += 1.0

    init()

    for u in range(10):
        assert F_x[u][1] == 1.0
