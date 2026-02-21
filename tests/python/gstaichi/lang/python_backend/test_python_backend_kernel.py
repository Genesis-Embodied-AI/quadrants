import pytest

import quadrants as qd


def test_python_backend_kernel() -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo2(a: qd.types.ndarray[qd.i32, 1]):
        a[0] = 5

    a = qd.ndarray(qd.i32, (10,))
    a[0] = 3
    foo2(a)
    assert a[0] == 5

    @qd.kernel
    def foo3(a: qd.types.ndarray[qd.i32, 1]):
        B = a.shape[0]
        for i_b in range(B):
            a[i_b] += 1

    a.fill(7)
    foo3(a)
    assert a[4] == 8

    @qd.kernel
    def foo4(a: qd.types.ndarray[qd.i32, 1]):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] += 1

    a.fill(3)
    foo4(a)
    assert a[2] == 4

    @qd.kernel(fastcache=True)
    def k1b(a: qd.types.ndarray[qd.i32, 1]):
        a[2] = 7

    a[2] = 9
    k1b(a)
    assert a[2] == 7


@pytest.mark.parametrize(
    "atomic_func,python_func",
    [
        (qd.atomic_add, "x + y"),
        (qd.atomic_sub, "x - y"),
        (qd.atomic_or, "x | y"),
        (qd.atomic_and, "x & y"),
        (qd.atomic_mul, "x * y"),
        (qd.atomic_max, "max(x, y)"),
        (qd.atomic_min, "min(x, y)"),
    ],
)
def test_python_backend_atomics(atomic_func, python_func) -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo5(a: qd.types.ndarray[qd.i32, 1]):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            atomic_func(a[i_b], 5)

    N = 10
    a = qd.ndarray(qd.i32, (N,))
    for i in range(N):
        a[i] = i
    foo5(a)
    for i in range(N):
        expected = eval(python_func, {}, {"x": i, "y": 5})
        assert a[i] == expected


@pytest.mark.parametrize(
    "static_flag,expected_output",
    [
        (False, 3),
        (True, 5),
    ],
)
def test_python_backend_static(static_flag: bool, expected_output: int) -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo5(static_flag: qd.template(), a: qd.types.ndarray[qd.i32, 1]):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            if qd.static(static_flag):
                a[i_b] = 5
            else:
                a[i_b] = 3

    N = 10
    a = qd.ndarray(qd.i32, (N,))
    a.fill(0)
    foo5(static_flag, a)
    assert a[2] == expected_output


def test_python_backend_field() -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo5(a: qd.template()):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] = 5

    N = 10
    a = qd.field(qd.i32, (N,))
    a.fill(0)
    foo5(a)
    assert a[2] == 5


def test_python_backend_field_vector() -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo5(a: qd.template()):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] = qd.Vector([5, 4, 3])

    N = 10
    vec3 = qd.types.vector(3, qd.i32)
    a = qd.field(vec3, (N,))
    a.fill(0)
    foo5(a)
    assert a[2].tolist() == [5, 4, 3]


def test_python_backend_field_matrix() -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo5(a: qd.template()):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] = qd.Matrix([[5, 4, 3], [2, 3, 4]])

    N = 10
    mat2x3 = qd.types.matrix(2, 3, qd.i32)
    a = qd.field(mat2x3, (N,))
    a.fill(0)
    foo5(a)
    assert a[2].tolist() == [[5, 4, 3], [2, 3, 4]]


def test_python_backend_ndarray_vector() -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo5(a: qd.types.ndarray()):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] = qd.Vector([5, 4, 3])

    N = 10
    vec3 = qd.types.vector(3, qd.i32)
    a = qd.ndarray(vec3, (N,))
    a.fill(0)
    foo5(a)
    assert a[2].tolist() == [5, 4, 3]


def test_python_backend_ndarray_matrix() -> None:
    qd.init(qd.python)

    @qd.kernel
    def foo5(a: qd.types.ndarray()):
        B = a.shape[0]
        qd.loop_config(serialize=False)
        for i_b in range(B):
            a[i_b] = qd.Matrix([[5, 4, 3], [2, 3, 4]])

    N = 10
    mat2x3 = qd.types.matrix(2, 3, qd.i32)
    a = qd.ndarray(mat2x3, (N,))
    a.fill(0)
    foo5(a)
    assert a[2].tolist() == [[5, 4, 3], [2, 3, 4]]
