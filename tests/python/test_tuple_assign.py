import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests.helpers import test_utils


@test_utils.test()
def test_fibonacci():
    @qd.kernel
    def qd_fibonacci(n: qd.i32) -> qd.i32:
        a, b = 0, 1
        # This is to make the inner for loop serial on purpose...
        for _ in range(1):
            for i in range(n):
                a, b = b, a + b
        return b

    def py_fibonacci(n):
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return b

    for n in range(5):
        assert qd_fibonacci(n) == py_fibonacci(n)


@test_utils.test(arch=get_host_arch_list())
def test_assign2():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[()], b[()] = 2, 3

    func()
    assert a[()] == 2
    assert b[()] == 3


@test_utils.test(arch=get_host_arch_list())
def test_assign2_mismatch3():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[()], b[()] = 2, 3, 4

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_assign2_mismatch1():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[()], b[()] = 2

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_swap2():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[()], b[()] = b[()], a[()]

    a[()] = 2
    b[()] = 3
    func()
    assert a[()] == 3
    assert b[()] == 2


@test_utils.test(arch=get_host_arch_list())
def test_assign2_static():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        # XXX: why a, b = qd.static(b, a) doesn't work?
        c, d = qd.static(b, a)
        c[()], d[()] = 2, 3

    func()
    assert a[()] == 3
    assert b[()] == 2


@test_utils.test(arch=get_host_arch_list())
def test_swap3():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())
    c = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[()], b[()], c[()] = b[()], c[()], a[()]

    a[()] = 2
    b[()] = 3
    c[()] = 4
    func()
    assert a[()] == 3
    assert b[()] == 4
    assert c[()] == 2


@test_utils.test(arch=get_host_arch_list())
def test_unpack_from_tuple():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())
    c = qd.field(qd.f32, ())

    list = [2, 3, 4]

    @qd.kernel
    def func():
        a[()], b[()], c[()] = list

    func()
    assert a[()] == 2
    assert b[()] == 3
    assert c[()] == 4


@test_utils.test(arch=get_host_arch_list())
def test_unpack_mismatch_tuple():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    list = [2, 3, 4]

    @qd.kernel
    def func():
        a[()], b[()] = list

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_unpack_from_vector():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())
    c = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        vector = qd.Vector([2, 3, 4])
        a[()], b[()], c[()] = vector

    func()
    assert a[()] == 2
    assert b[()] == 3
    assert c[()] == 4


@test_utils.test(arch=get_host_arch_list())
def test_unpack_mismatch_vector():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        vector = qd.Vector([2, 3, 4])
        a[()], b[()] = vector

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_unpack_mismatch_type():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    bad = 12

    @qd.kernel
    def func():
        a[()], b[()] = bad

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list(), print_full_traceback=False)
def test_unpack_mismatch_matrix():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())
    c = qd.field(qd.f32, ())
    d = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        bad = qd.Matrix([[2, 3], [4, 5]])
        a[()], b[()], c[()], d[()] = bad

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_unpack_from_shape():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())
    c = qd.field(qd.f32, ())
    d = qd.field(qd.f32, (2, 3, 4))

    @qd.kernel
    def func():
        a[()], b[()], c[()] = d.shape

    func()
    assert a[()] == 2
    assert b[()] == 3
    assert c[()] == 4
