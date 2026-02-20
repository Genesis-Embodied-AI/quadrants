import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test()
def test_fibonacci():
    @qd.kernel
    def ti_fibonacci(n: qd.i32) -> qd.i32:
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
        assert ti_fibonacci(n) == py_fibonacci(n)


@test_utils.test(arch=get_host_arch_list())
def test_assign2():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[None], b[None] = 2, 3

    func()
    assert a[None] == 2
    assert b[None] == 3


@test_utils.test(arch=get_host_arch_list())
def test_assign2_mismatch3():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[None], b[None] = 2, 3, 4

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_assign2_mismatch1():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[None], b[None] = 2

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_swap2():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[None], b[None] = b[None], a[None]

    a[None] = 2
    b[None] = 3
    func()
    assert a[None] == 3
    assert b[None] == 2


@test_utils.test(arch=get_host_arch_list())
def test_assign2_static():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        # XXX: why a, b = qd.static(b, a) doesn't work?
        c, d = qd.static(b, a)
        c[None], d[None] = 2, 3

    func()
    assert a[None] == 3
    assert b[None] == 2


@test_utils.test(arch=get_host_arch_list())
def test_swap3():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())
    c = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a[None], b[None], c[None] = b[None], c[None], a[None]

    a[None] = 2
    b[None] = 3
    c[None] = 4
    func()
    assert a[None] == 3
    assert b[None] == 4
    assert c[None] == 2


@test_utils.test(arch=get_host_arch_list())
def test_unpack_from_tuple():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())
    c = qd.field(qd.f32, ())

    list = [2, 3, 4]

    @qd.kernel
    def func():
        a[None], b[None], c[None] = list

    func()
    assert a[None] == 2
    assert b[None] == 3
    assert c[None] == 4


@test_utils.test(arch=get_host_arch_list())
def test_unpack_mismatch_tuple():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    list = [2, 3, 4]

    @qd.kernel
    def func():
        a[None], b[None] = list

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
        a[None], b[None], c[None] = vector

    func()
    assert a[None] == 2
    assert b[None] == 3
    assert c[None] == 4


@test_utils.test(arch=get_host_arch_list())
def test_unpack_mismatch_vector():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        vector = qd.Vector([2, 3, 4])
        a[None], b[None] = vector

    with pytest.raises(qd.QuadrantsCompilationError):
        func()


@test_utils.test(arch=get_host_arch_list())
def test_unpack_mismatch_type():
    a = qd.field(qd.f32, ())
    b = qd.field(qd.f32, ())

    bad = 12

    @qd.kernel
    def func():
        a[None], b[None] = bad

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
        a[None], b[None], c[None], d[None] = bad

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
        a[None], b[None], c[None] = d.shape

    func()
    assert a[None] == 2
    assert b[None] == 3
    assert c[None] == 4
