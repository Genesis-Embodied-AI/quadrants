import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_explicit_local_atomic_add():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 0
        for i in range(10):
            qd.atomic_add(a, i)
        A[None] = a

    func()
    assert A[None] == 45


@test_utils.test()
def test_implicit_local_atomic_add():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 0
        for i in range(10):
            a += i
        A[None] = a

    func()
    assert A[None] == 45


@test_utils.test()
def test_explicit_local_atomic_sub():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 0
        for i in range(10):
            qd.atomic_sub(a, i)
        A[None] = a

    func()
    assert A[None] == -45


@test_utils.test()
def test_implicit_local_atomic_sub():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 0
        for i in range(10):
            a -= i
        A[None] = a

    func()
    assert A[None] == -45


@test_utils.test()
def test_explicit_local_atomic_min():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = 1000
        for i in range(10):
            qd.atomic_min(a, i)
        A[None] = a

    func()
    assert A[None] == 0


@test_utils.test()
def test_explicit_local_atomic_max():
    A = qd.field(qd.f32, shape=())

    @qd.kernel
    def func():
        a = -1000
        for i in range(10):
            qd.atomic_max(a, i)
        A[None] = a

    func()
    assert A[None] == 9


@test_utils.test()
def test_explicit_local_atomic_and():
    A = qd.field(qd.i32, shape=())
    max_int = 2147483647

    @qd.kernel
    def func():
        a = 1023
        for i in range(10):
            qd.atomic_and(a, max_int - 2**i)
        A[None] = a

    func()
    assert A[None] == 0


@test_utils.test()
def test_implicit_local_atomic_and():
    A = qd.field(qd.i32, shape=())
    max_int = 2147483647

    @qd.kernel
    def func():
        a = 1023
        for i in range(10):
            a &= max_int - 2**i
        A[None] = a

    func()
    assert A[None] == 0


@test_utils.test()
def test_explicit_local_atomic_or():
    A = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = 0
        for i in range(10):
            qd.atomic_or(a, 2**i)
        A[None] = a

    func()
    assert A[None] == 1023


@test_utils.test()
def test_implicit_local_atomic_or():
    A = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = 0
        for i in range(10):
            a |= 2**i
        A[None] = a

    func()
    assert A[None] == 1023


@test_utils.test()
def test_explicit_local_atomic_xor():
    A = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = 1023
        for i in range(10):
            qd.atomic_xor(a, 2**i)
        A[None] = a

    func()
    assert A[None] == 0


@test_utils.test()
def test_implicit_local_atomic_xor():
    A = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        a = 1023
        for i in range(10):
            a ^= 2**i
        A[None] = a

    func()
    assert A[None] == 0
