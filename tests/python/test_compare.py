import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


@test_utils.test(require=qd.extension.sparse, exclude=qd.metal)
def test_compare_basics():
    a = qd.field(qd.i32)
    qd.root.dynamic(qd.i, 256).place(a)
    b = qd.field(qd.i32, shape=())
    c = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        b[None] = 3
        c[None] = 5
        a[0] = b[None] < c[None]
        a[1] = b[None] <= c[None]
        a[2] = b[None] > c[None]
        a[3] = b[None] >= c[None]
        a[4] = b[None] == c[None]
        a[5] = b[None] != c[None]
        a[6] = c[None] < b[None]
        a[7] = c[None] <= b[None]
        a[8] = c[None] > b[None]
        a[9] = c[None] >= b[None]
        a[10] = c[None] == b[None]
        a[11] = c[None] != b[None]

    func()
    assert a[0]
    assert a[1]
    assert not a[2]
    assert not a[3]
    assert not a[4]
    assert a[5]
    assert not a[6]
    assert not a[7]
    assert a[8]
    assert a[9]
    assert not a[10]
    assert a[11]


@test_utils.test(require=qd.extension.sparse, exclude=qd.metal)
def test_compare_equality():
    a = qd.field(qd.i32)
    qd.root.dynamic(qd.i, 256).place(a)
    b = qd.field(qd.i32, shape=())
    c = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        b[None] = 3
        c[None] = 3
        a[0] = b[None] < c[None]
        a[1] = b[None] <= c[None]
        a[2] = b[None] > c[None]
        a[3] = b[None] >= c[None]
        a[4] = b[None] == c[None]
        a[5] = b[None] != c[None]
        a[6] = c[None] < b[None]
        a[7] = c[None] <= b[None]
        a[8] = c[None] > b[None]
        a[9] = c[None] >= b[None]
        a[10] = c[None] == b[None]
        a[11] = c[None] != b[None]

    func()
    assert not a[0]
    assert a[1]
    assert not a[2]
    assert a[3]
    assert a[4]
    assert not a[5]
    assert not a[6]
    assert a[7]
    assert not a[8]
    assert a[9]
    assert a[10]
    assert not a[11]


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_no_duplicate_eval():
    a = qd.field(qd.i32)
    qd.root.dynamic(qd.i, 256).place(a)

    @qd.kernel
    def func():
        a[2] = 0 <= qd.append(a.parent(), [], 10) < 1

    func()
    assert a[0] == 10
    assert a[1] == 0  # not appended twice
    assert a[2]  # qd.append returns 0


@test_utils.test()
def test_no_duplicate_eval_func():
    a = qd.field(qd.i32, ())
    b = qd.field(qd.i32, ())

    @qd.func
    def why_this_foo_fail(n):
        return qd.atomic_add(b[None], n)

    def foo(n):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        return qd.atomic_add(impl.subscript(ast_builder, b, None), n)

    @qd.kernel
    def func():
        a[None] = 0 <= foo(2) < 1

    func()
    assert a[None] == 1
    assert b[None] == 2


@test_utils.test(require=qd.extension.sparse, exclude=qd.metal)
def test_chain_compare():
    a = qd.field(qd.i32)
    qd.root.dynamic(qd.i, 256).place(a)
    b = qd.field(qd.i32, shape=())
    c = qd.field(qd.i32, shape=())
    d = qd.field(qd.i32, shape=())

    @qd.kernel
    def func():
        b[None] = 2
        c[None] = 3
        d[None] = 3
        a[0] = c[None] == d[None] != b[None] < d[None] > b[None] >= b[None] <= c[None]
        a[1] = b[None] <= c[None] != d[None] > b[None] == b[None]

    func()
    assert a[0]
    assert not a[1]


@test_utils.test()
def test_static_in():
    @qd.kernel
    def foo(a: qd.template()) -> qd.i32:
        b = 0
        if qd.static(a in [qd.i32, qd.u32]):
            b = 1
        elif qd.static(a not in [qd.f32, qd.f64]):
            b = 2
        return b

    assert foo(qd.u32) == 1
    assert foo(qd.i64) == 2
    assert foo(qd.f32) == 0


@test_utils.test()
def test_non_static_in():
    with pytest.raises(qd.QuadrantsCompilationError, match='"In" is only supported inside `qd.static`.'):

        @qd.kernel
        def foo(a: qd.template()) -> qd.i32:
            b = 0
            if a in [qd.i32, qd.u32]:
                b = 1
            return b

        foo(qd.i32)


@test_utils.test(default_ip=qd.i64, require=qd.extension.data64)
def test_compare_ret_type():
    # The purpose of this test is to make sure a comparison returns i32
    # regardless of default_ip so that it can always serve as the condition of
    # an if/while statement.
    @qd.kernel
    def foo():
        for i in range(100):
            if i == 0:
                pass
        i = 100
        while i != 0:
            i -= 1

    foo()


@test_utils.test()
def test_python_scope_compare():
    v = qd.math.vec3(0, 1, 2)
    assert (v < 1)[0] == 1
