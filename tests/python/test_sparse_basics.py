import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse)
def test_pointer():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    qd.root.pointer(qd.i, n).dense(qd.i, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[256] = 1

    func()
    assert s[None] == 256


@test_utils.test(require=qd.extension.sparse)
def test_pointer_is_active():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    ptr = qd.root.pointer(qd.i, n)
    ptr.dense(qd.i, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def func():
        for i in range(n * n):
            s[None] += qd.is_active(ptr, qd.rescale_index(x, ptr, [i]))

    x[0] = 1
    x[127] = 1
    x[256] = 1

    func()
    assert s[None] == 256


@test_utils.test(require=qd.extension.sparse)
def test_pointer_is_active_2():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    qd.root.dense(qd.i, n).pointer(qd.j, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def func():
        for i, j in qd.ndrange(n, n):
            s[None] += qd.is_active(x.parent(), [i, j])

    x[0, 0] = 1
    x[0, 127] = 1
    x[127, 127] = 1

    func()
    assert s[None] == 3


@test_utils.test(require=qd.extension.sparse)
def test_pointer2():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 128

    qd.root.pointer(qd.i, n).pointer(qd.i, n).dense(qd.i, n).place(x)
    qd.root.place(s)

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[127] = 1
    x[254] = 1
    x[256 + n * n] = 1

    x[257 + n * n] = 1
    x[257 + n * n * 2] = 1
    x[257 + n * n * 5] = 1

    func()
    assert s[None] == 5 * n
    print(x[257 + n * n * 7])
    assert s[None] == 5 * n


@pytest.mark.skip(reason="https://github.com/taichi-dev/quadrants/issues/2520")
@test_utils.test(require=qd.extension.sparse)
def test_pointer_direct_place():
    x, y = qd.field(qd.i32), qd.field(qd.i32)

    N = 1
    qd.root.pointer(qd.i, N).place(x)
    qd.root.pointer(qd.i, N).place(y)

    @qd.kernel
    def foo():
        pass

    foo()
