import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.sparse)
def test_pointer():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32, shape=())

    n = 16

    ptr = qd.root.pointer(qd.i, n)
    ptr.dense(qd.i, n).place(x)

    s[None] = 0

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[19] = 1
    func()
    assert s[None] == 32

    @qd.kernel
    def deactivate():
        qd.deactivate(ptr, 0)

    deactivate()
    s[None] = 0
    func()
    assert s[None] == 16


@test_utils.test(require=qd.extension.sparse)
def test_pointer1():
    x = qd.field(qd.f32)
    s = qd.field(qd.i32)

    n = 16

    ptr = qd.root.pointer(qd.i, n)
    ptr.dense(qd.i, n).place(x)
    qd.root.place(s)

    s[None] = 0

    @qd.kernel
    def func():
        for i in x:
            s[None] += 1

    x[0] = 1
    x[19] = 1
    x[20] = 1
    x[45] = 1
    func()
    assert s[None] == 48

    @qd.kernel
    def deactivate():
        qd.deactivate(ptr, 0)

    deactivate()
    s[None] = 0
    func()
    assert s[None] == 32


@test_utils.test(require=qd.extension.sparse)
def test_pointer2():
    x = qd.field(qd.f32)

    n = 16

    ptr = qd.root.pointer(qd.i, n)
    ptr.dense(qd.i, n).place(x)

    @qd.kernel
    def func():
        for i in range(n * n):
            x[i] = 1.0

    @qd.kernel
    def set10():
        x[10] = 10.0

    @qd.kernel
    def clear():
        for i in ptr:
            qd.deactivate(ptr, i)

    func()
    clear()

    for i in range(n * n):
        assert x[i] == 0.0

    set10()

    for i in range(n * n):
        if i != 10:
            assert x[i] == 0.0
        else:
            assert x[i] == 10.0


@test_utils.test(require=qd.extension.sparse)
def test_pointer3():
    x = qd.field(qd.f32)
    x_temp = qd.field(qd.f32)

    n = 16

    ptr1 = qd.root.pointer(qd.ij, n)
    ptr1.dense(qd.ij, n).place(x)
    ptr2 = qd.root.pointer(qd.ij, n)
    ptr2.dense(qd.ij, n).place(x_temp)

    @qd.kernel
    def fill():
        for j in range(n * n):
            for i in range(n * n):
                x[i, j] = i + j

    @qd.kernel
    def fill2():
        for i, j in x_temp:
            if x_temp[i, j] < 100:
                x[i, j] = x_temp[i, j]

    @qd.kernel
    def copy_to_temp():
        for i, j in x:
            x_temp[i, j] = x[i, j]

    @qd.kernel
    def copy_from_temp():
        for i, j in x_temp:
            x[i, j] = x_temp[i, j]

    @qd.kernel
    def clear():
        for i, j in ptr1:
            qd.deactivate(ptr1, [i, j])

    @qd.kernel
    def clear_temp():
        for i, j in ptr2:
            qd.deactivate(ptr2, [i, j])

    fill()
    copy_to_temp()
    clear()
    fill2()
    clear_temp()

    for itr in range(100):
        copy_to_temp()
        clear()
        copy_from_temp()
        clear_temp()

        xn = x.to_numpy()
        for j in range(n * n):
            for i in range(n * n):
                if i + j < 100:
                    assert xn[i, j] == i + j


@test_utils.test(require=qd.extension.sparse, exclude=[qd.metal])
def test_dynamic():
    x = qd.field(qd.i32)
    s = qd.field(qd.i32)

    n = 16

    lst = qd.root.dense(qd.i, n).dynamic(qd.j, 4096)
    lst.place(x)
    qd.root.dense(qd.i, n).place(s)

    @qd.kernel
    def func(mul: qd.i32):
        for i in range(n):
            for j in range(i * i * mul):
                qd.append(lst, i, j)

    @qd.kernel
    def fetch_length():
        for i in range(n):
            s[i] = qd.length(lst, i)

    func(1)
    fetch_length()
    for i in range(n):
        assert s[i] == i * i

    @qd.kernel
    def clear():
        for i in range(n):
            qd.deactivate(lst, [i])

    func(2)
    fetch_length()
    for i in range(n):
        assert s[i] == i * i * 3

    clear()
    fetch_length()
    for i in range(n):
        assert s[i] == 0

    func(4)
    fetch_length()
    for i in range(n):
        assert s[i] == i * i * 4
