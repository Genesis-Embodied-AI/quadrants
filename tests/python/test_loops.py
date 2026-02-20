import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_loops():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    N = 512

    qd.root.dense(qd.i, N).place(x)
    qd.root.dense(qd.i, N).place(y)
    qd.root.lazy_grad()

    for i in range(N // 2, N):
        y[i] = i - 300

    @qd.kernel
    def func():
        for i in range(qd.static(N // 2 + 3), N):
            x[i] = abs(y[i])

    func()

    for i in range(N // 2 + 3):
        assert x[i] == 0

    for i in range(N // 2 + 3, N):
        assert x[i] == abs(y[i])


@test_utils.test()
def test_numpy_loops():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    N = 512

    qd.root.dense(qd.i, N).place(x)
    qd.root.dense(qd.i, N).place(y)
    qd.root.lazy_grad()

    for i in range(N // 2, N):
        y[i] = i - 300

    import numpy as np

    begin = (np.ones(1) * (N // 2 + 3)).astype(np.int32).reshape(())
    end = (np.ones(1) * N).astype(np.int32).reshape(())

    @qd.kernel
    def func():
        for i in range(begin, end):
            x[i] = abs(y[i])

    func()

    for i in range(N // 2 + 3):
        assert x[i] == 0

    for i in range(N // 2 + 3, N):
        assert x[i] == abs(y[i])


@test_utils.test()
def test_nested_loops():
    # this may crash if any LLVM allocas are called in the loop body
    x = qd.field(qd.i32)

    n = 2048

    qd.root.dense(qd.ij, n).place(x)

    @qd.kernel
    def paint():
        for i in range(n):
            for j in range(n):
                x[0, 0] = i

    paint()


@test_utils.test()
def test_zero_outer_loop():
    x = qd.field(qd.i32, shape=())

    @qd.kernel
    def test():
        for i in range(0):
            x[None] = 1

    test()

    assert x[None] == 0


@test_utils.test()
def test_zero_inner_loop():
    x = qd.field(qd.i32, shape=())

    @qd.kernel
    def test():
        for i in range(1):
            for j in range(0):
                x[None] = 1

    test()

    assert x[None] == 0


@test_utils.test()
def test_dynamic_loop_range():
    x = qd.field(qd.i32)
    c = qd.field(qd.i32)
    n = 2000

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(c)

    @qd.kernel
    def test():
        for i in x:
            x[i] = qd.atomic_add(c[None], 1)
        for i in range(c[None], c[None] * 2):
            x[i - n] += c[None]

    test()
    assert c[None] == n
    assert sum(x.to_numpy()) == (n * (n - 1) // 2) + n * n


@test_utils.test()
def test_loop_arg_as_range():
    # Dynamic range loops are intended to make sure global tmps work
    x = qd.field(qd.i32)
    n = 1000

    qd.root.dense(qd.i, n).place(x)

    @qd.kernel
    def test(b: qd.i32, e: qd.i32):
        for i in range(b, e):
            x[i - b] = i

    pairs = [
        (0, n // 2),
        (n // 2, n),
        (-n // 2, -n // 3),
    ]
    for b, e in pairs:
        test(b, e)
        for i in range(b, e):
            assert x[i - b] == i


@test_utils.test()
def test_assignment_in_nested_loops():
    # https://github.com/taichi-dev/quadrants/issues/1109
    m = qd.field(qd.f32, 3)
    x = qd.field(qd.f32, ())

    @qd.kernel
    def func():
        a = x[None]
        for i in m:
            b = a
            for j in range(1):
                b = b
            x[None] = b

    x[None] = 1
    func()
    assert x[None] == 1


@test_utils.test(print_full_traceback=False)
def test_break_in_outermost_for_not_in_outermost_scope():
    @qd.kernel
    def foo() -> qd.i32:
        a = 0
        if True:
            for i in range(1000):
                if i == 100:
                    break
                a += 1
        return a

    assert foo() == 100


@test_utils.test()
def test_cache_loop_invariant_global_var_in_nested_loops():
    p = qd.field(float, 1)

    @qd.kernel
    def k():
        for n in range(1):
            for t in range(2):
                for m in range(1):
                    p[n] = p[n] + 1.0
                p[n] = p[n] + 1.0

    k()
    assert p[0] == 4.0
