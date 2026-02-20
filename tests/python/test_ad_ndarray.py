import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsRuntimeError

from tests import test_utils

archs_support_ndarray_ad = [qd.cpu, qd.cuda]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_sum():

    N = 10

    @qd.kernel
    def compute_sum(a: qd.types.ndarray(), b: qd.types.ndarray(), p: qd.types.ndarray()):
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret + a[i]
            p[i] = ret

    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.i32, shape=N)
    p = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    for i in range(N):
        a[i] = 3
        b[i] = i

    compute_sum(a, b, p)

    for i in range(N):
        assert p[i] == a[i] * b[i] + 1
        p.grad[i] = 1

    compute_sum.grad(a, b, p)

    for i in range(N):
        assert a.grad[i] == b[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_sum_local_atomic():

    N = 10
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.i32, shape=N)
    p = qd.ndarray(qd.f32, shape=N, needs_grad=True)

    @qd.kernel
    def compute_sum(a: qd.types.ndarray(), b: qd.types.ndarray(), p: qd.types.ndarray()):
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret += a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    compute_sum(a, b, p)

    for i in range(N):
        assert p[i] == 3 * b[i] + 1
        p.grad[i] = 1

    compute_sum.grad(a, b, p)

    for i in range(N):
        assert a.grad[i] == b[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_power():
    N = 10
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.i32, shape=N)
    p = qd.ndarray(qd.f32, shape=N, needs_grad=True)

    @qd.kernel
    def power(a: qd.types.ndarray(), b: qd.types.ndarray(), p: qd.types.ndarray()):
        for i in range(N):
            ret = 1.0
            for j in range(b[i]):
                ret = ret * a[i]
            p[i] = ret

    for i in range(N):
        a[i] = 3
        b[i] = i

    power(a, b, p)

    for i in range(N):
        assert p[i] == 3 ** b[i]
        p.grad[i] = 1

    power.grad(a, b, p)

    for i in range(N):
        assert a.grad[i] == b[i] * 3 ** (b[i] - 1)


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_ad_fibonacci():
    N = 15
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    c = qd.ndarray(qd.i32, shape=N)
    f = qd.ndarray(qd.f32, shape=N, needs_grad=True)

    @qd.kernel
    def fib(a: qd.types.ndarray(), b: qd.types.ndarray(), c: qd.types.ndarray(), f: qd.types.ndarray()):
        for i in range(N):
            p = a[i]
            q = b[i]
            for j in range(c[i]):
                p, q = q, p + q
            f[i] = q

    b.fill(1)

    for i in range(N):
        c[i] = i

    fib(a, b, c, f)

    for i in range(N):
        f.grad[i] = 1

    fib.grad(a, b, c, f)

    for i in range(N):
        print(a.grad[i], b.grad[i])
        if i == 0:
            assert a.grad[i] == 0
        else:
            assert a.grad[i] == f[i - 1]
        assert b.grad[i] == f[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f32, require=qd.extension.adstack)
def test_ad_fibonacci_index():
    N = 5
    M = 10
    a = qd.ndarray(qd.f32, shape=M, needs_grad=True)
    b = qd.ndarray(qd.f32, shape=M, needs_grad=True)
    f = qd.ndarray(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def fib(a: qd.types.ndarray(), b: qd.types.ndarray(), f: qd.types.ndarray()):
        for i in range(N):
            p = 0
            q = 1
            for j in range(5):
                p, q = q, p + q
                b[q] += a[q]

        for i in range(M):
            f[None] += b[i]

    f.grad[None] = 1
    a.fill(1)

    fib(a, b, f)
    fib.grad(a, b, f)

    for i in range(M):
        is_fib = int(i in [1, 2, 3, 5, 8])
        assert a.grad[i] == is_fib * N
        assert b[i] == is_fib * N


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_integer_stack():
    N = 5
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    c = qd.ndarray(qd.i32, shape=N)
    f = qd.ndarray(qd.f32, shape=N, needs_grad=True)

    @qd.kernel
    def int_stack(a: qd.types.ndarray(), b: qd.types.ndarray(), c: qd.types.ndarray(), f: qd.types.ndarray()):
        for i in range(N):
            weight = 1
            s = 0.0
            for j in range(c[i]):
                s += weight * a[i] + b[i]
                weight *= 10
            f[i] = s

    a.fill(1)
    b.fill(1)

    for i in range(N):
        c[i] = i

    int_stack(a, b, c, f)

    for i in range(N):
        print(f[i])
        f.grad[i] = 1

    int_stack.grad(a, b, c, f)

    t = 0
    for i in range(N):
        assert a.grad[i] == t
        assert b.grad[i] == i
        t = t * 10 + 1


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_double_for_loops():
    N = 5
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    c = qd.ndarray(qd.i32, shape=N)
    f = qd.ndarray(qd.f32, shape=N, needs_grad=True)

    @qd.kernel
    def double_for(a: qd.types.ndarray(), b: qd.types.ndarray(), c: qd.types.ndarray(), f: qd.types.ndarray()):
        for i in range(N):
            weight = 1.0
            for j in range(c[i]):
                weight *= a[i]
            s = 0.0
            for j in range(c[i] * 2):
                s += weight + b[i]
            f[i] = s

    a.fill(2)
    b.fill(1)

    for i in range(N):
        c[i] = i

    double_for(a, b, c, f)

    for i in range(N):
        assert f[i] == 2 * i * (1 + 2**i)
        f.grad[i] = 1

    double_for.grad(a, b, c, f)

    for i in range(N):
        assert a.grad[i] == 2 * i * i * 2 ** (i - 1)
        assert b.grad[i] == 2 * i


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_double_for_loops_more_nests():
    N = 6
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    c = qd.ndarray(qd.i32, shape=(N, N // 2))
    f = qd.ndarray(qd.f32, shape=(N, N // 2), needs_grad=True)

    @qd.kernel
    def double_for(a: qd.types.ndarray(), b: qd.types.ndarray(), c: qd.types.ndarray(), f: qd.types.ndarray()):
        for i in range(N):
            for k in range(N // 2):
                weight = 1.0
                for j in range(c[i, k]):
                    weight *= a[i]
                s = 0.0
                for j in range(c[i, k] * 2):
                    s += weight + b[i]
                f[i, k] = s

    a.fill(2)
    b.fill(1)

    for i in range(N):
        for k in range(N // 2):
            c[i, k] = i + k

    double_for(a, b, c, f)

    for i in range(N):
        for k in range(N // 2):
            assert f[i, k] == 2 * (i + k) * (1 + 2 ** (i + k))
            f.grad[i, k] = 1

    double_for.grad(a, b, c, f)

    for i in range(N):
        total_grad_a = 0
        total_grad_b = 0
        for k in range(N // 2):
            total_grad_a += 2 * (i + k) ** 2 * 2 ** (i + k - 1)
            total_grad_b += 2 * (i + k)
        assert a.grad[i] == total_grad_a
        assert b.grad[i] == total_grad_b


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_complex_body():
    N = 5
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    c = qd.ndarray(qd.i32, shape=N)
    f = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    g = qd.ndarray(qd.f32, shape=N, needs_grad=True)

    @qd.kernel
    def complex(a: qd.types.ndarray(), c: qd.types.ndarray(), f: qd.types.ndarray(), g: qd.types.ndarray()):
        for i in range(N):
            weight = 2.0
            tot = 0.0
            tot_weight = 0.0
            for j in range(c[i]):
                tot_weight += weight + 1
                tot += (weight + 1) * a[i]
                weight = weight + 1
                weight = weight * 4
                weight = qd.cast(weight, qd.f64)
                weight = qd.cast(weight, qd.f32)

            g[i] = tot_weight
            f[i] = tot

    a.fill(2)

    for i in range(N):
        c[i] = i
        f.grad[i] = 1

    complex(a, c, f, g)
    complex.grad(a, c, f, g)

    for i in range(N):
        print(a.grad.to_numpy())
        # assert a.grad[i] == g[i]


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_mixed_inner_loops():
    x = qd.ndarray(dtype=qd.f32, shape=(1,), needs_grad=True)
    arr = qd.ndarray(dtype=qd.f32, shape=(5))
    loss = qd.ndarray(dtype=qd.f32, shape=(1,), needs_grad=True)

    @qd.kernel
    def mixed_inner_loops(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in arr:
            loss[0] += qd.sin(x[0])
            for j in range(2):
                loss[0] += qd.sin(x[0]) + 1.0

    loss.grad[0] = 1.0
    x[0] = 0.0
    mixed_inner_loops(x, arr, loss)
    mixed_inner_loops.grad(x, arr, loss)

    assert loss[0] == 10.0
    assert x.grad[0] == 15.0


@test_utils.test(arch=archs_support_ndarray_ad, default_fp=qd.f64, require=qd.extension.adstack)
def test_mixed_inner_loops_tape():
    x = qd.ndarray(dtype=qd.f32, shape=(1,), needs_grad=True)
    arr = qd.ndarray(dtype=qd.f32, shape=(5))
    loss = qd.ndarray(dtype=qd.f32, shape=(1,), needs_grad=True)

    @qd.kernel
    def mixed_inner_loops_tape(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in arr:
            loss[0] += qd.sin(x[0])
            for j in range(2):
                loss[0] += qd.sin(x[0]) + 1.0

    x[0] = 0.0
    with qd.ad.Tape(loss=loss):
        mixed_inner_loops_tape(x, arr, loss)
    assert loss[0] == 10.0
    assert x.grad[0] == 15.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=32)
def test_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = qd.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(1), needs_grad=True)

    @qd.kernel
    def test_inner_loops_local_variable(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += qd.sin(x[0]) + 1.0
                    t += qd.sin(x[0])
                loss[0] += s + t

    loss.grad[0] = 1.0
    x[0] = 0.0
    test_inner_loops_local_variable(x, arr, loss)
    test_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[0] == 18.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=0)
def test_inner_loops_local_variable_adaptive_stack_size_tape():
    x = qd.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(1), needs_grad=True)

    @qd.kernel
    def test_inner_loops_local_variable(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in arr:
            for j in range(3):
                s = 0.0
                t = 0.0
                for k in range(3):
                    s += qd.sin(x[0]) + 1.0
                    t += qd.sin(x[0])
                loss[0] += s + t

    x[0] = 0.0
    with qd.ad.Tape(loss=loss):
        test_inner_loops_local_variable(x, arr, loss)

    assert loss[0] == 18.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=0)
def test_more_inner_loops_local_variable_adaptive_stack_size_tape():
    x = qd.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(1), needs_grad=True)

    @qd.kernel
    def test_more_inner_loops_local_variable(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in arr:
            for j in range(2):
                s = 0.0
                for k in range(3):
                    u = 0.0
                    s += qd.sin(x[0]) + 1.0
                    for l in range(2):
                        u += qd.sin(x[0])
                    loss[0] += u
                loss[0] += s

    x[0] = 0.0
    with qd.ad.Tape(loss=loss):
        test_more_inner_loops_local_variable(x, arr, loss)

    assert loss[0] == 12.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=32)
def test_more_inner_loops_local_variable_fixed_stack_size_tape():
    x = qd.ndarray(dtype=float, shape=(1), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(1), needs_grad=True)

    @qd.kernel
    def test_more_inner_loops_local_variable(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in arr:
            for j in range(2):
                s = 0.0
                for k in range(3):
                    u = 0.0
                    s += qd.sin(x[0]) + 1.0
                    for l in range(2):
                        u += qd.sin(x[0])
                    loss[0] += u
                loss[0] += s

    x[0] = 0.0
    with qd.ad.Tape(loss=loss):
        test_more_inner_loops_local_variable(x, arr, loss)

    assert loss[0] == 12.0
    assert x.grad[0] == 36.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=32)
def test_stacked_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = qd.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(), needs_grad=True)

    @qd.kernel
    def test_stacked_inner_loops_local_variable(
        x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()
    ):
        for i in arr:
            loss[None] += qd.sin(x[None])
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += qd.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += qd.sin(x[None]) + 1.0
                loss[None] += s

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_inner_loops_local_variable(x, arr, loss)
    test_stacked_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 36.0
    assert x.grad[None] == 38.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=32)
def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable_fixed_stack_size_kernel_grad():
    x = qd.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(), needs_grad=True)

    @qd.kernel
    def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(
        x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()
    ):
        for i in arr:
            loss[None] += qd.sin(x[None])
            for j in range(3):
                for k in range(3):
                    loss[None] += qd.sin(x[None]) + 1.0
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += qd.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                for k in range(3):
                    loss[None] += qd.sin(x[None]) + 1.0

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(x, arr, loss)
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 54.0
    assert x.grad[None] == 56.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=0)
def test_stacked_inner_loops_local_variable_adaptive_stack_size_kernel_grad():
    x = qd.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(), needs_grad=True)

    @qd.kernel
    def test_stacked_inner_loops_local_variable(
        x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()
    ):
        for i in arr:
            loss[None] += qd.sin(x[None])
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += qd.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += qd.sin(x[None]) + 1.0
                loss[None] += s

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_inner_loops_local_variable(x, arr, loss)
    test_stacked_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 36.0
    assert x.grad[None] == 38.0


@pytest.mark.flaky(reruns=5)
@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=0)
def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable_adaptive_stack_size_kernel_grad():
    x = qd.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(), needs_grad=True)

    @qd.kernel
    def test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(
        x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()
    ):
        for i in arr:
            loss[None] += qd.sin(x[None])
            for j in range(3):
                for k in range(3):
                    loss[None] += qd.sin(x[None]) + 1.0
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += qd.sin(x[None]) + 1.0
                loss[None] += s
            for j in range(3):
                for k in range(3):
                    loss[None] += qd.sin(x[None]) + 1.0

    loss.grad[None] = 1.0
    x[None] = 0.0
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable(x, arr, loss)
    test_stacked_mixed_ib_and_non_ib_inner_loops_local_variable.grad(x, arr, loss)

    assert loss[None] == 54.0
    assert x.grad[None] == 56.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=0)
def test_large_for_loops_adaptive_stack_size():
    x = qd.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(), needs_grad=True)

    @qd.kernel
    def test_large_loop(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in range(5):
            for j in range(2000):
                for k in range(1000):
                    loss[None] += qd.sin(x[None]) + 1.0

    with qd.ad.Tape(loss=loss):
        test_large_loop(x, arr, loss)

    assert loss[None] == 1e7
    assert x.grad[None] == 1e7


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack, ad_stack_size=1)
def test_large_for_loops_fixed_stack_size():
    x = qd.ndarray(dtype=float, shape=(), needs_grad=True)
    arr = qd.ndarray(dtype=float, shape=(2), needs_grad=True)
    loss = qd.ndarray(dtype=float, shape=(), needs_grad=True)

    @qd.kernel
    def test_large_loop(x: qd.types.ndarray(), arr: qd.types.ndarray(), loss: qd.types.ndarray()):
        for i in range(5):
            for j in range(2000):
                for k in range(1000):
                    loss[None] += qd.sin(x[None]) + 1.0

    with qd.ad.Tape(loss=loss):
        test_large_loop(x, arr, loss)

    assert loss[None] == 1e7
    assert x.grad[None] == 1e7


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_multiple_ib():
    x = qd.ndarray(float, (), needs_grad=True)
    y = qd.ndarray(float, (), needs_grad=True)

    @qd.kernel
    def compute_y(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with qd.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 12.0
    assert x.grad[None] == 12.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_multiple_ib_multiple_outermost():
    x = qd.ndarray(float, (), needs_grad=True)
    y = qd.ndarray(float, (), needs_grad=True)

    @qd.kernel
    def compute_y(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with qd.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 24.0
    assert x.grad[None] == 24.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_multiple_ib_multiple_outermost_mixed():
    x = qd.ndarray(float, (), needs_grad=True)
    y = qd.ndarray(float, (), needs_grad=True)

    @qd.kernel
    def compute_y(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
                for ii in range(3):
                    y[None] += x[None]

    x[None] = 1.0
    with qd.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 42.0
    assert x.grad[None] == 42.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_multiple_ib_mixed():
    x = qd.ndarray(float, (), needs_grad=True)
    y = qd.ndarray(float, (), needs_grad=True)

    @qd.kernel
    def compute_y(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                y[None] += x[None]
                for k in range(2):
                    y[None] += x[None]
            for i in range(3):
                y[None] += x[None]

    x[None] = 1.0
    with qd.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 30.0
    assert x.grad[None] == 30.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_multiple_ib_deeper():
    x = qd.ndarray(float, (), needs_grad=True)
    y = qd.ndarray(float, (), needs_grad=True)

    @qd.kernel
    def compute_y(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    for iii in range(2):
                        y[None] += x[None]

    x[None] = 1.0
    with qd.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 42.0
    assert x.grad[None] == 42.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_multiple_ib_deeper_non_scalar():
    N = 10
    x = qd.ndarray(float, shape=N, needs_grad=True)
    y = qd.ndarray(float, shape=N, needs_grad=True)

    @qd.kernel
    def compute_y(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for j in range(N):
            for i in range(j):
                y[j] += x[j]
            for i in range(3):
                for ii in range(j):
                    y[j] += x[j]
            for i in range(3):
                for ii in range(2):
                    for iii in range(j):
                        y[j] += x[j]

    x.fill(1.0)
    for i in range(N):
        y.grad[i] = 1.0
    compute_y(x, y)
    compute_y.grad(x, y)
    for i in range(N):
        assert y[i] == i * 10.0
        assert x.grad[i] == i * 10.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_multiple_ib_inner_mixed():
    x = qd.ndarray(float, (), needs_grad=True)
    y = qd.ndarray(float, (), needs_grad=True)

    @qd.kernel
    def compute_y(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for j in range(2):
            for i in range(3):
                y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    y[None] += x[None]
                for iii in range(2):
                    y[None] += x[None]
                    for iiii in range(2):
                        y[None] += x[None]
            for i in range(3):
                for ii in range(2):
                    for iii in range(2):
                        y[None] += x[None]

    x[None] = 1.0
    with qd.ad.Tape(y):
        compute_y(x, y)

    assert y[None] == 78.0
    assert x.grad[None] == 78.0


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ib_global_load():
    N = 10
    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.i32, shape=N)
    p = qd.ndarray(qd.f32, shape=N, needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(), b: qd.types.ndarray(), p: qd.types.ndarray()):
        for i in range(N):
            val = a[i]
            for j in range(b[i]):
                p[i] += i
            p[i] = val * i

    for i in range(N):
        a[i] = i
        b[i] = 2

    compute(a, b, p)

    for i in range(N):
        assert p[i] == i * i
        p.grad[i] = 1

    compute.grad(a, b, p)
    for i in range(N):
        assert a.grad[i] == i


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ad_if_simple():
    x = qd.ndarray(qd.f32, shape=(), needs_grad=True)
    y = qd.ndarray(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def func(x: qd.types.ndarray(), y: qd.types.ndarray()):
        if x[None] > 0.0:
            y[None] = x[None]

    x[None] = 1
    y.grad[None] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[None] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ad_if():
    x = qd.ndarray(qd.f32, shape=2, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=2, needs_grad=True)

    @qd.kernel
    def func(i: qd.i32, x: qd.types.ndarray(), y: qd.types.ndarray()):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 2 * x[i]

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0, x, y)
    func.grad(0, x, y)
    func(1, x, y)
    func.grad(1, x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ad_if_nested():
    n = 20
    x = qd.ndarray(qd.f32, shape=n, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=n, needs_grad=True)
    z = qd.ndarray(qd.f32, shape=n, needs_grad=True)

    @qd.kernel
    def func(x: qd.types.ndarray(), y: qd.types.ndarray(), z: qd.types.ndarray()):
        for i in x:
            if x[i] < 2:
                if x[i] == 0:
                    y[i] = 0
                else:
                    y[i] = z[i] * 1
            else:
                if x[i] == 2:
                    y[i] = z[i] * 2
                else:
                    y[i] = z[i] * 3

    z.fill(1)

    for i in range(n):
        x[i] = i % 4

    func(x, y, z)
    for i in range(n):
        assert y[i] == i % 4
        y.grad[i] = 1
    func.grad(x, y, z)

    for i in range(n):
        assert z.grad[i] == i % 4


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ad_if_mutable():
    x = qd.ndarray(qd.f32, shape=2, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=2, needs_grad=True)

    @qd.kernel
    def func(i: qd.i32, x: qd.types.ndarray(), y: qd.types.ndarray()):
        t = x[i]
        if t > 0:
            y[i] = t
        else:
            y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0, x, y)
    func.grad(0, x, y)
    func(1, x, y)
    func.grad(1, x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ad_if_parallel():
    x = qd.ndarray(qd.f32, shape=2, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=2, needs_grad=True)

    @qd.kernel
    def func(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ad_if_parallel_f64():
    x = qd.ndarray(qd.f64, shape=2, needs_grad=True)
    y = qd.ndarray(qd.f64, shape=2, needs_grad=True)

    @qd.kernel
    def func(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ad_if_parallel_complex():
    x = qd.ndarray(qd.f32, shape=2, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=2, needs_grad=True)

    @qd.kernel
    def func(x: qd.types.ndarray(), y: qd.types.ndarray()):
        qd.loop_config(parallelize=1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2
    y.grad[0] = 1
    y.grad[1] = 1

    func(x, y)
    func.grad(x, y)

    assert x.grad[0] == 0
    assert x.grad[1] == -0.25


@pytest.mark.flaky(retries=5)
@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_ndarray_i32():
    with pytest.raises(QuadrantsRuntimeError, match=r"i32 is not supported for ndarray"):
        qd.ndarray(qd.i32, shape=3, needs_grad=True)


@test_utils.test(arch=archs_support_ndarray_ad)
@pytest.mark.flaky(retries=5)
def test_ad_sum_vector():
    N = 10

    @qd.kernel
    def compute_sum(a: qd.types.ndarray(), p: qd.types.ndarray()):
        for i in p:
            p[i] = a[i] * 2

    a = qd.ndarray(qd.math.vec2, shape=N, needs_grad=True)
    p = qd.ndarray(qd.math.vec2, shape=N, needs_grad=True)
    for i in range(N):
        a[i] = [3, 3]

    compute_sum(a, p)

    for i in range(N):
        assert p[i] == [a[i] * 2, a[i] * 3]
        p.grad[i] = [1, 1]

    compute_sum.grad(a, p)

    for i in range(N):
        for j in range(2):
            assert a.grad[i][j] == 2


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_multiple_tapes():
    N = 10

    @qd.kernel
    def compute_sum(a: qd.types.ndarray(), p: qd.types.ndarray()):
        for i in a:
            p[None] += a[i][0] * 2 + a[i][1] * 3

    a = qd.ndarray(qd.math.vec2, shape=N, needs_grad=True)
    p = qd.ndarray(qd.f32, shape=(), needs_grad=True)

    init_val = 3
    for i in range(N):
        a[i] = [init_val, init_val]

    with qd.ad.Tape(loss=p):
        compute_sum(a, p)

    assert p[None] == N * (2 + 3) * init_val

    for i in range(N):
        assert a.grad[i][0] == 2
        assert a.grad[i][1] == 3

    # second run
    a.grad.fill(0)
    with qd.ad.Tape(loss=p):
        compute_sum(a, p)

    assert p[None] == N * (2 + 3) * init_val

    for i in range(N):
        assert a.grad[i][0] == 2
        assert a.grad[i][1] == 3


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_set_loss_grad():
    x = qd.ndarray(dtype=qd.f32, shape=(), needs_grad=True)
    loss = qd.ndarray(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def eval_x(x: qd.types.ndarray()):
        x[None] = 1.0

    @qd.kernel
    def compute_1(x: qd.types.ndarray(), loss: qd.types.ndarray()):
        loss[None] = x[None]

    @qd.kernel
    def compute_2(x: qd.types.ndarray(), loss: qd.types.ndarray()):
        loss[None] = 2 * x[None]

    @qd.kernel
    def compute_3(x: qd.types.ndarray(), loss: qd.types.ndarray()):
        loss[None] = 4 * x[None]

    eval_x(x)
    with qd.ad.Tape(loss=loss):
        compute_1(x, loss)
        compute_2(x, loss)
        compute_3(x, loss)

    assert loss[None] == 4
    assert x.grad[None] == 4


@test_utils.test(arch=archs_support_ndarray_ad)
def test_grad_tensor_in_kernel():
    N = 10

    a = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    b = qd.ndarray(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def test(x: qd.types.ndarray(), b: qd.types.ndarray()):
        for i in x:
            b[None] += x.grad[i]

    a.grad.fill(2.0)
    test(a, b)
    assert b[None] == N * 2.0

    with pytest.raises(RuntimeError, match=r"Cannot automatically differentiate through a grad tensor"):
        test.grad(a, b)


@test_utils.test(arch=archs_support_ndarray_ad, require=qd.extension.adstack)
def test_ndarray_needs_grad_false():
    N = 3

    @qd.kernel
    def test(x: qd.types.ndarray(needs_grad=False), y: qd.types.ndarray()):
        for i in range(N):
            a = 2.0
            for j in range(N):
                a += x[i] / x.shape[0]
            y[0] += a

    x = qd.ndarray(qd.f32, shape=N, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=1, needs_grad=True)

    test(x, y)

    y.grad.fill(1.0)
    test.grad(x, y)
    for i in range(N):
        assert x.grad[i] == 0.0


@test_utils.test(arch=archs_support_ndarray_ad)
def test_ad_vector_arg():
    N = 10

    @qd.kernel
    def compute_sum(a: qd.types.ndarray(), p: qd.types.ndarray(), z: qd.math.vec2):
        for i in p:
            p[i] = a[i] * z[0]

    a = qd.ndarray(qd.math.vec2, shape=N, needs_grad=True)
    p = qd.ndarray(qd.math.vec2, shape=N, needs_grad=True)
    z = qd.math.vec2([2.0, 3.0])
    for i in range(N):
        a[i] = [3, 3]

    compute_sum(a, p, z)

    for i in range(N):
        assert p[i] == [a[i] * 2, a[i] * 2]
        p.grad[i] = [1, 1]

    compute_sum.grad(a, p, z)

    for i in range(N):
        for j in range(2):
            assert a.grad[i][j] == 2


@test_utils.test(arch=archs_support_ndarray_ad)
def test_hash_encoder_simple():
    @qd.kernel
    def hash_encoder_kernel(
        table: qd.types.ndarray(),
        output_embedding: qd.types.ndarray(),
    ):
        qd.loop_config(block_dim=256)
        for level in range(1):
            local_features = qd.Vector([0.0])

            tmp0 = local_features[0] + table[0]
            local_features[0] = tmp0

            if level < 0:
                # To keep this IfStmt
                print(1111)

            tmp1 = local_features[0] + table[0]
            local_features[0] = tmp1

            output_embedding[0, 0] = local_features[0]

    table = qd.ndarray(shape=(1), dtype=qd.f32, needs_grad=True)
    output_embedding = qd.ndarray(shape=(1, 1), dtype=qd.f32, needs_grad=True)

    table[0] = 0.2924
    table.grad[0] = 0.0
    output_embedding[0, 0] = 0.7515
    output_embedding.grad[0, 0] = 2.8942e-06

    hash_encoder_kernel.grad(table, output_embedding)

    assert table.grad[0] > 5.788399e-06 and table.grad[0] < 5.7884e-06
