import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test()
def test_nested_subscript():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    qd.root.dense(qd.i, 1).place(x)
    qd.root.dense(qd.i, 1).place(y)

    x[0] = 0

    @qd.kernel
    def inc():
        for i in range(1):
            x[x[i]] += 1

    inc()

    assert x[0] == 1


@test_utils.test()
def test_norm():
    val = qd.field(qd.i32)
    f = qd.field(qd.f32)

    n = 1024

    qd.root.dense(qd.i, n).dense(qd.i, 2).place(val, f)

    @qd.kernel
    def test():
        for i in range(n):
            s = 0
            for j in range(10):
                s += j
            a = qd.Vector([0.4, 0.3])
            val[i] = s + qd.cast(a.norm() * 100, qd.i32) + i

    test()

    @qd.kernel
    def test2():
        for i in range(n):
            val[i] += 1

    test2()

    for i in range(n):
        assert val[i] == 96 + i


@test_utils.test()
def test_simple2():
    val = qd.field(qd.i32)
    f = qd.field(qd.f32)

    n = 16

    qd.root.dense(qd.i, n).place(val, f)

    @qd.kernel
    def test():
        for i in range(n):
            val[i] = i * 2

    test()

    @qd.kernel
    def test2():
        for i in range(n):
            val[i] += 1

    test2()

    for i in range(n):
        assert val[i] == 1 + i * 2


@test_utils.test()
def test_recreate():
    @qd.kernel
    def test():
        a = 0
        a, b = 1, 2

    test()


@test_utils.test(exclude=[qd.amdgpu])
def test_local_atomics():
    n = 32
    val = qd.field(qd.i32, shape=n)

    @qd.kernel
    def test():
        for i in range(n):
            s = 0
            s += 45
            print(s)
            val[i] = s + i
            print(val[i])

    test()

    for i in range(n):
        assert val[i] == i + 45


@test_utils.test(arch=get_host_arch_list())
def test_loop_var_life():
    @qd.kernel
    def test():
        for i in qd.static(range(8)):
            pass
        print(i)

    with pytest.raises(Exception):
        test()


@test_utils.test(arch=get_host_arch_list())
def test_loop_var_life_double_iters():
    @qd.kernel
    def test():
        for i, v in qd.static(enumerate(range(8))):
            pass
        print(i)

    with pytest.raises(Exception):
        test()


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.i64, qd.f64])
@pytest.mark.parametrize("ti_zero,zero", [(qd.zero, 0), (qd.one, 1)])
@pytest.mark.parametrize("is_mat", [False, True])
@test_utils.test(arch=qd.cpu)
def test_meta_zero_one(dtype, ti_zero, zero, is_mat):
    if is_mat:
        x = qd.Matrix.field(2, 3, dtype, ())
        y = qd.Matrix.field(2, 3, dtype, ())
    else:
        x = qd.field(dtype, ())
        y = qd.field(dtype, ())

    @qd.kernel
    def func():
        y[None] = ti_zero(x[None])

    for a in [-1, -2.3, -1, -0.3, 0, 1, 1.9, 2, 3]:
        if qd.types.is_integral(dtype):
            a = int(a)
        x.fill(a)
        func()
        assert np.all(y.to_numpy() == zero)
