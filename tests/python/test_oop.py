import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_classfunc():
    @qd.data_oriented
    class Array2D:
        def __init__(self, n, m):
            self.n = n
            self.m = m
            self.val = qd.field(qd.f32, shape=(n, m))

        @qd.func
        def inc(self, i, j):
            self.val[i, j] += i * j

        @qd.func
        def mul(self, i, j):
            return i * j

        @qd.kernel
        def fill(self):
            for i, j in self.val:
                self.inc(i, j)
                self.val[i, j] += self.mul(i, j)

    arr = Array2D(128, 128)

    arr.fill()

    for i in range(arr.n):
        for j in range(arr.m):
            assert arr.val[i, j] == i * j * 2


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_class_real_func():
    @qd.data_oriented
    class Array2D:
        def __init__(self, n, m):
            self.n = n
            self.m = m
            self.val = qd.field(qd.f32, shape=(n, m))

        @qd.real_func
        def inc(self, i: qd.i32, j: qd.i32):
            self.val[i, j] += i * j

        @qd.real_func
        def mul(self, i: qd.i32, j: qd.i32) -> qd.i32:
            return i * j

        @qd.kernel
        def fill(self):
            for i, j in self.val:
                self.inc(i, j)
                self.val[i, j] += self.mul(i, j)

    arr = Array2D(128, 128)

    arr.fill()

    for i in range(arr.n):
        for j in range(arr.m):
            assert arr.val[i, j] == i * j * 2


@test_utils.test(arch=get_host_arch_list())
def test_oop():
    @qd.data_oriented
    class Array2D:
        def __init__(self, n, m, increment):
            self.n = n
            self.m = m
            self.val = qd.field(qd.f32)
            self.total = qd.field(qd.f32)
            self.increment = increment

            qd.root.dense(qd.ij, (self.n, self.m)).place(self.val)
            qd.root.place(self.total)

        @qd.kernel
        def inc(self):
            for i, j in self.val:
                self.val[i, j] += self.increment

        @qd.kernel
        def inc2(self, increment: qd.i32):
            for i, j in self.val:
                self.val[i, j] += increment

        @qd.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * 4

    arr = Array2D(128, 128, 3)

    double_total = qd.field(qd.f32)

    qd.root.place(double_total)
    qd.root.lazy_grad()

    arr.inc()
    arr.inc.grad()
    assert arr.val[3, 4] == 3
    arr.inc2(4)
    assert arr.val[3, 4] == 7

    with qd.ad.Tape(loss=arr.total):
        arr.reduce()

    for i in range(arr.n):
        for j in range(arr.m):
            assert arr.val.grad[i, j] == 4

    @qd.kernel
    def double():
        double_total[None] = 2 * arr.total[None]

    with qd.ad.Tape(loss=double_total):
        arr.reduce()
        double()

    for i in range(arr.n):
        for j in range(arr.m):
            assert arr.val.grad[i, j] == 8


@test_utils.test(arch=get_host_arch_list())
def test_oop_two_items():
    @qd.data_oriented
    class Array2D:
        def __init__(self, n, m, increment, multiplier):
            self.n = n
            self.m = m
            self.val = qd.field(qd.f32)
            self.total = qd.field(qd.f32)
            self.increment = increment
            self.multiplier = multiplier
            qd.root.dense(qd.ij, (self.n, self.m)).place(self.val)
            qd.root.place(self.total)

        @qd.kernel
        def inc(self):
            for i, j in self.val:
                self.val[i, j] += self.increment

        @qd.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * self.multiplier

    arr1_inc, arr1_mult = 3, 4
    arr2_inc, arr2_mult = 6, 8
    arr1 = Array2D(128, 128, arr1_inc, arr1_mult)
    arr2 = Array2D(16, 32, arr2_inc, arr2_mult)

    qd.root.lazy_grad()

    arr1.inc()
    arr1.inc.grad()
    arr2.inc()
    arr2.inc.grad()
    assert arr1.val[3, 4] == arr1_inc
    assert arr2.val[8, 6] == arr2_inc

    with qd.ad.Tape(loss=arr1.total):
        arr1.reduce()
    with qd.ad.Tape(loss=arr2.total, clear_gradients=False):
        arr2.reduce()
    for i in range(arr1.n):
        for j in range(arr1.m):
            assert arr1.val.grad[i, j] == arr1_mult
    for i in range(arr2.n):
        for j in range(arr2.m):
            assert arr2.val.grad[i, j] == arr2_mult


@test_utils.test(arch=get_host_arch_list())
def test_oop_inherit_ok():
    # Array1D inherits from object, which makes the callstack being 'class Array2D(object)'
    # instead of '@qd.data_oriented'. Make sure this also works.
    @qd.data_oriented
    class Array1D(object):
        def __init__(self, n, mul):
            self.n = n
            self.val = qd.field(qd.f32)
            self.total = qd.field(qd.f32)
            self.mul = mul
            qd.root.dense(qd.ij, (self.n,)).place(self.val)
            qd.root.place(self.total)

        @qd.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * self.mul

    arr = Array1D(128, 42)

    qd.root.lazy_grad()

    with qd.ad.Tape(loss=arr.total):
        arr.reduce()
    for i in range(arr.n):
        for j in range(arr.n):
            assert arr.val.grad[i, j] == 42


@test_utils.test(arch=get_host_arch_list())
def test_oop_class_must_be_data_oriented():
    class Array1D(object):
        def __init__(self, n, mul):
            self.n = n
            self.val = qd.field(qd.f32)
            self.total = qd.field(qd.f32)
            self.mul = mul
            qd.root.dense(qd.ij, (self.n,)).place(self.val)
            qd.root.place(self.total)

        @qd.kernel
        def reduce(self):
            for i, j in self.val:
                self.total[None] += self.val[i, j] * self.mul

    arr = Array1D(128, 42)

    qd.root.lazy_grad()

    # Array1D is not properly decorated, this will raise an Exception
    with pytest.raises(qd.QuadrantsSyntaxError):
        arr.reduce()


@test_utils.test(arch=get_host_arch_list())
def test_hook():
    @qd.data_oriented
    class Solver:
        def __init__(self, n, m, hook):
            self.val = qd.field(qd.f32, shape=(n, m))
            self.hook = hook

        def run_hook(self):
            self.hook(self.val)

    @qd.kernel
    def hook(x: qd.template()):
        for i, j in x:
            x[i, j] = 1.0

    solver = Solver(32, 32, hook)
    solver.run_hook()

    for i in range(32):
        for j in range(32):
            assert solver.val[i, j] == 1.0


@test_utils.test()
def test_oop_with_property_decorator():
    @qd.data_oriented
    class TestProperty:
        @property
        @qd.kernel
        def kernel_property(self) -> qd.i32:
            return 42

        @property
        def raw_property(self):
            return 3

    a = TestProperty()
    assert a.kernel_property == 42

    assert a.raw_property == 3


@test_utils.test()
def test_oop_with_static_decorator():
    @qd.data_oriented
    class TestStatic:
        @staticmethod
        @qd.kernel
        def kernel_static() -> qd.i32:
            return 42

        @staticmethod
        def raw_static():
            return 3

    a = TestStatic()
    assert a.kernel_static() == 42

    assert a.raw_static() == 3
