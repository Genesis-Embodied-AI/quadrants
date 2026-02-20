import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_customized_kernels_tape():
    x = qd.field(qd.f32)
    total = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(total)
    qd.root.lazy_grad()

    @qd.kernel
    def func(mul: qd.f32):
        for i in range(n):
            qd.atomic_add(total[None], x[i] * mul)

    @qd.ad.grad_replaced
    def forward(mul):
        func(mul)
        func(mul)

    @qd.ad.grad_for(forward)
    def backward(mul):
        func.grad(mul)

    with qd.ad.Tape(loss=total):
        forward(4)
    assert x.grad[0] == 4


@test_utils.test()
def test_customized_kernels_grad():
    x = qd.field(qd.f32)
    total = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(total)
    qd.root.lazy_grad()

    @qd.kernel
    def func(mul: qd.f32):
        for i in range(n):
            qd.atomic_add(total[None], x[i] * mul)

    @qd.ad.grad_replaced
    def forward(mul):
        func(mul)
        func(mul)

    @qd.ad.grad_for(forward)
    def backward(mul):
        func.grad(mul)

    total.grad[None] = 1
    forward(4)
    forward.grad(4)
    assert x.grad[0] == 4


@test_utils.test()
def test_customized_kernels_indirect():
    x = qd.field(qd.f32)
    total = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(total)
    qd.root.lazy_grad()

    @qd.kernel
    def func(mul: qd.f32):
        for i in range(n):
            qd.atomic_add(total[None], x[i] * mul)

    def func_proxy(mul):
        func(mul)

    @qd.ad.grad_replaced
    def forward(mul):
        func_proxy(mul)
        func_proxy(mul)

    @qd.ad.grad_for(forward)
    def backward(mul):
        func.grad(mul)

    with qd.ad.Tape(loss=total):
        forward(4)
    assert x.grad[0] == 4


@test_utils.test()
def test_customized_kernels_oop():
    @qd.data_oriented
    class A:
        def __init__(self):
            self.x = qd.field(qd.f32)
            self.total = qd.field(qd.f32)
            self.n = 128

            qd.root.dense(qd.i, self.n).place(self.x)
            qd.root.place(self.total)

        @qd.kernel
        def func(self, mul: qd.f32):
            for i in range(self.n):
                qd.atomic_add(self.total[None], self.x[i] * mul)

        @qd.ad.grad_replaced
        def forward(self, mul):
            self.func(mul)
            self.func(mul)

        @qd.ad.grad_for(forward)
        def backward(self, mul):
            self.func.grad(mul)

    a = A()

    qd.root.lazy_grad()

    with qd.ad.Tape(loss=a.total):
        a.forward(4)
    assert a.x.grad[0] == 4


@test_utils.test()
def test_customized_kernels_oop2():
    @qd.data_oriented
    class A:
        def __init__(self):
            self.x = qd.field(qd.f32)
            self.total = qd.field(qd.f32)
            self.n = 128

            qd.root.dense(qd.i, self.n).place(self.x)
            qd.root.place(self.total)

        @qd.kernel
        def func(self, mul: qd.f32):
            for i in range(self.n):
                qd.atomic_add(self.total[None], self.x[i] * mul)

        def func_proxy(self, mul):
            self.func(mul)

        @qd.ad.grad_replaced
        def forward(self, mul):
            self.func_proxy(mul)
            self.func_proxy(mul)

        @qd.ad.grad_for(forward)
        def backward(self, mul):
            self.func.grad(mul)

    a = A()

    qd.root.lazy_grad()

    with qd.ad.Tape(loss=a.total):
        a.forward(4)
    assert a.x.grad[0] == 4


@test_utils.test()
def test_decorated_primal_is_quadrants_kernel():
    x = qd.field(qd.f32)
    total = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(total)
    qd.root.lazy_grad()

    @qd.kernel
    def func(mul: qd.f32):
        for i in range(n):
            qd.atomic_add(total[None], x[i] * mul)

    with pytest.raises(RuntimeError):

        @qd.ad.grad_for(func)
        def backward(mul):
            func.grad(mul)

    with qd.ad.Tape(loss=total):
        func(4)


@test_utils.test()
def test_decorated_primal_missing_decorator():
    x = qd.field(qd.f32)
    total = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(total)
    qd.root.lazy_grad()

    @qd.kernel
    def func(mul: qd.f32):
        for i in range(n):
            qd.atomic_add(total[None], x[i] * mul)

    def forward(mul):
        func(mul)
        func(mul)

    with pytest.raises(RuntimeError):

        @qd.ad.grad_for(func)
        def backward(mul):
            func.grad(mul)

    with qd.ad.Tape(loss=total):
        func(4)


@test_utils.test()
def test_customized_kernels_tape_no_grad():
    x = qd.field(qd.f32)
    total = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(total)
    qd.root.lazy_grad()

    @qd.kernel
    def func(mul: qd.f32):
        for i in range(n):
            qd.atomic_add(total[None], x[i] * mul)

    @qd.ad.no_grad
    def forward(mul):
        func(mul)
        func(mul)

    with qd.ad.Tape(loss=total):
        forward(4)
        func(5)
    assert x.grad[0] == 5


@test_utils.test()
def test_customized_kernels_grad_no_grad():
    x = qd.field(qd.f32)
    total = qd.field(qd.f32)

    n = 128

    qd.root.dense(qd.i, n).place(x)
    qd.root.place(total)
    qd.root.lazy_grad()

    @qd.kernel
    def func(mul: qd.f32):
        for i in range(n):
            qd.atomic_add(total[None], x[i] * mul)

    @qd.ad.no_grad
    def forward(mul):
        func(mul)
        func(mul)

    total.grad[None] = 1
    forward(4)
    forward.grad(4)
    assert x.grad[0] == 0
