import functools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

has_autograd = False

try:
    import autograd.numpy as np
    from autograd import grad

    has_autograd = True
except:
    pass


def if_has_autograd(func):
    # functools.wraps is nececssary for pytest parametrization to work
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if has_autograd:
            func(*args, **kwargs)

    return wrapper


# Note: test happens at v = 0.2
def grad_test(tifunc, npfunc=None):
    npfunc = npfunc or tifunc

    print(f"arch={qd.lang.impl.current_cfg().arch} default_fp={qd.lang.impl.current_cfg().default_fp}")
    x = qd.field(qd.lang.impl.current_cfg().default_fp)
    y = qd.field(qd.lang.impl.current_cfg().default_fp)

    qd.root.dense(qd.i, 1).place(x, x.grad, y, y.grad)

    @qd.kernel
    def func():
        for i in x:
            y[i] = tifunc(x[i])

    v = 0.234

    y.grad[0] = 1
    x[0] = v
    func()
    func.grad()

    assert y[0] == test_utils.approx(npfunc(v), rel=1e-4)
    assert x.grad[0] == test_utils.approx(grad(npfunc)(v), rel=1e-4)


def grad_test_fwd(tifunc, npfunc=None):
    npfunc = npfunc or tifunc

    print(f"arch={qd.lang.impl.current_cfg().arch} default_fp={qd.lang.impl.current_cfg().default_fp}")
    x = qd.field(qd.lang.impl.current_cfg().default_fp)
    y = qd.field(qd.lang.impl.current_cfg().default_fp)

    qd.root.dense(qd.i, 1).place(x, x.dual, y, y.dual)

    @qd.kernel
    def func():
        for i in x:
            y[i] = tifunc(x[i])

    v = 0.234

    x[0] = v
    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0]):
        func()

    assert y[0] == test_utils.approx(npfunc(v), rel=1e-4)
    assert y.dual[0] == test_utils.approx(grad(npfunc)(v), rel=1e-4)


@if_has_autograd
@test_utils.test()
def test_size1():
    x = qd.field(qd.i32)

    qd.root.dense(qd.i, 1).place(x)

    x[0] = 1
    assert x[0] == 1


@pytest.mark.parametrize(
    "tifunc",
    [
        lambda x: x,
        lambda x: qd.abs(-x),
        lambda x: -x,
        lambda x: x * x,
        lambda x: x**2,
        lambda x: x * x * x,
        lambda x: x * x * x * x,
        lambda x: 0.4 * x * x - 3,
        lambda x: (x - 3) * (x - 1),
        lambda x: (x - 3) * (x - 1) + x * x,
    ],
)
@if_has_autograd
@test_utils.test()
def test_poly(tifunc):
    grad_test(tifunc)
    grad_test_fwd(tifunc)


@pytest.mark.parametrize(
    "tifunc,npfunc",
    [
        (lambda x: qd.tanh(x), lambda x: np.tanh(x)),
        (lambda x: qd.sin(x), lambda x: np.sin(x)),
        (lambda x: qd.cos(x), lambda x: np.cos(x)),
        (lambda x: qd.acos(x), lambda x: np.arccos(x)),
        (lambda x: qd.asin(x), lambda x: np.arcsin(x)),
    ],
)
@if_has_autograd
@test_utils.test()
def test_trigonometric(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize(
    "tifunc",
    [
        lambda x: 1 / x,
        lambda x: (x + 1) / (x - 1),
        lambda x: (x + 1) * (x + 2) / ((x - 1) * (x + 3)),
    ],
)
@if_has_autograd
@test_utils.test()
def test_frac(tifunc):
    grad_test(tifunc)
    grad_test_fwd(tifunc)


@pytest.mark.parametrize(
    "tifunc,npfunc",
    [
        (lambda x: qd.sqrt(x), lambda x: np.sqrt(x)),
        (lambda x: qd.rsqrt(x), lambda x: 1 / np.sqrt(x)),
        (lambda x: qd.exp(x), lambda x: np.exp(x)),
        (lambda x: qd.log(x), lambda x: np.log(x)),
    ],
)
@if_has_autograd
@test_utils.test()
def test_unary(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize(
    "tifunc,npfunc",
    [
        (lambda x: qd.min(x, 0), lambda x: np.minimum(x, 0)),
        (lambda x: qd.min(x, 1), lambda x: np.minimum(x, 1)),
        (lambda x: qd.min(0, x), lambda x: np.minimum(0, x)),
        (lambda x: qd.min(1, x), lambda x: np.minimum(1, x)),
        (lambda x: qd.max(x, 0), lambda x: np.maximum(x, 0)),
        (lambda x: qd.max(x, 1), lambda x: np.maximum(x, 1)),
        (lambda x: qd.max(0, x), lambda x: np.maximum(0, x)),
        (lambda x: qd.max(1, x), lambda x: np.maximum(1, x)),
    ],
)
@if_has_autograd
@test_utils.test()
def test_minmax(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@if_has_autograd
@test_utils.test()
def test_mod():
    x = qd.field(qd.i32)
    y = qd.field(qd.i32)

    qd.root.dense(qd.i, 1).place(x, y)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        y[0] = x[0] % 3

    @qd.kernel
    def func2():
        qd.atomic_add(y[0], x[0] % 3)

    func()
    func.grad()

    func2()
    func2.grad()


@if_has_autograd
@test_utils.test()
def test_mod_fwd():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x, y)
    qd.root.lazy_dual()

    @qd.kernel
    def func():
        y[0] = x[0] % 3

    @qd.kernel
    def func2():
        qd.atomic_add(y[0], x[0] % 3)

    with qd.ad.FwdMode(loss=y, param=x, seed=[1.0]):
        func()
        func2()


@pytest.mark.parametrize(
    "tifunc,npfunc",
    [
        (lambda x: qd.atan2(0.4, x), lambda x: np.arctan2(0.4, x)),
        (lambda y: qd.atan2(y, 0.4), lambda y: np.arctan2(y, 0.4)),
    ],
)
@if_has_autograd
@test_utils.test()
def test_atan2(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize(
    "tifunc,npfunc",
    [
        (lambda x: qd.atan2(0.4, x), lambda x: np.arctan2(0.4, x)),
        (lambda y: qd.atan2(y, 0.4), lambda y: np.arctan2(y, 0.4)),
    ],
)
@if_has_autograd
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64)
def test_atan2_f64(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize(
    "tifunc,npfunc",
    [
        (lambda x: 0.4**x, lambda x: np.power(0.4, x)),
        (lambda y: y**0.4, lambda y: np.power(y, 0.4)),
    ],
)
@if_has_autograd
@test_utils.test()
def test_pow(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    # grad_test_fwd(tifunc, npfunc)


@pytest.mark.parametrize(
    "tifunc,npfunc",
    [
        (lambda x: 0.4**x, lambda x: np.power(0.4, x)),
        (lambda y: y**0.4, lambda y: np.power(y, 0.4)),
    ],
)
@if_has_autograd
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64)
def test_pow_f64(tifunc, npfunc):
    grad_test(tifunc, npfunc)
    grad_test_fwd(tifunc, npfunc)


@test_utils.test()
def test_select():
    N = 5
    loss = qd.field(qd.f32, shape=N)
    x = qd.field(qd.f32, shape=N)
    y = qd.field(qd.f32, shape=N)
    qd.root.lazy_grad()

    for i in range(N):
        x[i] = i
        y[i] = -i
        loss.grad[i] = 1.0

    @qd.kernel
    def func():
        for i in range(N):
            loss[i] += qd.select(i % 2, x[i], y[i])

    func()
    func.grad()
    for i in range(N):
        if i % 2:
            assert loss[i] == i
        else:
            assert loss[i] == -i
        assert x.grad[i] == i % 2 * 1.0
        assert y.grad[i] == (not i % 2) * 1.0


@test_utils.test()
def test_select_fwd():
    N = 5
    loss = qd.field(qd.f32, shape=N)
    x = qd.field(qd.f32, shape=N)
    y = qd.field(qd.f32, shape=N)
    qd.root.lazy_dual()

    for i in range(N):
        x[i] = i
        y[i] = -i

    @qd.kernel
    def func():
        for i in range(N):
            loss[i] = qd.select(i % 2, x[i], y[i])

    with qd.ad.FwdMode(loss=loss, param=x, seed=[1.0 for _ in range(N)]):
        func()

    for i in range(N):
        if i % 2:
            assert loss[i] == i
        else:
            assert loss[i] == -i
        assert loss.dual[i] == i % 2 * 1.0

    with qd.ad.FwdMode(loss=loss, param=y, seed=[1.0 for _ in range(N)]):
        func()

    for i in range(N):
        if i % 2:
            assert loss[i] == i
        else:
            assert loss[i] == -i
        assert loss.dual[i] == (not i % 2) * 1.0


@test_utils.test()
def test_obey_kernel_simplicity():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x, y)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in x:
            # OK: nested for loop
            for j in qd.static(range(3)):
                # OK: a series of non-for-loop statements
                y[i] += x[i] * 42
                y[i] -= x[i] * 5

    y.grad[0] = 1.0
    x[0] = 0.1

    func()
    func.grad()
    assert x.grad[0] == test_utils.approx((42 - 5) * 3)


@test_utils.test()
def test_violate_kernel_simplicity1():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x, y)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in x:
            y[i] = x[i] * 42
            for j in qd.static(range(3)):
                y[i] += x[i]

    func()
    func.grad()


@test_utils.test()
def test_violate_kernel_simplicity2():
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)

    qd.root.dense(qd.i, 1).place(x, y)
    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in x:
            for j in qd.static(range(3)):
                y[i] += x[i]
            y[i] += x[i] * 42

    func()
    func.grad()


@test_utils.test(require=qd.extension.data64)
def test_cast():
    @qd.kernel
    def func():
        print(qd.cast(qd.cast(qd.cast(1.0, qd.f64), qd.f32), qd.f64))

    func()


@test_utils.test(require=qd.extension.data64)
def test_ad_precision_1():
    loss = qd.field(qd.f32, shape=())
    x = qd.field(qd.f64, shape=())

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        loss[None] = x[None]

    loss.grad[None] = 1
    func.grad()

    assert x.grad[None] == 1


@test_utils.test(require=qd.extension.data64)
def test_ad_precision_2():
    loss = qd.field(qd.f64, shape=())
    x = qd.field(qd.f32, shape=())

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        loss[None] = x[None]

    with qd.ad.Tape(loss):
        func()

    assert x.grad[None] == 1


@test_utils.test()
def test_ad_rand():
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    x = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def work():
        loss[None] = x[None] * qd.random()

    x[None] = 10
    with pytest.raises(RuntimeError) as e:
        with qd.ad.Tape(loss):
            work()
    assert "RandStmt not supported" in e.value.args[0]


@test_utils.test(exclude=[qd.vulkan])
def test_ad_frac():
    @qd.func
    def frac(x):
        fractional = x - qd.floor(x) if x > 0.0 else x - qd.ceil(x)
        return fractional

    @qd.kernel
    def ti_frac(input_field: qd.template(), output_field: qd.template()):
        for i in input_field:
            output_field[i] = frac(input_field[i]) ** 2

    @qd.kernel
    def calc_loss(input_field: qd.template(), loss: qd.template()):
        for i in input_field:
            loss[None] += input_field[i]

    n = 10
    field0 = qd.field(dtype=qd.f32, shape=(n,), needs_grad=True)
    randoms = np.random.randn(10).astype(np.float32)
    field0.from_numpy(randoms)
    field1 = qd.field(dtype=qd.f32, shape=(n,), needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    with qd.ad.Tape(loss):
        ti_frac(field0, field1)
        calc_loss(field1, loss)

    grads = field0.grad.to_numpy()
    expected = np.modf(randoms)[0] * 2
    for i in range(n):
        assert grads[i] == test_utils.approx(expected[i], rel=1e-4)


@test_utils.test()
def test_ad_global_store_forwarding():
    x = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    a = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    b = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    c = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    d = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    e = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def func():
        a[None] = x[None]
        b[None] = a[None] * 2
        c[None] = b[None] * 3
        d[None] = c[None] * 4
        e[None] = d[None] * 5

    x[None] = 1

    with qd.ad.Tape(loss=e):
        func()
    assert x.grad[None] == 120.0
    assert a.grad[None] == 0.0
    assert b.grad[None] == 0.0
    assert c.grad[None] == 0.0
    assert d.grad[None] == 0.0


@test_utils.test()
def test_ad_set_loss_grad():
    x = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def eval_x(x: qd.template()):
        x[None] = 1.0

    @qd.kernel
    def compute_1(x: qd.template(), loss: qd.template()):
        loss[None] = x[None]

    @qd.kernel
    def compute_2(x: qd.template(), loss: qd.template()):
        loss[None] = 2 * x[None]

    @qd.kernel
    def compute_3(x: qd.template(), loss: qd.template()):
        loss[None] = 4 * x[None]

    eval_x(x)
    with qd.ad.Tape(loss=loss):
        compute_1(x, loss)
        compute_2(x, loss)
        compute_3(x, loss)

    assert loss[None] == 4
    assert x.grad[None] == 4
