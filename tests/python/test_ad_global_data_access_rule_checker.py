import pytest

import quadrants as qd
from quadrants.types.enums import AutodiffMode

from tests import test_utils


@test_utils.test(debug=True)
def test_adjoint_checkbit_needs_grad():
    x = qd.field(float, shape=(), needs_grad=True)

    @qd.kernel
    def test():
        x[None] = 1

    with qd.ad.Tape(loss=x, validation=True):
        test()

    assert x.snode.ptr.has_adjoint_checkbit()


@test_utils.test(debug=True)
def test_adjoint_checkbit_lazy_grad():
    x = qd.field(float, shape=())
    qd.root.lazy_grad()

    @qd.kernel
    def test():
        x[None] = 1

    with qd.ad.Tape(loss=x, validation=True):
        test()

    assert x.snode.ptr.has_adjoint_checkbit()


@test_utils.test(debug=True)
def test_adjoint_checkbit_place_grad():
    x = qd.field(float)
    y = qd.field(float)
    qd.root.place(x, x.grad, y)

    @qd.kernel
    def test():
        x[None] = 1

    with qd.ad.Tape(loss=x, validation=True):
        test()

    assert x.snode.ptr.has_adjoint_checkbit()
    assert not y.snode.ptr.has_adjoint_checkbit()


@test_utils.test(debug=False)
def test_adjoint_checkbit_needs_grad():
    x = qd.field(float, shape=(), needs_grad=True)

    @qd.kernel
    def test():
        x[None] = 1

    with pytest.warns(Warning) as record:
        with qd.ad.Tape(loss=x, validation=True):
            test()

    warn_raised = False
    for warn in record:
        print("warn.message.args[0]", warn.message.args[0])
        if (
            "Debug mode is disabled, autodiff valid check will not work. Please specify `qd.init(debug=True)` to enable the check."
            in warn.message.args[0]
        ):
            warn_raised = True
    assert warn_raised


@test_utils.test(require=qd.extension.assertion, debug=True)
def test_break_gdar_rule_1():
    N = 16
    x = qd.field(dtype=qd.f32, shape=N, needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    b = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def func_broke_rule_1():
        loss[None] = x[1] * b[None]
        b[None] += 100

    for i in range(N):
        x[i] = i

    b[None] = 10
    loss.grad[None] = 1

    with pytest.raises(qd.QuadrantsAssertionError):
        with qd.ad.Tape(loss=loss, validation=True):
            func_broke_rule_1()


@test_utils.test(require=qd.extension.assertion, debug=True)
def test_skip_grad_replaced():
    N = 16
    x = qd.field(dtype=qd.f32, shape=N, needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    b = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    # This kernel breaks the global data access rule
    @qd.kernel
    def kernel_1():
        loss[None] = x[1] * b[None]
        b[None] += 100

    @qd.ad.grad_replaced
    def kernel_2():
        loss[None] = x[1] * b[None]
        b[None] += 100

    # The user defined grad kernel is not restricted by the global data access rule, thus should be skipped when checking
    @qd.ad.grad_for(kernel_2)
    def kernel_2_grad():
        pass

    for i in range(N):
        x[i] = i

    b[None] = 10
    loss.grad[None] = 1

    with pytest.raises(qd.QuadrantsAssertionError):
        with qd.ad.Tape(loss=loss, validation=True):
            kernel_1()

    with qd.ad.Tape(loss=loss, validation=True):
        kernel_2()


@test_utils.test(require=qd.extension.assertion, debug=True)
def test_autodiff_mode_recovered():
    N = 16
    x = qd.field(dtype=qd.f32, shape=N, needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    b = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def kernel_1():
        loss[None] = x[1] * b[None]

    @qd.kernel
    def kernel_2():
        loss[None] = x[1] * b[None]

    for i in range(N):
        x[i] = i

    b[None] = 10
    loss.grad[None] = 1

    func_calls = []
    with qd.ad.Tape(loss=loss, validation=True) as t:
        kernel_1()
        kernel_2()
        for f, _ in t.calls:
            assert f.autodiff_mode == AutodiffMode.VALIDATION
        func_calls = t.calls
    for f, _ in func_calls:
        assert f.autodiff_mode == AutodiffMode.NONE


@test_utils.test(require=qd.extension.assertion, debug=True)
def test_validation_kernel_capture():
    N = 16
    T = 8
    x = qd.field(dtype=qd.f32, shape=N, needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    b = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def kernel_1():
        loss[None] = x[1] * b[None]

    @qd.kernel
    def kernel_2():
        loss[None] = x[1] * b[None]

    def forward(T):
        for t in range(T):
            kernel_1()
            kernel_2()

    for i in range(N):
        x[i] = i

    b[None] = 10
    loss.grad[None] = 1

    with qd.ad.Tape(loss=loss, validation=True) as t:
        forward(T)
        assert len(t.calls) == 2 * T and len(t.modes) == 2 * T
