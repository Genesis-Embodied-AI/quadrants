import math

import pytest

import quadrants as qd

from tests import test_utils


def _arrays():
    x = qd.ndarray(qd.f32, shape=4, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=4, needs_grad=True)
    n = qd.ndarray(qd.i32, shape=4)
    for i in range(4):
        x[i] = 0.2 + i * 0.1
        n[i] = i + 1
    return x, y, n


@pytest.mark.parametrize("debug", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_forward_and_gradient(debug):
    qd.lang.impl.current_cfg().debug = debug
    x, y, n = _arrays()

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n[i]):
                value = qd.sin(value)
            y[i] = value

    for _ in range(2):
        x.grad.fill(0)
        y.grad.fill(1)
        with qd.adstack as scope:
            compute(x, y, n)
        assert scope.capacities[0] > 1
        compute.grad(x, y, n)
        for i in range(4):
            value, derivative = float(x[i]), 1.0
            for _ in range(n[i]):
                derivative *= math.cos(value)
                value = math.sin(value)
            assert y[i] == pytest.approx(value, rel=2e-5)
            assert x.grad[i] == pytest.approx(derivative, rel=2e-5)
        n[3] += 2


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_nested_conditional_loops():
    x, y, n = _arrays()

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n[i]):
                for k in range(j + 1):
                    if k % 2 == 0:
                        value = value * value + 0.1
            y[i] = value

    x.grad.fill(0)
    y.grad.fill(1)
    with qd.adstack:
        compute(x, y, n)
    compute.grad(x, y, n)
    for i in range(4):
        value, derivative = float(x[i]), 1.0
        for j in range(n[i]):
            for k in range(j + 1):
                if k % 2 == 0:
                    derivative *= 2 * value
                    value = value * value + 0.1
        assert y[i] == pytest.approx(value, rel=2e-5)
        assert x.grad[i] == pytest.approx(derivative, rel=2e-5)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_primal_dependent_branches():
    x, y, n = _arrays()
    n.fill(8)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n[i]):
                if value > 0.0 and value < 0.5:
                    value = qd.sin(value) + 0.3
                else:
                    value *= 0.7
            y[i] = value

    x.grad.fill(0)
    y.grad.fill(1)
    with qd.adstack:
        compute(x, y, n)
    compute.grad(x, y, n)
    for i in range(4):
        value, derivative = float(x[i]), 1.0
        for _ in range(n[i]):
            if 0.0 < value < 0.5:
                derivative *= math.cos(value)
                value = math.sin(value) + 0.3
            else:
                derivative *= 0.7
                value *= 0.7
        assert y[i] == pytest.approx(value, rel=2e-5)
        assert x.grad[i] == pytest.approx(derivative, rel=2e-5)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_overflow_is_recoverable_exception():
    x, y, n = _arrays()

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n[i]):
                value = qd.sin(value)
            y[i] = value

    with qd.adstack:
        compute(x, y, n)
    n.fill(100)
    y.grad.fill(1)
    with pytest.raises(RuntimeError, match="Adstack capacity exceeded"):
        compute.grad(x, y, n)
    with pytest.raises(RuntimeError, match="No pending counted forward"):
        compute.grad(x, y, n)
    # The GPU context must remain usable; reset invalid results and start a fresh pair.
    n.fill(2)
    x.grad.fill(0)
    y.grad.fill(1)
    with qd.adstack:
        compute(x, y, n)
    compute.grad(x, y, n)
    assert x.grad[0] == pytest.approx(
        math.cos(x[0]) * math.cos(math.sin(x[0])), rel=2e-5
    )


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_invocation_order():
    x, y, _ = _arrays()

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.i32):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n):
                value = qd.sin(value)
            y[i] = value

    with qd.adstack:
        compute(x, y, 0)
        compute(x, y, 5)
    with pytest.raises(RuntimeError, match="reverse invocation order"):
        compute.grad(x, y, 0)
    y.grad.fill(1)
    compute.grad(x, y, 5)
    x.grad.fill(0)
    y.grad.fill(1)
    compute.grad(x, y, 0)
    assert x.grad[0] == pytest.approx(1)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_tape():
    x = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    loss = qd.ndarray(qd.f32, shape=(), needs_grad=True)
    x[0] = 0.2

    @qd.kernel
    def compute(x: qd.types.ndarray(), loss: qd.types.ndarray(), n: qd.i32):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n):
                value = qd.sin(value)
            loss[None] += value

    with qd.ad.Tape(loss):
        with qd.adstack:
            compute(x, loss, 2)
            compute(x, loss, 3)
    d2 = math.cos(0.2) * math.cos(math.sin(0.2))
    d3 = d2 * math.cos(math.sin(math.sin(0.2)))
    assert x.grad[0] == pytest.approx(d2 + d3, rel=2e-5)


@pytest.mark.parametrize("debug", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_overflow_with_reduction(debug):
    qd.lang.impl.current_cfg().debug = debug
    x = qd.ndarray(qd.f32, shape=(), needs_grad=True)
    y = qd.ndarray(qd.f32, shape=4096, needs_grad=True)
    n = qd.ndarray(qd.i32, shape=4096)
    x[None] = 0.2
    n.fill(2)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(y.shape[0]):
            value = x[None]
            for j in range(n[i]):
                value = qd.sin(value)
            y[i] = value

    # All iterations accumulate into the same input gradient with compiler optimizations enabled.
    with qd.adstack:
        compute(x, y, n)
    # Only one lane overflows; the remaining iterations must still complete their accumulation.
    n[0] = 100
    y.grad.fill(1)
    with pytest.raises(RuntimeError, match="Adstack capacity exceeded"):
        compute.grad(x, y, n)
    qd.sync()

    # Discard the invalid gradients, then check a complete subsequent reduction numerically.
    n.fill(2)
    x.grad.fill(0)
    y.grad.fill(1)
    with qd.adstack:
        compute(x, y, n)
    compute.grad(x, y, n)
    expected = 4096 * math.cos(0.2) * math.cos(math.sin(0.2))
    assert x.grad[None] == pytest.approx(expected, rel=2e-4)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_multiple_tasks():
    x, _, n = _arrays()
    y = qd.ndarray(qd.f32, shape=8, needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(4):
            value = x[i]
            for j in range(n[i]):
                value = qd.sin(value)
            y[i] = value
        for i in range(4):
            value = x[i]
            for j in range(n[i] + 2):
                value = qd.sin(value)
            y[i + 4] = value

    y.grad.fill(1)
    with qd.adstack:
        compute(x, y, n)
    compute.grad(x, y, n)
    for i in range(4):
        expected = 0.0
        for extra in (0, 2):
            value, derivative = float(x[i]), 1.0
            for _ in range(n[i] + extra):
                derivative *= math.cos(value)
                value = math.sin(value)
            assert y[i + (4 if extra else 0)] == pytest.approx(value, rel=2e-5)
            expected += derivative
        assert x.grad[i] == pytest.approx(expected, rel=2e-5)


@pytest.mark.parametrize("field_iteration", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_dense_fields(field_iteration):
    x = qd.field(qd.f32, shape=(3, 5), needs_grad=True)
    y = qd.field(qd.f32, shape=(3, 5), needs_grad=True)
    n = qd.field(qd.i32, shape=(3, 5))
    for i in range(3):
        for j in range(5):
            x[i, j] = 0.2 + 0.01 * (i * 5 + j)
            n[i, j] = i + j

    @qd.kernel
    def compute():
        if qd.static(field_iteration):
            for i, j in x:
                value = x[i, j]
                for k in range(n[i, j]):
                    if k % 2 == 0:
                        value = qd.sin(value)
                    else:
                        value *= 0.7
                y[i, j] = value
        else:
            for t in range(15):
                i, j = t // 5, t % 5
                value = x[i, j]
                for k in range(n[i, j]):
                    if k % 2 == 0:
                        value = qd.sin(value)
                    else:
                        value *= 0.7
                y[i, j] = value

    for _ in range(2):
        x.grad.fill(0)
        y.grad.fill(1)
        with qd.adstack as scope:
            compute()
        assert scope.capacities[0] > 1
        compute.grad()
        for i in range(3):
            for j in range(5):
                value, derivative = float(x[i, j]), 1.0
                for k in range(n[i, j]):
                    if k % 2 == 0:
                        derivative *= math.cos(value)
                        value = math.sin(value)
                    else:
                        derivative *= 0.7
                        value *= 0.7
                assert y[i, j] == pytest.approx(value, rel=2e-5)
                assert x.grad[i, j] == pytest.approx(derivative, rel=2e-5)
        n[2, 4] += 2

    with qd.adstack:
        compute()
    n[0, 0] = 100
    with pytest.raises(RuntimeError, match="Adstack capacity exceeded"):
        compute.grad()
    n.fill(0)
    x.grad.fill(0)
    y.grad.fill(1)
    with qd.adstack:
        compute()
    compute.grad()
    assert x.grad[0, 0] == pytest.approx(1.0)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_adstack_dense_field_tape():
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)
    x[None] = 0.2

    @qd.kernel
    def compute(n: qd.i32):
        for i in range(128):
            value = x[None]
            for j in range(n):
                value = qd.sin(value)
            loss[None] += value

    with qd.ad.Tape(loss):
        with qd.adstack:
            compute(2)
            compute(3)
    d2 = math.cos(0.2) * math.cos(math.sin(0.2))
    d3 = d2 * math.cos(math.sin(math.sin(0.2)))
    assert x.grad[None] == pytest.approx(128 * (d2 + d3), rel=2e-5)
    assert loss[None] == pytest.approx(
        128 * (math.sin(math.sin(0.2)) + math.sin(math.sin(math.sin(0.2)))), rel=2e-5
    )


@test_utils.test(
    arch=[qd.cpu, qd.cuda], offline_cache=False, require=qd.extension.sparse
)
def test_counted_adstack_rejects_sparse_fields():
    x = qd.field(qd.f32, needs_grad=True)
    qd.root.pointer(qd.i, 4).dense(qd.i, 4).place(x, x.grad)

    @qd.kernel
    def compute():
        for i in range(16):
            x[i] = 1.0

    with pytest.raises(Exception, match="dense fields only"):
        with qd.adstack:
            compute()
