import math

import numpy as np
import pytest

import quadrants as qd
from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], ad_stack_replay_count=True, ad_stack_experimental_enabled=True)
def test_backward_only_remeasures_current_inputs():
    x = qd.ndarray(qd.f32, shape=4, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=4, needs_grad=True)
    n = qd.ndarray(qd.i32, shape=1)
    x.fill(0.2)
    y.fill(17)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n[0]):
                value = qd.sin(value)
            y[i] = value

    for depth in (2, 9, 1):
        n[0] = depth
        x.grad.fill(0)
        y.grad.fill(1)
        compute.grad(y=y, n=n, x=x)
        value, derivative = 0.2, 1.0
        for _ in range(depth):
            derivative *= math.cos(value)
            value = math.sin(value)
        np.testing.assert_allclose(x.grad.to_numpy(), derivative, rtol=2e-5)
        np.testing.assert_array_equal(y.to_numpy(), np.full(4, 17))
        np.testing.assert_allclose(x.to_numpy(), 0.2)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], ad_stack_replay_count=True, ad_stack_experimental_enabled=True)
def test_replay_dense_fields_tape():
    x = qd.field(qd.f32, shape=4, needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)
    n = qd.field(qd.i32, shape=())
    x.fill(0.2)
    n[None] = 3

    @qd.kernel
    def compute():
        for i in x:
            value = x[i]
            for j in range(n[None]):
                value = qd.sin(value)
            loss[None] += value

    with qd.ad.Tape(loss):
        compute()
    value, derivative = 0.2, 1.0
    for _ in range(3):
        derivative *= math.cos(value)
        value = math.sin(value)
    assert loss[None] == pytest.approx(4 * value, rel=2e-5)
    np.testing.assert_allclose(x.grad.to_numpy(), derivative, rtol=2e-5)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], ad_stack_replay_count=True, ad_stack_experimental_enabled=True)
def test_count_rejects_residual_primal_mutation():
    x = qd.field(qd.f32, shape=1, needs_grad=True)
    y = qd.field(qd.f32, shape=1, needs_grad=True)
    counter = qd.field(qd.i32, shape=1)

    @qd.kernel
    def compute():
        for i in range(1):
            counter[i] += 1
            y[i] = x[i] * x[i]

    with pytest.raises(Exception, match="non-differentiable buffer"):
        compute.grad()
    assert counter[0] == 0
