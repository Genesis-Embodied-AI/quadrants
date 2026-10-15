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
            value = x[i]
            for j in range(counter[i]):
                value = qd.sin(value)
            y[i] = value

    with pytest.raises(Exception, match="non-differentiable buffer"):
        compute.grad()
    assert counter[0] == 0


@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], ad_stack_replay_count=True, ad_stack_experimental_enabled=True)
def test_backward_only_persistent_cache(tmp_path, fastcache):
    import subprocess
    import sys
    from pathlib import Path

    worker = Path(__file__).with_name("adstack_replay_cache_worker.py")
    for warm in (False, True):
        subprocess.run(
            [
                sys.executable,
                str(worker),
                qd.lang.impl.current_cfg().arch.name,
                str(tmp_path),
                str(int(fastcache)),
                str(int(warm)),
            ],
            check=True,
            timeout=180,
        )


@pytest.mark.parametrize("read_modify_write", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], ad_stack_replay_count=True, ad_stack_experimental_enabled=True)
def test_write_only_array_output_and_alias_guard(read_modify_write):
    x = qd.ndarray(qd.f32, shape=2, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=2, needs_grad=True)
    n = qd.ndarray(qd.i32, shape=2)
    scratch = qd.ndarray(qd.i32, shape=2)
    x.fill(0.2)
    n.fill(3)
    scratch.fill(7)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray(), scratch: qd.types.ndarray()):
        for i in range(2):
            value = x[i]
            for j in range(n[i]):
                value = qd.sin(value)
            if qd.static(read_modify_write):
                scratch[i] = scratch[i] | 16
            else:
                scratch[i] = 19
            y[i] = value

    # The count variant must not write even non-differentiable outputs. The normal backward still does.
    from quadrants.lang.adstack_replay import _Record, _variant

    variant = _variant(compute._adjoint, 1)
    record = _Record(qd.ndarray(qd.i32 if qd.lang.impl.current_cfg().arch == qd.vulkan else qd.i64, shape=2))
    variant.adstack_record = record
    try:
        variant(x, y, n, scratch)
        np.testing.assert_array_equal(scratch.to_numpy(), [7, 7])
    finally:
        variant.adstack_record = None
    x.grad.fill(0)
    y.grad.fill(1)
    compute.grad(x, y, n, scratch)
    np.testing.assert_array_equal(scratch.to_numpy(), [23 if read_modify_write else 19] * 2)
    value, derivative = 0.2, 1.0
    for _ in range(3):
        derivative *= math.cos(value)
        value = math.sin(value)
    np.testing.assert_allclose(x.grad.to_numpy(), derivative, rtol=2e-5)
    with pytest.raises(RuntimeError, match="aliased non-differentiable"):
        compute.grad(x, y, n, n)
    np.testing.assert_array_equal(n.to_numpy(), [3, 3])


@pytest.mark.parametrize("debug", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan], ad_stack_replay_count=True, ad_stack_experimental_enabled=True)
def test_replay_overflow_with_reduction(monkeypatch, debug):
    from quadrants.lang import adstack_replay

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

    original_variant = adstack_replay._variant

    def change_one_lane_after_count(kernel, mode):
        if mode == 2:
            n[0] = 100
        return original_variant(kernel, mode)

    monkeypatch.setattr(adstack_replay, "_variant", change_one_lane_after_count)
    y.grad.fill(1)
    with pytest.raises(RuntimeError, match="Adstack capacity exceeded"):
        compute.grad(x, y, n)
    monkeypatch.undo()
    n.fill(2)
    x.grad.fill(0)
    y.grad.fill(1)
    compute.grad(x, y, n)
    assert x.grad[None] == pytest.approx(4096 * math.cos(0.2) * math.cos(math.sin(0.2)), rel=2e-4)
