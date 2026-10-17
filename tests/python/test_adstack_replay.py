import math

import numpy as np
import pytest

import quadrants as qd
from quadrants.types.enums import AdstackMode

from tests import test_utils


@test_utils.test(arch=[qd.vulkan, qd.metal])
def test_adjoint_buffer_load_survives_clear():
    upstream = qd.ndarray(qd.f32, shape=17)
    downstream = qd.ndarray(qd.f32, shape=17)
    seeds = np.linspace(0.25, 2.0, 17, dtype=np.float32)
    upstream.from_numpy(seeds)
    downstream.fill(0)

    @qd.kernel
    def transfer(upstream: qd.types.ndarray(), downstream: qd.types.ndarray()):
        for i in range(upstream.shape[0]):
            # Reverse AD consumes an output adjoint before clearing its slot. Shader load forwarding must preserve the
            # pre-clear value, including on MoltenVK's Metal shader compiler.
            value = upstream[i]
            upstream[i] = 0.0
            downstream[i] += value

    transfer(upstream, downstream)
    np.testing.assert_array_equal(downstream.to_numpy(), seeds)
    np.testing.assert_array_equal(upstream.to_numpy(), np.zeros_like(seeds))


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
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


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
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


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
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
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
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
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
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

    variant = _variant(compute._adjoint, AdstackMode.COUNT)
    record = _Record(
        qd.ndarray(qd.i32 if qd.lang.impl.current_cfg().arch in (qd.vulkan, qd.metal) else qd.i64, shape=2)
    )
    variant.adstack_record = record
    try:
        x.grad.fill(3)
        y.grad.fill(5)
        variant(x, y, n, scratch)
        np.testing.assert_array_equal(scratch.to_numpy(), [7, 7])
        np.testing.assert_array_equal(x.grad.to_numpy(), [3, 3])
        np.testing.assert_array_equal(y.grad.to_numpy(), [5, 5])
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
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
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
        if mode == AdstackMode.REVERSE:
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


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
def test_replay_disjoint_frame_outputs():
    x = qd.field(qd.f32, shape=4, needs_grad=True)
    y = qd.field(qd.f32, shape=4, needs_grad=True)
    active = qd.field(qd.i32, shape=(3, 4))
    x.fill(0.2)
    active.fill(1)

    @qd.kernel
    def compute(frame: qd.i32):
        for i in range(4):
            value = x[i]
            if active[frame, i]:
                for j in range(3):
                    value = qd.sin(value)
            active[frame + 1, i] = active[frame, i]
            y[i] = value

    from quadrants.lang.adstack_replay import _Record, _variant

    active[1, 0] = 7
    variant = _variant(compute._adjoint, AdstackMode.COUNT)
    variant.adstack_record = _Record(
        qd.ndarray(qd.i32 if qd.lang.impl.current_cfg().arch in (qd.vulkan, qd.metal) else qd.i64, shape=2)
    )
    try:
        variant(0)
        assert active[1, 0] == 7
    finally:
        variant.adstack_record = None
    y.grad.fill(1)
    compute.grad(0)
    assert active[1, 0] == 1
    value, derivative = 0.2, 1.0
    for _ in range(3):
        derivative *= math.cos(value)
        value = math.sin(value)
    np.testing.assert_allclose(x.grad.to_numpy(), derivative, rtol=2e-5)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
def test_replay_rejects_cross_iteration_output_dependency():
    x = qd.field(qd.f32, shape=4, needs_grad=True)
    y = qd.field(qd.f32, shape=4, needs_grad=True)
    flags = qd.field(qd.i32, shape=5)
    flags.fill(1)

    @qd.kernel
    def compute():
        for i in range(4):
            value = x[i]
            for j in range(flags[i]):
                value = qd.sin(value)
            flags[i + 1] = 7
            y[i] = value

    with pytest.raises(RuntimeError, match="non-differentiable buffer"):
        compute.grad()
    np.testing.assert_array_equal(flags.to_numpy(), np.ones(5))


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_replay_torch_arguments_and_overlapping_views():
    torch = pytest.importorskip("torch")
    device = "cuda" if qd.lang.impl.current_cfg().arch == qd.cuda else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA-enabled PyTorch is unavailable")
    x = torch.full((4,), 0.2, device=device, requires_grad=True)
    y = torch.full((4,), 17.0, device=device, requires_grad=True)
    n = torch.full((4,), 2, dtype=torch.int32, device=device)
    scratch = torch.zeros(4, dtype=torch.int32, device=device)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray(), scratch: qd.types.ndarray()):
        for i in range(4):
            value = x[i]
            for j in range(n[i]):
                value = qd.sin(value)
            y[i] = value
            scratch[i] = 1

    y.grad = torch.ones_like(y)
    compute.grad(scratch=scratch, n=n, y=y, x=x)
    torch.testing.assert_close(x.grad, torch.full_like(x, math.cos(0.2) * math.cos(math.sin(0.2))))
    torch.testing.assert_close(y, torch.full_like(y, 17))
    storage = torch.full((5,), 2, dtype=torch.int32, device=device)
    with pytest.raises(RuntimeError, match="aliased non-differentiable"):
        compute.grad(x, y, storage[:4], storage[1:])
    torch.testing.assert_close(storage, torch.full_like(storage, 2))


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
def test_measurement_failure_does_not_fall_back(monkeypatch):
    import importlib

    replay = importlib.import_module("quadrants.lang.adstack_replay")
    x = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    x.fill(0.2)
    y.fill(17)
    x.grad.fill(0)
    y.grad.fill(1)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.i32):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n):
                value = qd.sin(value)
            y[i] = value

    def fail_readback(*args):
        raise RuntimeError("measurement unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(replay, "_read_measurement", fail_readback)
        with pytest.raises(RuntimeError, match="measurement unavailable"):
            compute.grad(x, y, 3)
    np.testing.assert_array_equal(x.grad.to_numpy(), [0])
    np.testing.assert_array_equal(y.grad.to_numpy(), [1])
    np.testing.assert_array_equal(y.to_numpy(), [17])
    # A failed measurement leaves scratch and dispatch state reusable, but never runs an unmeasured reverse.
    compute.grad(x, y, 3)
    expected = math.cos(0.2) * math.cos(math.sin(0.2)) * math.cos(math.sin(math.sin(0.2)))
    np.testing.assert_allclose(x.grad.to_numpy(), [expected], rtol=2e-5)


@pytest.mark.parametrize("block_dim", [16, 32, 128])
@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=[qd.cuda, qd.amdgpu])
def test_replay_stack_layout_divergent_depths(block_dim, dtype):
    n = 97
    x = qd.ndarray(dtype, shape=n, needs_grad=True)
    y = qd.ndarray(dtype, shape=n, needs_grad=True)
    values = np.linspace(0.1, 0.9, n, dtype=np.float64 if dtype == qd.f64 else np.float32)
    x.from_numpy(values)
    y.grad.fill(1)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray()):
        qd.loop_config(block_dim=block_dim)
        for i in range(n):
            value = x[i]
            for j in range(i % 11 + 1):
                value = qd.sin(value)
            y[i] = value

    compute.grad(x, y)
    expected = np.ones(n)
    for i in range(n):
        value = float(values[i])
        for _ in range(i % 11 + 1):
            expected[i] *= math.cos(value)
            value = math.sin(value)
    np.testing.assert_allclose(x.grad.to_numpy(), expected, rtol=2e-5)


@pytest.mark.parametrize("affects_depth", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], cfg_optimization=False)
def test_count_prunes_only_unused_local_recurrences(affects_depth):
    x = qd.field(qd.f32, shape=1, needs_grad=True)
    y = qd.field(qd.f32, shape=1, needs_grad=True)
    scratch = qd.field(qd.f32, shape=1)
    x[0] = 0.2
    scratch[0] = 1.0
    y.grad[0] = 1.0

    @qd.kernel
    def compute():
        for i in range(1):
            scratch[i] = 2.0
            unused = scratch[i]
            for j in range(3):
                unused = unused * unused + 1.0
            scratch[i] = unused
            depth = 3
            if qd.static(affects_depth):
                depth = qd.cast(unused, qd.i32)
            value = x[i]
            for j in range(depth):
                value = qd.sin(value)
            y[i] = value

    if affects_depth:
        # A recurrence used by the count must remain live and trigger the replay-dependency guard.
        with pytest.raises(RuntimeError, match="non-differentiable buffer"):
            compute.grad()
        assert scratch[0] == 1.0
        assert x.grad[0] == 0.0
        assert y.grad[0] == 1.0
    else:
        compute.grad()
        expected = math.cos(0.2) * math.cos(math.sin(0.2)) * math.cos(math.sin(math.sin(0.2)))
        assert x.grad[0] == pytest.approx(expected, rel=2e-5)


@pytest.mark.parametrize("force_sync", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
def test_caught_overflow_process_teardown(force_sync):
    import subprocess
    import sys
    from pathlib import Path

    worker = Path(__file__).with_name("adstack_replay_overflow_worker.py")
    result = subprocess.run(
        [sys.executable, str(worker), qd.lang.impl.current_cfg().arch.name, str(int(force_sync))],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OVERFLOW_TEARDOWN_OK" in result.stdout


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
def test_interleaved_kernel_measurements_are_independent():
    x = qd.ndarray(qd.f32, shape=4, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=4, needs_grad=True)
    z = qd.ndarray(qd.f32, shape=4, needs_grad=True)
    n = qd.ndarray(qd.i32, shape=1)
    x.fill(0.2)

    @qd.kernel
    def first(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(n[0]):
                value = qd.sin(value)
            y[i] = value

    @qd.kernel
    def second(x: qd.types.ndarray(), z: qd.types.ndarray(), depth: qd.i32):
        for i in range(x.shape[0]):
            value = x[i]
            for j in range(depth):
                value = value * 0.9 + 0.01
            z[i] = value

    for depth in (2, 19, 1):
        n[0] = depth
        x.grad.fill(0)
        y.grad.fill(1)
        first.grad(x, y, n)
        value, derivative = 0.2, 1.0
        for _ in range(depth):
            derivative *= math.cos(value)
            value = math.sin(value)
        np.testing.assert_allclose(x.grad.to_numpy(), derivative, rtol=2e-5)
        first_capacity = first._adjoint._adstack_replay_scratch.capacity
        z.grad.fill(1)
        other_depth = 23 - depth
        second.grad(x, z, other_depth)
        np.testing.assert_allclose(x.grad.to_numpy(), derivative + 0.9**other_depth, rtol=2e-5)
        assert first._adjoint._adstack_replay_scratch.capacity == first_capacity
        assert first._adjoint._adstack_replay_scratch.buffer is not second._adjoint._adstack_replay_scratch.buffer
