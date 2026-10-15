import gc
import math
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_template_fields_and_keywords():
    x = qd.field(qd.f32, shape=4, needs_grad=True)
    y = qd.field(qd.f32, shape=4, needs_grad=True)
    x.fill(0.2)

    @qd.kernel
    def compute(x: qd.template(), y: qd.template(), n: qd.i32):
        for i in x:
            value = x[i]
            for j in range(n):
                value = qd.sin(value)
            y[i] = value

    with qd.adstack:
        compute(y=y, n=2, x=x)
    y.grad.fill(1)
    compute.grad(x, n=2, y=y)
    assert x.grad[0] == pytest.approx(math.cos(0.2) * math.cos(math.sin(0.2)), rel=2e-5)


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_reusable_out_of_order_records():
    x = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    x[0] = 0.2

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.i32):
        for i in range(1):
            value = x[i]
            for j in range(n):
                value = qd.sin(value)
            y[i] = value

    with qd.adstack as first:
        compute(x, y, 2)
    with qd.adstack as second:
        compute(x, y, 3)
    assert first is not second
    d2 = math.cos(0.2) * math.cos(math.sin(0.2))
    d3 = d2 * math.cos(math.sin(math.sin(0.2)))
    for record, derivative in (
        (first.records[0], d2),
        (second.records[0], d3),
        (first.records[0], d2),
    ):
        x.grad.fill(0)
        y.grad.fill(2)
        record.backward()
        assert x.grad[0] == pytest.approx(2 * derivative, rel=2e-5)
    # Reseed the output gradient consumed by backward; leave the input gradient to accumulate.
    y.grad.fill(2)
    first.records[0].backward()
    assert x.grad[0] == pytest.approx(4 * d2, rel=2e-5)
    with pytest.raises(RuntimeError, match="No pending counted forward"):
        compute.grad(x, y, 2)
    first.records[0].release()
    first.records[0].release()
    with pytest.raises(RuntimeError, match="released"):
        first.records[0].backward()


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_reusable_overflow_and_release():
    x = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    n = qd.ndarray(qd.i32, shape=1)
    x[0], n[0] = 0.2, 2

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(1):
            value = x[i]
            for j in range(n[i]):
                value = qd.sin(value)
            y[i] = value

    with qd.adstack as scope:
        compute(x, y, n)
    record = scope.records[0]
    n[0] = 100
    with pytest.raises(RuntimeError, match="Adstack capacity exceeded"):
        record.backward()
    n[0] = 2
    x.grad.fill(0)
    y.grad.fill(1)
    record.backward()
    assert x.grad[0] == pytest.approx(math.cos(0.2) * math.cos(math.sin(0.2)), rel=2e-5)
    # The invocation retains its arrays until release, even after explicit backward removed it from the queue.
    ref = weakref.ref(n)
    del n
    gc.collect()
    assert ref() is not None
    record.release()
    gc.collect()
    assert ref() is None


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_counted_numpy_argument():
    x = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=1, needs_grad=True)
    n = np.array([2], dtype=np.int32)
    x[0] = 0.2

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
        for i in range(1):
            value = x[i]
            for j in range(n[i]):
                value = qd.sin(value)
            y[i] = value

    with qd.adstack:
        compute(n=n, x=x, y=y)
    y.grad.fill(1)
    compute.grad(x, y, n)
    assert x.grad[0] == pytest.approx(math.cos(0.2) * math.cos(math.sin(0.2)), rel=2e-5)


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_counted_torch_arguments():
    torch = pytest.importorskip("torch")
    device = "cuda" if qd.lang.impl.current_cfg().arch == qd.cuda and torch.cuda.is_available() else "cpu"
    x = torch.full((4,), 0.2, device=device, requires_grad=True)
    y = torch.zeros((4,), device=device, requires_grad=True)

    @qd.kernel
    def compute(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for i in range(4):
            value = x[i]
            for j in range(2):
                value = qd.sin(value)
            y[i] = value

    with qd.adstack as scope:
        compute(y=y, x=x)
    for _ in range(2):
        x.grad = torch.zeros_like(x)
        y.grad = torch.ones_like(y)
        scope.records[0].backward()
        assert x.grad.cpu().numpy() == pytest.approx(np.full(4, math.cos(0.2) * math.cos(math.sin(0.2))), rel=2e-5)


@pytest.mark.parametrize("fastcache", [False, True])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_persistent_cache(tmp_path, fastcache):
    arch = qd.lang.impl.current_cfg().arch.name
    worker = Path(__file__).with_name("adstack_cache_worker.py")
    for warm in (False, True):
        subprocess.run(
            [
                sys.executable,
                str(worker),
                arch,
                str(tmp_path),
                str(int(fastcache)),
                str(int(warm)),
            ],
            check=True,
        )


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.vulkan, qd.metal], offline_cache=False)
def test_counted_class_kernel():
    @qd.data_oriented
    class Model:
        def __init__(self):
            self.x = qd.field(qd.f32, shape=1, needs_grad=True)
            self.y = qd.field(qd.f32, shape=1, needs_grad=True)
            self.x[0] = 0.2

        @qd.kernel
        def compute(self, n: qd.i32):
            for i in self.x:
                value = self.x[i]
                for j in range(n):
                    value = qd.sin(value)
                self.y[i] = value

    model = Model()
    with qd.adstack as scope:
        model.compute(n=2)
    model.y.grad.fill(1)
    scope.records[0].backward()
    assert model.x.grad[0] == pytest.approx(math.cos(0.2) * math.cos(math.sin(0.2)), rel=2e-5)


@test_utils.test(arch=[qd.cpu], offline_cache=False)
def test_counted_record_cannot_cross_reset():
    x = qd.ndarray(qd.f32, shape=1)

    @qd.kernel
    def compute(x: qd.types.ndarray()):
        for i in range(1):
            x[i] = 1.0

    with qd.adstack as scope:
        compute(x)
    qd.reset()
    with pytest.raises(RuntimeError, match="cannot be used after"):
        scope.records[0].backward()
    scope.records[0].release()
