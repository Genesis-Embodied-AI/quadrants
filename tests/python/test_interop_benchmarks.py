"""Standalone Quadrants interop benchmarks.

Exercises the same Python-side bottlenecks as Genesis benchmarks (franka_accessors,
anymal_zero, kinematic, etc.) without any Genesis dependency.  The hot-path operations
are field.to_torch(copy=True/False), field.to_numpy(copy=True/False), field.from_torch(),
and trivial kernel launch overhead.

Run all benchmarks::

    pytest tests/python/test_interop_benchmarks.py -v -s

Run a specific benchmark::

    pytest tests/python/test_interop_benchmarks.py -v -s -k "zerocopy and cuda"

Write results to a file for W&B upload::

    pytest tests/python/test_interop_benchmarks.py -v -s --bench-out results.txt
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import quadrants as qd

# ---------------------------------------------------------------------------
# Pytest options and marks
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.benchmarks]

N_LINKS = 11
N_DOFS = 9
WARMUP_S = 5.0
RECORD_S = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pprint_oneline(data: dict, delimiter: str = " \t| ") -> str:
    items = []
    for k, v in data.items():
        if isinstance(v, float):
            v = f"{v:.2f}"
        items.append(f"{k}={v}")
    return delimiter.join(items)


def _make_fields(n_links: int, n_dofs: int, n_envs: int) -> dict:
    """Create fields shaped like rigid-solver state in Genesis.

    Genesis SNode layout is (link_idx, env_idx, vec) — the first axis is the
    "structural" index, the second is the batch (env) dimension.
    """
    shapes = {
        "pos": (n_links, n_envs, 3),
        "quat": (n_links, n_envs, 4),
        "vel": (n_links, n_envs, 3),
        "ang": (n_links, n_envs, 3),
        "dofs_pos": (n_dofs, n_envs),
        "dofs_vel": (n_dofs, n_envs),
    }
    fields = {}
    for name, shape in shapes.items():
        f = qd.field(qd.f32, shape=shape)
        f.fill(0.0)
        fields[name] = f
    qd.sync()
    return fields


def _run_bench(step_fn, *, warmup_s: float = WARMUP_S, record_s: float = RECORD_S) -> dict:
    """Run *step_fn* in a warmup+record loop and return timing results."""
    qd.sync()

    # One cold step + sync to flush lazy init / JIT
    step_fn()
    qd.sync()

    # Warmup
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warmup_s:
        step_fn()
    qd.sync()

    # Record
    n_steps = 0
    t0 = time.perf_counter()
    while True:
        step_fn()
        n_steps += 1
        if time.perf_counter() - t0 >= record_s:
            qd.sync()
            break
    elapsed = time.perf_counter() - t0
    steps_per_sec = n_steps / elapsed
    us_per_step = 1e6 * elapsed / n_steps
    return {"steps_per_sec": round(steps_per_sec, 1), "us_per_step": round(us_per_step, 2)}


# ---------------------------------------------------------------------------
# Benchmark scenarios (each returns a result dict)
# ---------------------------------------------------------------------------


def _bench_to_torch_zerocopy(fields: dict) -> dict:
    """Read all fields via DLPack zero-copy each step — the franka_accessors hot path."""
    import torch

    flist = list(fields.values())

    def step():
        for f in flist:
            torch.utils.dlpack.from_dlpack(f.to_dlpack())

    return _run_bench(step)


def _bench_to_torch_copy(fields: dict) -> dict:
    """Read all fields via to_torch(copy=True) each step."""
    flist = list(fields.values())

    def step():
        for f in flist:
            f.to_torch()

    return _run_bench(step)


def _bench_to_numpy_copy(fields: dict) -> dict:
    """Read all fields via to_numpy() each step."""
    flist = list(fields.values())

    def step():
        for f in flist:
            f.to_numpy()

    return _run_bench(step)


def _bench_from_torch(fields: dict) -> dict:
    """Write torch tensors back into fields each step — the control_dofs/set_qpos path."""
    tensors = {name: f.to_torch() for name, f in fields.items()}

    def step():
        for name, f in fields.items():
            f.from_torch(tensors[name])

    return _run_bench(step)


def _bench_roundtrip(fields: dict) -> dict:
    """Read via DLPack zero-copy + write via from_torch each step."""
    import torch

    flist = list(fields.values())

    def step():
        for f in flist:
            t = torch.utils.dlpack.from_dlpack(f.to_dlpack())
            f.from_torch(t)

    return _run_bench(step)


def _bench_kernel_launch_only(fields: dict) -> dict:
    """Launch a trivial kernel each step — baseline kernel overhead measurement."""
    f = fields["pos"]

    @qd.kernel
    def _noop_kernel(x: qd.types.ndarray()):
        pass

    def step():
        _noop_kernel(f)

    return _run_bench(step)


def _bench_to_torch_zerocopy_single(fields: dict) -> dict:
    """Read a single field via DLPack zero-copy — isolate per-field DLPack cost."""
    import torch

    f = fields["pos"]

    def step():
        torch.utils.dlpack.from_dlpack(f.to_dlpack())

    return _run_bench(step)


def _bench_to_torch_copy_single(fields: dict) -> dict:
    """Read a single field via to_torch(copy=True) — isolate per-field kernel copy cost."""
    f = fields["pos"]

    def step():
        f.to_torch()

    return _run_bench(step)


SCENARIOS = {
    "to_torch_zerocopy": _bench_to_torch_zerocopy,
    "to_torch_copy": _bench_to_torch_copy,
    "to_numpy_copy": _bench_to_numpy_copy,
    "from_torch": _bench_from_torch,
    "roundtrip": _bench_roundtrip,
    "kernel_launch_only": _bench_kernel_launch_only,
    "to_torch_zerocopy_single": _bench_to_torch_zerocopy_single,
    "to_torch_copy_single": _bench_to_torch_copy_single,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bench_writer(request):
    """Yields a callable that writes one result line to --bench-out and stdout."""
    report_path = Path(request.config.getoption("--bench-out"))

    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    report_name = "-".join(filter(None, (report_path.stem, worker_id)))
    report_path = report_path.with_name(f"{report_name}{report_path.suffix}")

    if report_path.exists():
        report_path.unlink()
    fd = open(report_path, "w")

    def _write(msg: str):
        print(msg, flush=True)
        print(msg, file=fd, flush=True)

    yield _write
    fd.close()


@pytest.fixture(scope="module")
def fields_cuda_4096():
    qd.init(arch=qd.cuda, default_fp=qd.f32, device_memory_GB=1)
    fields = _make_fields(N_LINKS, N_DOFS, 4096)
    yield fields
    qd.reset()


@pytest.fixture(scope="module")
def fields_cuda_30000():
    qd.init(arch=qd.cuda, default_fp=qd.f32, device_memory_GB=1)
    fields = _make_fields(N_LINKS, N_DOFS, 30000)
    yield fields
    qd.reset()


@pytest.fixture(scope="module")
def fields_cpu_4096():
    qd.init(arch=qd.cpu, default_fp=qd.f32)
    fields = _make_fields(N_LINKS, N_DOFS, 4096)
    yield fields
    qd.reset()


# ---------------------------------------------------------------------------
# Parametrized benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario, arch, n_envs",
    [
        *(("to_torch_zerocopy", "cuda", n) for n in (4096, 30000)),
        *(("to_torch_copy", "cuda", n) for n in (4096, 30000)),
        *(("to_numpy_copy", "cuda", n) for n in (4096, 30000)),
        *(("from_torch", "cuda", n) for n in (4096, 30000)),
        *(("roundtrip", "cuda", n) for n in (4096, 30000)),
        *(("kernel_launch_only", "cuda", n) for n in (4096,)),
        *(("to_torch_zerocopy_single", "cuda", n) for n in (4096, 30000)),
        *(("to_torch_copy_single", "cuda", n) for n in (4096, 30000)),
        ("to_torch_copy", "cpu", 4096),
        ("to_numpy_copy", "cpu", 4096),
        ("from_torch", "cpu", 4096),
        ("to_torch_zerocopy", "cpu", 4096),
        ("to_torch_zerocopy_single", "cpu", 4096),
        ("kernel_launch_only", "cpu", 4096),
    ],
)
def test_interop_speed(bench_writer, request, scenario, arch, n_envs):
    fixture_name = f"fields_{arch}_{n_envs}"
    fields = request.getfixturevalue(fixture_name)

    result = SCENARIOS[scenario](fields)

    hparams = {"scenario": scenario, "arch": arch, "n_envs": n_envs, "n_links": N_LINKS, "n_dofs": N_DOFS}
    bench_writer(_pprint_oneline({**hparams, **result}))
