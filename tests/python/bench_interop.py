"""Standalone Quadrants interop benchmarks.

Exercises the same Python-side bottlenecks as Genesis benchmarks (franka_accessors,
anymal_zero, kinematic, etc.) without any Genesis dependency.  The hot-path operations
are field.to_torch(copy=True/False), field.to_numpy(copy=True/False), field.from_torch(),
and trivial kernel launch overhead.

Run locally::

    python tests/python/bench_interop.py --arch cuda --n-envs 4096 --json results.json

Upload results to W&B (CI)::

    python tests/python/bench_interop.py --arch cuda --n-envs 4096 --json results.json
    python tests/python/upload_bench_to_wandb.py --in-file results.json --project quadrants-interop-benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import quadrants as qd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fields(arch, n_links: int, n_dofs: int, n_envs: int):
    """Create fields shaped like rigid-solver state in Genesis.

    Genesis SNode layout is (link_idx, env_idx, vec) — the first axis is the
    "structural" index, the second is the batch (env) dimension.
    """
    qd.init(arch=arch, default_fp=qd.f32, device_memory_GB=1)

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


def _bench_loop(step_fn, *, warmup_s: float = 5.0, record_s: float = 5.0) -> dict:
    """Run *step_fn* in a warmup+record loop and return timing results."""
    qd.sync()

    # One cold step + sync to flush lazy init
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
# Benchmark scenarios
# ---------------------------------------------------------------------------


def bench_to_torch_zerocopy(fields: dict) -> dict:
    """Read all fields via to_torch(copy=False) each step — the franka_accessors hot path."""
    import torch

    flist = list(fields.values())

    def step():
        for f in flist:
            torch.utils.dlpack.from_dlpack(f.to_dlpack())

    return _bench_loop(step)


def bench_to_torch_copy(fields: dict) -> dict:
    """Read all fields via to_torch(copy=True) each step."""
    flist = list(fields.values())

    def step():
        for f in flist:
            f.to_torch()

    return _bench_loop(step)


def bench_to_numpy_copy(fields: dict) -> dict:
    """Read all fields via to_numpy() each step."""
    flist = list(fields.values())

    def step():
        for f in flist:
            f.to_numpy()

    return _bench_loop(step)


def bench_from_torch(fields: dict) -> dict:
    """Write torch tensors back into fields each step — the control_dofs/set_qpos path."""
    tensors = {name: f.to_torch() for name, f in fields.items()}

    def step():
        for name, f in fields.items():
            f.from_torch(tensors[name])

    return _bench_loop(step)


def bench_roundtrip(fields: dict) -> dict:
    """Read via to_torch(copy=False) + write via from_torch each step."""
    import torch

    flist = list(fields.values())

    def step():
        for f in flist:
            t = torch.utils.dlpack.from_dlpack(f.to_dlpack())
            f.from_torch(t)

    return _bench_loop(step)


def bench_kernel_launch_only(fields: dict) -> dict:
    """Launch a trivial kernel each step — baseline kernel overhead measurement."""
    f = fields["pos"]

    @qd.kernel
    def _noop_kernel(x: qd.types.ndarray()):
        pass

    def step():
        _noop_kernel(f)

    return _bench_loop(step)


def bench_to_torch_zerocopy_single(fields: dict) -> dict:
    """Read a single field via to_torch(copy=False) — isolate per-field DLPack cost."""
    import torch

    f = fields["pos"]

    def step():
        torch.utils.dlpack.from_dlpack(f.to_dlpack())

    return _bench_loop(step)


def bench_to_torch_copy_single(fields: dict) -> dict:
    """Read a single field via to_torch(copy=True) — isolate per-field kernel copy cost."""
    f = fields["pos"]

    def step():
        f.to_torch()

    return _bench_loop(step)


BENCHMARKS = {
    "to_torch_zerocopy": bench_to_torch_zerocopy,
    "to_torch_copy": bench_to_torch_copy,
    "to_numpy_copy": bench_to_numpy_copy,
    "from_torch": bench_from_torch,
    "roundtrip": bench_roundtrip,
    "kernel_launch_only": bench_kernel_launch_only,
    "to_torch_zerocopy_single": bench_to_torch_zerocopy_single,
    "to_torch_copy_single": bench_to_torch_copy_single,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_arch(name: str):
    return getattr(qd, name)


def main():
    parser = argparse.ArgumentParser(description="Quadrants interop benchmarks")
    parser.add_argument("--arch", default="cuda", help="Quadrants arch name (cuda, cpu, vulkan, ...)")
    parser.add_argument("--n-envs", type=int, default=4096, help="Number of environments (batch dim)")
    parser.add_argument("--n-links", type=int, default=11, help="Number of links (structural dim)")
    parser.add_argument("--n-dofs", type=int, default=9, help="Number of DOFs")
    parser.add_argument("--warmup", type=float, default=5.0, help="Warmup duration in seconds")
    parser.add_argument("--record", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="Specific benchmarks to run (default: all)")
    parser.add_argument("--json", default=None, help="Write results to JSON file")
    args = parser.parse_args()

    arch = _resolve_arch(args.arch)
    fields = _make_fields(arch, args.n_links, args.n_dofs, args.n_envs)

    bench_names = args.benchmarks or list(BENCHMARKS.keys())
    all_results = []

    for name in bench_names:
        if name not in BENCHMARKS:
            print(f"Unknown benchmark: {name}", file=sys.stderr)
            continue
        print(f"Running {name} ...", end=" ", flush=True)
        result = BENCHMARKS[name](fields)
        result["benchmark"] = name
        result["arch"] = args.arch
        result["n_envs"] = args.n_envs
        result["n_links"] = args.n_links
        result["n_dofs"] = args.n_dofs
        all_results.append(result)
        print(f"{result['steps_per_sec']:.0f} steps/s  ({result['us_per_step']:.1f} us/step)")

    qd.reset()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.json}")

    return all_results


if __name__ == "__main__":
    main()
