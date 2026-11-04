"""Measure runtime schema inference and warmed dispatch for nested Tensor-bearing dataclasses.

Run from a Quadrants checkout with the implementation under test installed in the active environment:
    python misc/bench_runtime_dataclass_schema.py --label before

The source digest identifies the installed resolver independently of the checkout containing this benchmark.
"""

import argparse
import dataclasses
import hashlib
import inspect
import json
import statistics
import time
from importlib.metadata import version

import quadrants as qd
from quadrants.lang import _runtime_dataclass


def measure(call, iterations, repeats):
    for _ in range(20):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        samples.append((time.perf_counter_ns() - start) / iterations / 1000)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    qd.init(arch=qd.cpu, cpu_max_num_threads=1, offline_cache=False)
    report = {
        "label": args.label,
        "quadrants_version": version("quadrants"),
        "resolver_sha256": hashlib.sha256(inspect.getsource(_runtime_dataclass).encode()).hexdigest(),
        "iterations": args.iterations,
        "repeats": args.repeats,
        "cases": {},
    }
    for groups, width in ((1, 1), (1, 32), (8, 32)):
        Group = dataclasses.make_dataclass("Group", [(f"value{i}", object) for i in range(width)], frozen=True)
        State = dataclasses.make_dataclass("State", [(f"group{i}", object) for i in range(groups)], frozen=True)
        array = qd.ndarray(qd.f32, shape=1)
        array[0] = 2
        state = State(*(Group(*(qd.Tensor(array) for _ in range(width))) for _ in range(groups)))

        @qd.kernel
        def read(state: qd.RuntimeTyped) -> qd.f32:
            return state.group0.value0[0]

        assert read(state) == 2
        report["cases"][f"{groups}x{width}"] = {
            "schema": measure(
                lambda: _runtime_dataclass.resolve_dataclass(state, "state"), args.iterations, args.repeats
            ),
            "dispatch": measure(lambda: read._primal.ensure_compiled(state), args.iterations, args.repeats),
            "specializations": len(read._primal.materialized_kernels),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
