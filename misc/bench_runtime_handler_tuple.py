"""Compare warmed dispatch and launch costs; run from the parent and stacked branches on the same machine.

Use --handlers on the stacked branch to include the handler pipeline. Copy this script to a temporary location
before checking out the parent branch, which predates the script. Native code is identical in the two revisions.
"""

import argparse
import dataclasses
import json
import statistics
import subprocess
import time

import quadrants as qd


@dataclasses.dataclass(frozen=True)
class State:
    values: qd.Tensor


@dataclasses.dataclass(frozen=True)
class CPIC:
    separated: qd.Tensor
    indices: qd.Tensor


@qd.kernel
def scalar(value: qd.i32) -> qd.i32:
    return value + 1


@qd.kernel
def ndarray(values: qd.Tensor):
    values[0] += 1


@qd.kernel
def dataclass_arg(state: State):
    state.values[0] += 1


@qd.func
def add(i, value, data: State):
    return value + data.values[i]


@qd.func
def replace(i, value, cpic: CPIC, rigid: State):
    return rigid.values[cpic.indices[i]] if cpic.separated[i] else value


@qd.data_oriented
class Bound:
    def __init__(self, func, *data):
        self.func = func
        self.data = data

    @qd.func
    def __call__(self, i, value):
        return self.func(i, value, *self.data)


@qd.kernel
def explicit(grid: State, output: qd.Tensor, coupling: State, cpic: CPIC, rigid: State):
    for i in range(output.shape[0]):
        value = add(i, grid.values[i], coupling)
        output[i] = replace(i, value, cpic, rigid)


def measure(callback, iterations, repeats):
    for _ in range(100):
        callback()
    qd.sync()
    results = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            callback()
        qd.sync()
        results.append((time.perf_counter_ns() - start) / iterations / 1000)
    return {"median_us": statistics.median(results), "samples_us": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handlers", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Also measure native kernel time; affects launch timing")
    parser.add_argument("--arch", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    qd.init(arch=getattr(qd, args.arch), offline_cache=False, cpu_max_num_threads=1, kernel_profiler=args.profile)
    values = qd.ndarray(qd.f32, shape=4)
    values.fill(1)
    out = qd.ndarray(qd.f32, shape=4)
    coupling = State(qd.ndarray(qd.f32, shape=4))
    coupling.values.fill(0.5)
    cpic = CPIC(qd.ndarray(qd.i32, shape=4), qd.ndarray(qd.i32, shape=4))
    cpic.separated.fill(0)
    cpic.indices.fill(0)
    rigid = State(qd.ndarray(qd.f32, shape=4))
    rigid.values.fill(-2)
    state = State(values)
    cases = [("scalar", scalar, (1,)), ("ndarray", ndarray, (values,)), ("dataclass", dataclass_arg, (state,))]
    cases.append(("explicit_pipeline", explicit, (state, out, coupling, cpic, rigid)))
    if args.handlers:

        @qd.kernel
        def pipeline(grid: State, output: qd.Tensor, handlers: tuple):
            for i in range(output.shape[0]):
                value = grid.values[i]
                for handler in handlers:
                    value = handler(i, value)
                output[i] = value

        handlers = (Bound(add, coupling), Bound(replace, cpic, rigid))
        cases.append(("handler_pipeline", pipeline, (state, out, handlers)))
    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arch": args.arch,
        "iterations": args.iterations,
        "profiling": args.profile,
        "repeats": args.repeats,
        "cases": {},
    }
    for name, kernel, inputs in cases:
        kernel(*inputs)
        dispatch = measure(lambda: kernel._primal.ensure_compiled(*inputs), args.iterations, args.repeats)
        if args.profile:
            qd.profiler.clear_kernel_profiler_info()
        report["cases"][name] = {
            "dispatch": dispatch,
            "launch": measure(lambda: kernel(*inputs), args.iterations, args.repeats),
            "specializations": len(kernel._primal.materialized_kernels),
        }
        if args.profile:
            report["cases"][name]["native_mean_us"] = (
                qd.profiler.get_kernel_profiler_total_time() * 1e6 / (args.iterations * args.repeats + 100)
            )
    if args.handlers:
        before = len(pipeline._primal.materialized_kernels)
        report["cases"]["fresh_handlers"] = {
            "launch": measure(
                lambda: pipeline(state, out, (Bound(add, coupling), Bound(replace, cpic, rigid))),
                args.iterations,
                args.repeats,
            ),
            "new_specializations": len(pipeline._primal.materialized_kernels) - before,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
