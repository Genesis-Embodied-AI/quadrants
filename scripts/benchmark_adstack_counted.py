"""Measure forward counting on a synthetic nonlinear loop.

Run in a built Quadrants environment:
    python scripts/benchmark_adstack_counted.py --arch cuda --size 4096 --depth 32 --repeats 30

Forward and backward synchronize at the timing boundaries. This includes Python launch overhead and the prototype's
measurement/error synchronization; it is an end-to-end measurement, not isolated device arithmetic time.
"""

import argparse
import json
import statistics
import subprocess
import time

import numpy as np

import quadrants as qd


def workload(x: qd.types.ndarray(), y: qd.types.ndarray(), n: qd.types.ndarray()):
    for i in range(x.shape[0]):
        value = x[i]
        for j in range(n[i]):
            value = qd.sin(value) + 0.01
        y[i] = value


def measure(kernel, x, y, n, repeats):
    samples = []
    first_pair_ms = None
    capacity = None
    for iteration in range(repeats + 3):
        x.grad.fill(0)
        y.grad.fill(1)
        qd.sync()
        start = time.perf_counter()
        with qd.adstack as scope:
            kernel(x, y, n)
        capacity = scope.capacities[0]
        qd.sync()
        forward_done = time.perf_counter()
        kernel.grad(x, y, n)
        qd.sync()
        done = time.perf_counter()
        if iteration == 0:
            first_pair_ms = (done - start) * 1000
        if iteration >= 3:
            samples.append(((forward_done - start) * 1000, (done - forward_done) * 1000, (done - start) * 1000))
    return {
        "forward_ms": statistics.median(row[0] for row in samples),
        "backward_ms": statistics.median(row[1] for row in samples),
        "pair_ms": statistics.median(row[2] for row in samples),
        "first_pair_ms": first_pair_ms,
        "measured_uniform_capacity": capacity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cuda", "cpu", "vulkan", "metal"), default="cuda")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.size <= 0 or args.depth < 0 or args.repeats <= 0:
        parser.error("size and repeats must be positive; depth must be nonnegative")
    qd.init(arch=getattr(qd, args.arch), offline_cache=False)
    counted = qd.kernel(workload)
    x = qd.ndarray(qd.f32, shape=args.size, needs_grad=True)
    y = qd.ndarray(qd.f32, shape=args.size, needs_grad=True)
    n = qd.ndarray(qd.i32, shape=args.size)
    x.fill(0.2)
    n.fill(args.depth)
    reference_y = np.full(args.size, 0.2, dtype=np.float64)
    reference_grad = np.ones(args.size, dtype=np.float64)
    for _ in range(args.depth):
        reference_grad *= np.cos(reference_y)
        reference_y = np.sin(reference_y) + 0.01
    counted_result = measure(counted, x, y, n, args.repeats)
    np.testing.assert_allclose(y.to_numpy(), reference_y, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(x.grad.to_numpy(), reference_grad, rtol=2e-5, atol=1e-7)
    print(
        json.dumps(
            {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "configuration": vars(args),
                "counted": counted_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
