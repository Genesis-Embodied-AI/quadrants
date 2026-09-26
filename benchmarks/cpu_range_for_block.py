"""CPU chunk-size comparison for issue #931.

Run with: python benchmarks/cpu_range_for_block.py --threads 12 --repeats 10
Compilation is excluded. Prints CSV rows with the best and median launch time.
"""

import argparse
import platform
import statistics
import subprocess
import time

import numpy as np

import quadrants as qd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"# commit={commit} platform={platform.platform()}", flush=True)
    print("mode,threads,minimum,body,n,best_ms,median_ms", flush=True)
    references = {}
    for threads, minimum, transform in [
        (1, 512, True),
        (args.threads, 512, True),
        (args.threads, 16, True),
        (args.threads, 1, True),
        (args.threads, 512, False),
    ]:
        qd.init(
            arch=qd.cpu,
            cpu_max_num_threads=threads,
            cpu_min_range_for_block=minimum,
            make_cpu_multithreading_loop=transform,
            offline_cache=False,
            src_ll_cache=False,
        )
        out = qd.field(qd.f32, shape=1 << 20)

        @qd.kernel
        def heavy(n: qd.template()):
            for i in range(n):
                acc = 0.0
                for k in range(1_300_000 // n):
                    acc += qd.sin(acc + k * 1e-6)
                out[i] = acc

        @qd.kernel
        def cheap(n: qd.template()):
            for i in range(n):
                out[i] = i * 0.25 + 1.0

        for body, kernel, sizes in [
            ("heavy", heavy, (200, 512, 513, 1024, 8192)),
            ("cheap", cheap, (200, 8192, 1 << 20)),
        ]:
            for n in sizes:
                kernel(n)
                qd.sync()
                values = out.to_numpy()[:n]
                key = body, n
                if key in references:
                    np.testing.assert_allclose(values, references[key], rtol=1e-6, atol=1e-6)
                else:
                    references[key] = values.copy()
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    kernel(n)
                    qd.sync()
                    samples.append((time.perf_counter() - start) * 1000)
                mode = "chunked" if transform else "runtime"
                print(
                    f"{mode},{threads},{minimum},{body},{n},{min(samples):.6f},{statistics.median(samples):.6f}",
                    flush=True,
                )
        qd.reset()


if __name__ == "__main__":
    main()
