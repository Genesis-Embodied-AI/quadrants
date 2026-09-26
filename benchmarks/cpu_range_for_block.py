"""CPU range-for chunk-size comparison for issue #931.

Each iteration does the same total amount of work regardless of the loop length ``n`` (``TOTAL_WORK // n`` dependent
``sin`` evaluations), so timings isolate how well the parallel CPU range-for spreads a short, heavy-bodied loop across
the thread pool. With the default ``cpu_min_range_for_block=512`` a loop of <= 512 iterations runs on one thread;
lowering the floor lets every thread engage.

Run with, e.g.::

    python benchmarks/cpu_range_for_block.py --threads 12 --repeats 10

Compilation is excluded (each config is warmed up before timing). Prints CSV rows to stdout.
"""

import argparse
import platform
import statistics
import time

import quadrants as qd

TOTAL_WORK = 1_300_000
LOOP_SIZES = (200, 512, 513, 1024, 8192)


def bench_one(n_threads, minimum, repeats):
    qd.init(arch=qd.cpu, cpu_max_num_threads=n_threads, cpu_min_range_for_block=minimum, offline_cache=False)
    out = qd.field(qd.f32, shape=(max(LOOP_SIZES),))

    @qd.kernel
    def heavy_body(n: qd.template()):
        for i in range(n):
            acc = 0.0
            for k in range(TOTAL_WORK // n):
                acc += qd.sin(acc + k * 1e-6)
            out[i] = acc

    rows = []
    for n in LOOP_SIZES:
        heavy_body(n)
        qd.sync()  # warm up / exclude compilation
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            heavy_body(n)
            qd.sync()
            samples.append(time.perf_counter() - t0)
        rows.append((n, 1e3 * min(samples), 1e3 * statistics.median(samples)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")

    print(f"# platform={platform.platform()} threads={args.threads} repeats={args.repeats}", flush=True)
    print("threads,cpu_min_range_for_block,n,best_ms,median_ms", flush=True)
    for n_threads, minimum in [(1, 512), (args.threads, 512), (args.threads, 16), (args.threads, 1)]:
        for n, best_ms, median_ms in bench_one(n_threads, minimum, args.repeats):
            print(f"{n_threads},{minimum},{n},{best_ms:.3f},{median_ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
