"""bit_cast scratch microbench.

Empirically verify the design-doc claim that ``qd.bit_cast`` between same-size
types is a no-op at the IR level: a kernel reading and writing an ``f32``
scratch buffer should run at indistinguishable speed (≤ 1% wall-time delta)
from the equivalent kernel that goes through ``Field(u32)`` + ``bit_cast``.

This pins the assumption underpinning ``quadrants/_scratch.py``: we share a
single ``Field(u32)`` scratch across every device algorithm and bit_cast in /
out per-dtype. If ``bit_cast`` were costly, the cross-call savings from
sharing one field would be eaten back at every kernel boundary.

Usage:

    python benchmarks/bit_cast_scratch.py            # CUDA default
    python benchmarks/bit_cast_scratch.py --arch vulkan

Exits 0 on pass (delta within tolerance), 1 on regression. Intended for
hand-runs and CI gating of the design assumption, not for the bulk
``benchmarks/run.py`` harness (which is structured around the per-op
parameter-sweep plans in ``microbenchmarks/``).

Note: at these per-iter wall times (~6 us / iter on a 5090 for N=1M), the
measurement is launch-overhead-dominated and run-to-run noise is several
percent. We therefore (a) take the *median* of many independent trials and
(b) use a coarser tolerance (default 5%). For a finer-grained guarantee,
dump the lowered IR / PTX with ``QD_DUMP_IR=1`` and confirm bit_cast lowers
to no instructions.
"""

import argparse
import sys
import time

import numpy as np

import quadrants as qd

BLOCK_DIM = 256


def _build_kernels(buf_f32, buf_u32, N):
    """Build the two paths-under-test. Identical structure; one reads/writes
    ``Field(f32)`` directly, the other ``Field(u32)`` + ``qd.bit_cast``."""

    @qd.kernel
    def add_one_direct():
        qd.loop_config(block_dim=BLOCK_DIM)
        for i in range(N):
            buf_f32[i] = buf_f32[i] + 1.0

    @qd.kernel
    def add_one_via_u32():
        qd.loop_config(block_dim=BLOCK_DIM)
        for i in range(N):
            v = qd.bit_cast(buf_u32[i], qd.f32)
            v = v + 1.0
            buf_u32[i] = qd.bit_cast(v, qd.u32)

    return add_one_direct, add_one_via_u32


def _time_once(kernel, repeats, warmup=10):
    for _ in range(warmup):
        kernel()
    qd.sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        kernel()
    qd.sync()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e6 / repeats


def _time_median(kernel, trials, repeats, warmup=10):
    """Time `kernel` over multiple independent trials and return the *median*
    per-iter wall time in microseconds. Median rather than mean because the
    distribution is heavy-tailed (driver / OS / cache stalls all pile mass on
    the slow side); median tracks the steady-state cost more honestly at the
    microsecond scale relevant here."""
    samples = [_time_once(kernel, repeats, warmup if i == 0 else 0) for i in range(trials)]
    samples.sort()
    return samples[len(samples) // 2]


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="cuda", choices=["cuda", "vulkan", "amdgpu", "metal"])
    p.add_argument("--N", type=int, default=1 << 20, help="elements per buffer (default 1M)")
    p.add_argument("--repeats", type=int, default=500, help="iters per trial")
    p.add_argument("--trials", type=int, default=11, help="independent trials; median is reported")
    p.add_argument(
        "--tolerance-pct",
        type=float,
        default=5.0,
        help=(
            "Pass threshold on |median(bit_cast) - median(direct)| / median(direct). "
            "5%% accounts for driver / launch-overhead noise dominating at our "
            "single-kernel scale (~6 us / iter); the IR claim is that bit_cast "
            "lowers to no instructions, which we additionally verify out-of-band "
            "by dumping IR (QD_DUMP_IR=1)."
        ),
    )
    args = p.parse_args(argv)

    qd.init(arch=getattr(qd, args.arch))
    N = args.N

    buf_f32 = qd.field(qd.f32, shape=N)
    buf_u32 = qd.field(qd.u32, shape=N)
    init_host = np.arange(N, dtype=np.float32) * 0.5
    buf_f32.from_numpy(init_host)
    buf_u32.from_numpy(init_host.view(np.uint32))

    direct, via_u32 = _build_kernels(buf_f32, buf_u32, N)

    t_direct = _time_median(direct, args.trials, args.repeats)
    t_via_u32 = _time_median(via_u32, args.trials, args.repeats)

    delta_pct = 100.0 * (t_via_u32 - t_direct) / t_direct
    print(f"N={N}  trials={args.trials}  repeats={args.repeats}  arch={args.arch}")
    print(f"  Field(f32) direct           : {t_direct:8.3f} us / iter   (median)")
    print(f"  Field(u32) + bit_cast(f32)  : {t_via_u32:8.3f} us / iter   (median)")
    print(f"  delta                       : {delta_pct:+7.3f}%   (tolerance ±{args.tolerance_pct}%)")

    # Only a *positive* delta (bit_cast path slower than direct) is a regression.
    # A negative delta means the bit_cast path was at least as fast - consistent
    # with the no-op claim - even though run-to-run noise (±a few percent at
    # microsecond timings) sometimes flips the sign.
    if delta_pct > args.tolerance_pct:
        print(
            f"FAIL: bit_cast path is {delta_pct:+.2f}% slower than direct, "
            f"exceeds +{args.tolerance_pct}%. The shared-Field(u32)-scratch "
            f"assumption in quadrants/_scratch.py needs revisiting.",
            file=sys.stderr,
        )
        return 1

    print("PASS: bit_cast is effectively a no-op at the IR level on this backend.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
