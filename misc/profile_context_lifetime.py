"""Check varying field layouts across resets and record current host RSS."""

import argparse
import gc
import json
import time
from pathlib import Path

import quadrants as qd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for cycle in range(args.cycles):
        started = time.perf_counter()
        qd.init(
            arch=qd.amdgpu,
            offline_cache=True,
            offline_cache_file_path=str(args.output / "kernels"),
            fast_math=bool(cycle % 2),
            device_memory_GB=0.3,
        )
        initialized = time.perf_counter()
        size = (cycle % 7 + 1) * 9
        values = qd.Vector.field(2 + cycle % 2, qd.f32, shape=size)

        @qd.kernel
        def fill():
            for i in values:
                for j in qd.static(range(values.n)):
                    values[i][j] = i + j

        fill()
        assert values[size - 1][values.n - 1] == size + values.n - 2
        qd.reset()
        del values, fill
        gc.collect()
        rss = next(
            line
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith("VmRSS:")
        )
        print(
            json.dumps(
                {
                    "cycle": cycle,
                    "init": initialized - started,
                    "rss_kib": int(rss.split()[1]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
