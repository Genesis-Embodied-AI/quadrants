"""Compare runtime initialization with caching enabled and disabled in fresh processes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.child:
        for _ in range(3):
            command = [sys.executable, __file__, "--child", "--output", str(args.output), "--cycles", str(args.cycles)]
            if args.trace:
                command.append("--trace")
            subprocess.run(command, check=True, timeout=180)
        return

    import quadrants as qd

    print(json.dumps({"quadrants": qd.__file__, "torch_imported": "torch" in sys.modules}), flush=True)
    for enabled in (False, True):
        for cycle in range(args.cycles):
            started = time.perf_counter()
            qd.init(
                arch=qd.amdgpu,
                enable_fallback=False,
                offline_cache=enabled,
                offline_cache_file_path=str(args.output / "kernel-cache"),
                device_memory_GB=0.3,
                log_level=qd.TRACE if args.trace else qd.ERROR,
            )
            initialized = time.perf_counter()
            qd.reset()
            print(
                json.dumps(
                    {
                        "cache": enabled,
                        "cycle": cycle,
                        "init": initialized - started,
                        "reset": time.perf_counter() - initialized,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
