"""Bounded read-only GC traversal diagnostic; only for the isolated Mac attribution workflow.

The snapshot holds objects while native code reads their headers under the GIL. Address sorting changes only a
temporary pointer array, never the interpreter's links. Scans warm memory and are not primary benchmark timings.
"""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time

_count = 0


def capture(stage, triggering_gc=None):
    global _count
    if _count >= 4:
        return
    import _gc_walk

    enabled = gc.isenabled()
    gc.disable()
    try:
        snapshot_start = time.perf_counter()
        objects = gc.get_objects(2)
        snapshot_wall = time.perf_counter() - snapshot_start
        result = _gc_walk.measure(objects)
        del objects
    finally:
        if enabled:
            gc.enable()
    folder = Path(os.environ["QD_GC_PROFILE_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"walk-{os.environ.get('PYTEST_XDIST_WORKER', 'smoke')}-{os.getpid()}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(dict(stage=stage, timestamp=time.time(), triggering_gc=triggering_gc,
                                     snapshot_wall=snapshot_wall, scans=["linked", "array", "sorted_array"],
                                     **result)) + "\n")
    _count += 1


def build():
    folder = Path(__file__).resolve().parent
    destination = folder / ("_gc_walk" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(["clang", "-O2", "-bundle", "-undefined", "dynamic_lookup", "-I",
                    sysconfig.get_path("include"), str(folder / "gc_walk_probe.c"), "-o", str(destination)], check=True)
    print("Built", destination, flush=True)
    gc.collect()
    capture("smoke")
    print("Read-only traversal smoke passed", flush=True)


if __name__ == "__main__":
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("Only the pinned Mac Python 3.10 diagnostic is supported")
    build()
