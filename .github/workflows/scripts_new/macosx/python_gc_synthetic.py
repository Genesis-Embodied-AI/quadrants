"""Fixed-graph Mac GC control without importing Quadrants or any GPU runtime.

Three independent workers retain identical-sized graphs and collect them in allocation/random/allocation order.
Only owned synthetic lists are retracked through CPython's public GC API, with automatic GC disabled and the GIL
held by ctypes.PyDLL. Object contents and ownership remain unchanged. This is a diagnostic, not a runtime patch.
"""

import argparse
import ctypes
import gc
import json
import random
import resource
import subprocess
import sys
import time
from pathlib import Path


def worker(folder, index):
    assert "quadrants" not in sys.modules
    folder.mkdir(parents=True, exist_ok=True)
    enabled = gc.isenabled()
    gc.disable()
    nodes = [[bytearray(1024)] for _ in range(400000)]
    # Two edges per node, fixed throughout the experiment; payload keeps the process footprint substantial.
    for i, node in enumerate(nodes):
        node.extend((nodes[(i - 1) % len(nodes)], nodes[(i * 7919 + 17) % len(nodes)]))
    original = list(nodes)
    shuffled = list(nodes)
    random.Random(0).shuffle(shuffled)
    untrack = ctypes.pythonapi.PyObject_GC_UnTrack
    track = ctypes.pythonapi.PyObject_GC_Track
    for function in (untrack, track):
        function.argtypes = [ctypes.py_object]
        function.restype = None
    path = folder / f"gc-synthetic-{index}.jsonl"
    with path.open("w", buffering=1) as stream:
        def write(record):
            stream.write(json.dumps(dict(timestamp=time.time(), worker=index, **record)) + "\n")

        write(dict(kind="start", nodes=len(nodes), payload_bytes=1024, automatic_gc=False,
                   quadrants_imported="quadrants" in sys.modules))
        for stage, order in (("allocation_before", original), ("random", shuffled), ("allocation_after", original)):
            for node in order:
                untrack(node)
                track(node)
            gc.collect()
            actual = [obj for obj in gc.get_objects(2) if type(obj) is list and len(obj) == 3
                      and type(obj[0]) is bytearray and len(obj[0]) == 1024]
            assert len(actual) == len(order) and all(a is b for a, b in zip(actual, order))
            del actual
            write(dict(kind="stage", stage=stage, tracked=len(gc.get_objects(2)),
                       order_verified=True, maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            for repetition in range(100):
                before = resource.getrusage(resource.RUSAGE_SELF)
                wall, cpu, thread = time.perf_counter(), time.process_time(), time.thread_time()
                collected = gc.collect()
                wall, cpu, thread = time.perf_counter() - wall, time.process_time() - cpu, time.thread_time() - thread
                after = resource.getrusage(resource.RUSAGE_SELF)
                write(dict(kind="gc", phase="synthetic", generation=2, stage=stage, repetition=repetition,
                           wall_seconds=wall, cpu_seconds=cpu, thread_cpu_seconds=thread, collected=collected,
                           resource_delta={name: getattr(after, name) - getattr(before, name)
                                           for name in ("ru_utime", "ru_stime", "ru_minflt", "ru_majflt",
                                                        "ru_nvcsw", "ru_nivcsw", "ru_inblock", "ru_oublock")}))
                if repetition % 25 == 0:
                    print(index, stage, repetition, "GC", wall, "CPU", thread, flush=True)
            for i in range(0, len(nodes), 7919):
                assert nodes[i][1] is nodes[(i - 1) % len(nodes)]
                assert nodes[i][2] is nodes[(i * 7919 + 17) % len(nodes)]
                assert len(nodes[i][0]) == 1024
        assert "quadrants" not in sys.modules
        write(dict(kind="end", quadrants_imported=False))
    if enabled:
        gc.enable()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("This control is restricted to the pinned Mac Python 3.10 runner")
    if args.worker is not None:
        worker(args.output, args.worker)
        return
    children = [subprocess.Popen([sys.executable, __file__, str(args.output), "--worker", str(i)]) for i in range(3)]
    codes = [child.wait() for child in children]
    if any(codes):
        raise SystemExit(f"Synthetic worker failed: {codes}")


if __name__ == "__main__":
    main()
