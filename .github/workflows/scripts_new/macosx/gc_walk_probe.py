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
_thresholds = (3, 10, 30)
_next_threshold = 0


def capture(stage, triggering_gc=None):
    global _count, _next_threshold
    if _count >= 4:
        return
    if triggering_gc is not None:
        if _next_threshold >= len(_thresholds) or triggering_gc["wall_seconds"] < _thresholds[_next_threshold]:
            return
        _next_threshold += 1
    _count += 1
    import _gc_walk

    enabled = gc.isenabled()
    gc.disable()
    try:
        if os.environ.get("QD_GC_MEMORY") == "1":
            cold, objects, snapshot_wall = _gc_walk.snapshot()
        else:
            cold = _gc_walk.cold()
            snapshot_start = time.perf_counter()
            objects = gc.get_objects(2)
            snapshot_wall = time.perf_counter() - snapshot_start
        result = _gc_walk.measure(objects)
        if cold["objects"] != result["objects"]:
            raise RuntimeError(f"Direct generation-2 count {cold['objects']} differs from snapshot {result['objects']}")
        memory_started_at = time.time()
        memory = _gc_walk.memory() if os.environ.get("QD_GC_MEMORY") == "1" else None
        references = _gc_walk.references(objects) if memory is not None else None
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
                                     references=references, memory_started_at=memory_started_at, memory=memory, cold=cold, cold_scans=["arithmetic_before", "linked", "arithmetic_after"],
                                     **result)) + "\n")


def build():
    folder = Path(__file__).resolve().parent
    destination = folder / ("_gc_walk" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(["clang", "-O2", "-bundle", "-undefined", "dynamic_lookup", "-I",
                    sysconfig.get_path("include"), str(folder / "gc_walk_probe.c"), "-o", str(destination)], check=True)
    print("Built", destination, flush=True)
    gc.collect()
    capture("smoke")
    if os.environ.get("QD_GC_MEMORY") == "1":
        import _gc_walk

        memory = _gc_walk.memory()
        assert len(memory["resources"]) == 4
        chunks = memory["chunks"]
        assert len(chunks) == 64 and sum(c["visits"] for c in chunks) == memory["visits"]
        assert 1 <= sum(c["new_pages"] for c in chunks) <= memory["residency_before"]["pages"]
        assert all(0 <= c["new_pages"] <= c["visits"] for c in chunks)
        for pass_index in (0, 1):
            assert sum(c["thread_cpu_seconds"][pass_index] for c in chunks) <= memory["thread_cpu_seconds"][pass_index + 1]
        assert all(r["minor_faults"] >= 0 and r["major_faults"] >= 0 for r in memory["resources"])
        for name in ("residency_before", "residency_after"):
            assert memory[name]["status"] == 0 and memory[name]["pages"] > 0
            assert 0 <= memory[name]["resident"] <= memory[name]["pages"]
        print("Memory residency/resource smoke passed", memory["residency_after"], flush=True)
        enabled = gc.isenabled()
        gc.disable()
        try:
            public = gc.get_objects(2)
            _, pinned, _ = _gc_walk.snapshot()
            assert len(public) == len(pinned) and all(a is b for a, b in zip(public, pinned))
        finally:
            if enabled:
                gc.enable()
        print("Public/atomic snapshot identity check passed", flush=True)
    print("Read-only traversal smoke passed", flush=True)


if __name__ == "__main__":
    if sys.platform != "darwin" or sys.version_info[:2] != (3, 10):
        raise RuntimeError("Only the pinned Mac Python 3.10 diagnostic is supported")
    build()
