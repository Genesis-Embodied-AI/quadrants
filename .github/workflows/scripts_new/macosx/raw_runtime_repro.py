"""Direct MoltenVK reset reproducer: worker processes never import Quadrants.

Requires Python 3.10 on macOS, raw_vulkan_device.py/raw_vulkan_runtime.py, the pinned MoltenVK dylib, and the
_gc_walk extension built by gc_walk_probe.py. CPU stages only collect the retained graph; they create no runtimes.
The attribution controller supplies the independent memory process and the 30-minute execution bound.
"""

import argparse
import gc
import json
import multiprocessing
import os
import resource
import sys
import time
from multiprocessing.connection import wait
from pathlib import Path

from raw_vulkan_device import RawVulkanDevice


def worker(folder, index, barrier, library, work, count):
    import _gc_walk

    assert "quadrants" not in sys.modules
    nodes = [[bytearray(256)] for _ in range(300000)]
    for i, node in enumerate(nodes):
        node.extend((nodes[(i - 1) % len(nodes)], nodes[(i * 7919 + 17) % len(nodes)]))
    identities = tuple(map(id, nodes))
    stage, cycle = "baseline", -1
    active = None
    raw = None
    with (folder / f"cycles-{index}.jsonl").open("w", buffering=1) as stream:

        def write(**record):
            stream.write(
                json.dumps(dict(timestamp=time.time(), worker=index, stage=stage, cycle=cycle, **record)) + "\n"
            )

        def callback(event, info):
            nonlocal active
            if info["generation"] != 2:
                return
            if event == "start":
                active = time.perf_counter(), time.thread_time(), resource.getrusage(resource.RUSAGE_SELF)
            elif active is not None:
                wall, cpu, before = active
                wall, cpu = time.perf_counter() - wall, time.thread_time() - cpu
                after = resource.getrusage(resource.RUSAGE_SELF)
                active = None
                write(
                    kind="gc",
                    wall_seconds=wall,
                    thread_cpu_seconds=cpu,
                    **info,
                    resource_delta={
                        key: getattr(after, key) - getattr(before, key)
                        for key in ("ru_minflt", "ru_majflt", "ru_utime", "ru_stime", "ru_nvcsw", "ru_nivcsw")
                    },
                )

        def census():
            assert "quadrants" not in sys.modules
            assert len(nodes) == len(identities) and all(
                id(node) == expected for node, expected in zip(nodes, identities)
            )
            for i in range(0, len(nodes), 7919):
                assert nodes[i][1] is nodes[(i - 1) % len(nodes)]
                assert nodes[i][2] is nodes[(i * 7919 + 17) % len(nodes)]
                assert len(nodes[i][0]) == 256
            write(
                kind="census",
                tracked=len(gc.get_objects()),
                kernels=0,
                quadrants_imported=False,
                graph_identity_verified=True,
                graph_edges_sampled=True,
                memory=_gc_walk.memory(),
                maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )

        stages = dict(cpu_before=300, vulkan=count, cpu_after=300)
        write(
            kind="start",
            pid=os.getpid(),
            nodes=len(nodes),
            payload_bytes=256,
            stage_cycles=stages,
            target_backend="vulkan",
            raw_device_cycles=True,
            quadrants_imported=False,
            cpu_stage_work="two full collections only",
            automatic_gc=gc.isenabled(),
            unique_compile_constants=False,
            moltenvk_environment={
                key: os.environ.get(key) for key in ("MVK_CONFIG_USE_MTLHEAP", "MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS")
            },
        )
        gc.callbacks.append(callback)
        try:
            for _ in range(10):
                gc.collect()
            census()
            for stage, stage_count in stages.items():
                cycle = -1
                barrier.wait(timeout=1800)
                if stage == "vulkan":
                    raw = RawVulkanDevice(library, work)
                    write(kind="raw_device_setup", **raw.metadata)
                write(kind="stage_begin", actual_backend="raw-vulkan-device" if stage == "vulkan" else "python-only")
                for cycle in range(stage_count):
                    started = time.perf_counter()
                    if stage == "vulkan":
                        raw.cycle()
                        if cycle == 0:
                            write(kind="raw_device_buffers", allocations=raw.buffer_records)
                    elapsed = time.perf_counter() - started
                    gc.collect()
                    gc.collect()
                    write(kind="cycle", runtime_wall_seconds=elapsed)
                    if cycle % 25 == 0 or cycle == stage_count - 1:
                        census()
                        print(index, stage, cycle, "runtime", elapsed, flush=True)
                write(kind="stage_end", cycles_completed=stage_count, stopped_early=False)
            raw.close()
            write(kind="end", quadrants_imported=False, automatic_gc=gc.isenabled())
        finally:
            gc.callbacks.remove(callback)
            if raw is not None:
                raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    default_library = Path(os.environ.get("QD_LIB_DIR", ".")) / "libMoltenVK.dylib"
    parser.add_argument("--library", type=Path, default=default_library)
    parser.add_argument("--work", default=os.environ.get("QD_RAW_DEVICE_WORK", "private-bda-runtime"))
    parser.add_argument("--cycles", type=int, default=int(os.environ.get("QD_CYCLE_VULKAN_COUNT", "1500")))
    args = parser.parse_args()
    assert sys.platform == "darwin" and sys.version_info[:2] == (3, 10)
    assert args.library.is_file() and 1 <= args.cycles <= 2000
    assert args.work in ("private-bda-runtime", "private-bda-runtime-bare", "private-bda-runtime-smallroot")
    assert "quadrants" not in sys.modules
    args.output.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    children = [
        context.Process(target=worker, args=(args.output, i, barrier, args.library, args.work, args.cycles))
        for i in range(3)
    ]
    for child in children:
        child.start()
    pending = list(children)
    while pending:
        ready = wait([child.sentinel for child in pending])
        for child in list(pending):
            if child.sentinel in ready:
                child.join()
                pending.remove(child)
                if child.exitcode:
                    for other in pending:
                        other.terminate()
                    for other in pending:
                        other.join()
                    raise SystemExit(f"Raw runtime worker failed: {child.exitcode}")


if __name__ == "__main__":
    main()
