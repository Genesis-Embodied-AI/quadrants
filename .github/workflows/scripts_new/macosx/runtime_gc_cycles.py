"""Bounded minimal Mac runtime-cycle diagnostic with a retained synthetic graph and one kernel per worker.

Three workers run CPU/Vulkan/CPU stages on the same VM, retaining the same graph and kernel identities throughout.
This is a trigger search, not a full-suite benchmark or an order-independent backend comparison. No registry patch,
GC suppression, heap reordering, or process recycling is applied. Native memory controls are read-only.
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


def worker(folder, index, barrier):
    import _gc_walk

    import quadrants as qd
    from quadrants.lang import impl

    nodes = [[bytearray(256)] for _ in range(300000)]
    for i, node in enumerate(nodes):
        node.extend((nodes[(i - 1) % len(nodes)], nodes[(i * 7919 + 17) % len(nodes)]))
    identities = tuple(id(node) for node in nodes)

    @qd.kernel
    def retained(value: qd.i32) -> qd.i32:
        return value + 1

    kernel_identity = id(retained._primal)
    stage, cycle = "baseline", -1
    active = None
    path = folder / f"cycles-{index}.jsonl"
    with path.open("w", buffering=1) as stream:

        def write(record):
            stream.write(
                json.dumps(dict(timestamp=time.time(), worker=index, stage=stage, cycle=cycle, **record)) + "\n"
            )

        def callback(event, info):
            nonlocal active
            if info["generation"] != 2:
                return
            if event == "start":
                active = (time.perf_counter(), time.thread_time(), resource.getrusage(resource.RUSAGE_SELF))
            elif active is not None:
                wall, cpu, before = active
                wall, cpu = time.perf_counter() - wall, time.thread_time() - cpu
                after = resource.getrusage(resource.RUSAGE_SELF)
                active = None
                write(
                    dict(
                        kind="gc",
                        wall_seconds=wall,
                        thread_cpu_seconds=cpu,
                        **info,
                        resource_delta={
                            key: getattr(after, key) - getattr(before, key)
                            for key in ("ru_minflt", "ru_majflt", "ru_utime", "ru_stime", "ru_nvcsw", "ru_nivcsw")
                        },
                    )
                )

        def census():
            assert id(retained._primal) == kernel_identity
            assert len(nodes) == len(identities) and all(
                id(node) == expected for node, expected in zip(nodes, identities)
            )
            for i in range(0, len(nodes), 7919):
                assert nodes[i][1] is nodes[(i - 1) % len(nodes)]
                assert nodes[i][2] is nodes[(i * 7919 + 17) % len(nodes)]
                assert len(nodes[i][0]) == 256
            write(
                dict(
                    kind="census",
                    tracked=len(gc.get_objects()),
                    kernels=len(impl.get_runtime().kernels),
                    graph_identity_verified=True,
                    graph_edges_sampled=True,
                    memory=_gc_walk.memory(),
                    maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )
            )

        write(
            dict(
                kind="start",
                pid=os.getpid(),
                nodes=len(nodes),
                payload_bytes=256,
                cycles_per_stage=300,
                automatic_gc=gc.isenabled(),
                kernel_identity=kernel_identity,
            )
        )
        gc.callbacks.append(callback)
        try:
            for _ in range(10):
                gc.collect()
            census()
            for stage, backend in (("cpu_before", qd.cpu), ("vulkan", qd.vulkan), ("cpu_after", qd.cpu)):
                cycle = -1
                barrier.wait(timeout=1800)
                write(dict(kind="stage_begin", backend=str(backend)))
                for cycle in range(300):
                    start = time.perf_counter()
                    qd.init(arch=backend, enable_fallback=False, offline_cache=False, cpu_max_num_threads=1)
                    assert impl.current_cfg().arch == backend
                    assert retained(cycle) == cycle + 1
                    qd.reset()
                    runtime_wall = time.perf_counter() - start
                    gc.collect()
                    gc.collect()
                    write(dict(kind="cycle", runtime_wall_seconds=runtime_wall))
                    if cycle % 25 == 0 or cycle == 299:
                        census()
                        print(index, stage, cycle, "runtime", runtime_wall, flush=True)
                write(dict(kind="stage_end"))
            write(dict(kind="end", automatic_gc=gc.isenabled()))
        finally:
            gc.callbacks.remove(callback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert sys.platform == "darwin" and sys.version_info[:2] == (3, 10)
    args.output.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(3)
    children = [context.Process(target=worker, args=(args.output, index, barrier)) for index in range(3)]
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
                    raise SystemExit(f'Runtime-cycle worker failed: {child.exitcode}')
    codes = [child.exitcode for child in children]
    if any(codes):
        raise SystemExit(f"Runtime-cycle worker failed: {codes}")


if __name__ == "__main__":
    main()
