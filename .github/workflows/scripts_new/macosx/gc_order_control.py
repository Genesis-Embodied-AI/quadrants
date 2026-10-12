"""One-shot GC-order control on a retained real test heap, after a long collection.

Diagnostic only: keeps every sampled object alive and restores original tracking order before tests resume.
Runs outside the GC callback, because recursive collection inside a callback is not a valid measurement.
"""

import ctypes
import gc
import json
import os
import resource
import time
from pathlib import Path

import pytest

_start = None
_pending = None
_done = False


def observe(event, info):
    global _start, _pending
    if _done or info["generation"] != 2:
        return
    if event == "start":
        _start = time.perf_counter()
    elif _start is not None:
        elapsed = time.perf_counter() - _start
        if elapsed >= 10:
            _pending = elapsed
        _start = None


def capture():
    global _done
    _done = True
    enabled = gc.isenabled()
    gc.disable()
    untrack, track = ctypes.pythonapi.PyObject_GC_UnTrack, ctypes.pythonapi.PyObject_GC_Track
    for function in (untrack, track):
        function.argtypes = [ctypes.py_object]
        function.restype = None
    folder = Path(os.environ["QD_GC_PROFILE_DIR"])
    path = folder / f"order-{os.environ['PYTEST_XDIST_WORKER']}-{os.getpid()}.jsonl"
    original = None
    restored = False

    def retrack(objects):
        # Strong references in the snapshot protect every object for this whole experiment.
        for obj in objects:
            if not gc.is_tracked(obj):
                raise RuntimeError("A retained object's GC tracking status changed")
            untrack(obj)
            track(obj)

    try:
        with path.open("w", buffering=1) as stream:

            def write(record):
                stream.write(json.dumps(dict(timestamp=time.time(), **record)) + "\n")

            write(dict(kind="start", triggering_gc_wall=_pending))
            # Stabilize tuple/dict untracking before pinning the graph and changing traversal order.
            gc.collect()
            gc.collect()
            original = gc.get_objects(2)
            identities = {id(obj) for obj in original}
            ordered = sorted(original, key=id)
            write(dict(kind="snapshot", objects=len(original)))
            for stage, order in (
                ("original_before", original),
                ("address_sorted", ordered),
                ("original_after", original),
            ):
                retrack(order)
                gc.collect()
                actual = [obj for obj in gc.get_objects(2) if id(obj) in identities]
                if len(actual) != len(order) or not all(a is b for a, b in zip(actual, order)):
                    raise RuntimeError("GC tracking order did not match requested stage")
                del actual
                write(dict(kind="stage", stage=stage, objects=len(original), order_verified=True))
                for repetition in range(5):
                    before = resource.getrusage(resource.RUSAGE_SELF)
                    wall, cpu, thread = time.perf_counter(), time.process_time(), time.thread_time()
                    collected = gc.collect()
                    wall, cpu, thread = (
                        time.perf_counter() - wall,
                        time.process_time() - cpu,
                        time.thread_time() - thread,
                    )
                    after = resource.getrusage(resource.RUSAGE_SELF)
                    write(
                        dict(
                            kind="gc",
                            stage=stage,
                            repetition=repetition,
                            wall_seconds=wall,
                            cpu_seconds=cpu,
                            thread_cpu_seconds=thread,
                            collected=collected,
                            resource_delta={
                                name: getattr(after, name) - getattr(before, name)
                                for name in (
                                    "ru_utime",
                                    "ru_stime",
                                    "ru_minflt",
                                    "ru_majflt",
                                    "ru_nvcsw",
                                    "ru_nivcsw",
                                    "ru_inblock",
                                    "ru_oublock",
                                )
                            },
                        )
                    )
                    print("GC order control", stage, repetition, wall, thread, flush=True)
            actual = [obj for obj in gc.get_objects(2) if id(obj) in identities]
            assert len(actual) == len(original) and all(a is b for a, b in zip(actual, original))
            del actual
            restored = True
            write(dict(kind="end", restored=True, objects=len(original)))
    finally:
        if original is not None and not restored:
            # On an exceptional path, restore the relative order of every still-tracked sampled object.
            for obj in original:
                if gc.is_tracked(obj):
                    untrack(obj)
                    track(obj)
            gc.collect()
        if enabled:
            gc.enable()


def pytest_configure(config):
    if os.environ.get("PYTEST_XDIST_WORKER"):
        gc.callbacks.append(observe)


def pytest_unconfigure(config):
    if observe in gc.callbacks:
        gc.callbacks.remove(observe)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_teardown(item):
    yield
    if _pending is not None and not _done:
        capture()
