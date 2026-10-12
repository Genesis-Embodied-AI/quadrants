"""Compare one fixed test batch with small versus full-suite collection in fresh workers.

The full condition retains all collected items, including deselected ones, for the session. Optional object censuses
are separate diagnostic runs: walking the heap can perturb memory access and is excluded from primary timing runs.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import pytest

_selected = set()
_retained = ()
_seen_items = 0
_finished = 0
_stream = None


def pytest_configure(config):
    global _selected, _stream
    _selected = {Path(path).resolve() for path in json.loads(Path(os.environ["QD_BATCH_SELECTION"]).read_text())}
    if not _selected or any(not path.is_file() for path in _selected):
        raise RuntimeError("Invalid import experiment file selection")
    if os.environ.get("QD_IMPORT_SCOPE") not in ("small", "all"):
        raise RuntimeError("Expected small or all import scope")
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        folder = Path(os.environ["QD_GC_PROFILE_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        _stream = (folder / f"population-{worker}-{os.getpid()}.jsonl").open("w", buffering=1)


def pytest_ignore_collect(collection_path, config):
    if os.environ["QD_IMPORT_SCOPE"] == "small" and collection_path.is_file():
        if collection_path.suffix in (".py", ".ipynb"):
            return collection_path.resolve() not in _selected
    return None


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    global _retained, _seen_items
    _seen_items = len(items)
    if os.environ["QD_IMPORT_SCOPE"] == "all":
        _retained = tuple(items)
    selected = [item for item in items if item.path.resolve() in _selected]
    removed = [item for item in items if item.path.resolve() not in _selected]
    items[:] = selected
    if removed:
        config.hook.pytest_deselected(items=removed)


def _write(record):
    if _stream is not None:
        _stream.write(json.dumps(dict(timestamp=time.time(), finished_tests=_finished, **record)) + "\n")


def _census(stage):
    if _stream is None or os.environ.get("QD_OBJECT_CENSUS") != "1":
        return
    start, cpu = time.perf_counter(), time.process_time()
    enabled = gc.isenabled()
    gc.disable()
    try:
        counts = [len(gc.get_objects(generation)) for generation in range(3)]
    finally:
        if enabled:
            gc.enable()
    impl = sys.modules.get("quadrants.lang.impl")
    runtime_kernels = len(impl.get_runtime().kernels) if impl is not None else None
    _write(
        dict(
            kind="census",
            stage=stage,
            generation_counts=counts,
            total_tracked=sum(counts),
            runtime_kernels=runtime_kernels,
            wall_seconds=time.perf_counter() - start,
            cpu_seconds=time.process_time() - cpu,
            automatic_gc_restored=gc.isenabled() == enabled,
        )
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    root = Path(os.environ["QD_IMPORT_TEST_ROOT"]).resolve()
    modules = []
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).resolve().is_relative_to(root):
            modules.append(name)
    _write(
        dict(
            kind="collection",
            scope=os.environ["QD_IMPORT_SCOPE"],
            seen_items=_seen_items,
            retained_items=len(_retained),
            selected_items=len(session.items),
            modules=sorted(modules),
            nodeids=[item.nodeid for item in session.items],
            census=os.environ.get("QD_OBJECT_CENSUS") == "1",
        )
    )
    _census("after_collection")


def pytest_runtest_logfinish(nodeid, location):
    global _finished
    _finished += 1
    if _finished % 100 == 0:
        _census("after_tests")


def pytest_sessionfinish(session, exitstatus):
    _census("session_finish")
    _write(dict(kind="end", exitstatus=int(exitstatus)))
    if _stream is not None:
        _stream.close()
