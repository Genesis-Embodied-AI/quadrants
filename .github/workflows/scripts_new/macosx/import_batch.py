"""Compare one fixed test batch with small versus full-suite collection in fresh workers.

The full condition retains all collected items, including deselected ones, for the session. Optional object censuses
are separate diagnostic runs: walking the heap can perturb memory access and is excluded from primary timing runs.
"""

import gc
import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

FILES = (
    "test_scope_errors.py",
    "test_serial_execution.py",
    "test_shared_array.py",
    "test_shared_array_codegen_bug.py",
    "test_simt.py",
    "test_smoke.py",
    "test_snode.py",
    "test_snode_layout_inspection.py",
    "test_sort.py",
    "test_sparse_activate.py",
    "test_sparse_basics.py",
    "test_sparse_cg.py",
)
_selected = set()
_retained = ()
_seen_items = 0
_finished = 0
_stream = None


def run_bounded(invocation, root, env, output, label, timeout=1800):
    """Keep a stalled arm from preventing artifact upload; a timed-out arm is explicitly invalid/censored."""
    with (output / f"{label}.log").open("w") as log:
        proc = subprocess.Popen(
            invocation,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def terminate():
            if proc.poll() is None:
                (output / f"{label}-timeout.json").write_text(json.dumps(dict(timeout_seconds=timeout)) + "\n")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        timer = threading.Timer(timeout, terminate)
        timer.daemon = True
        timer.start()
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return proc.wait()
        finally:
            timer.cancel()
            timer.join()


def prepare(root, output, label, env, census):
    selection = output / "import-batch-selection.json"
    files = [root / "tests/python" / name for name in FILES]
    if not all(path.is_file() for path in files):
        raise RuntimeError("Fixed SIMT batch references missing test files")
    selection.write_text(json.dumps([str(path.resolve()) for path in files], indent=2) + "\n")
    env.update(
        QD_BATCH_SELECTION=str(selection),
        QD_IMPORT_SCOPE=label,
        QD_OBJECT_CENSUS="1" if census else "0",
        QD_IMPORT_TEST_ROOT=str(root / "tests/python"),
    )


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
    _write(
        dict(
            kind="census",
            stage=stage,
            generation_counts=counts,
            total_tracked=sum(counts),
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


def verify(output):
    selection = json.loads((output / "import-batch-selection.json").read_text())
    modules = {"python." + Path(path).stem for path in selection}
    extra = {}

    def cases(label):
        result = Counter()
        extra[label] = []
        for case in ET.parse(output / f"{label}.xml").iter("testcase"):
            classname, name = case.get("classname", ""), case.get("name", "")
            outcome = next((tag for tag in ("error", "failure", "skipped") if case.find(tag) is not None), "passed")
            module = classname or name
            if not any(module == wanted or module.startswith(wanted + ".") for wanted in modules):
                # Full collection also reports module-level skips outside the executed batch. Preserve them explicitly.
                if classname or outcome != "skipped":
                    raise RuntimeError(f"Unexpected case outside selected batch: {classname}::{name} ({outcome})")
                extra[label].append(dict(name=name, outcome=outcome))
                continue
            result[(classname, name, outcome)] += 1
        return result

    small, full = cases("small"), cases("all")
    differences = [
        dict(case=key, small=small[key], all=full[key])
        for key in sorted(small.keys() | full.keys())
        if small[key] != full[key]
    ]
    summary = dict(
        small_cases=sum(small.values()),
        all_cases=sum(full.values()),
        differences=differences,
        outside_batch_collection_skips=extra,
    )
    (output / "import-coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not small or differences:
        raise RuntimeError("Import experiment case outcomes differ")
