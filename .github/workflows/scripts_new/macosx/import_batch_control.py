"""Standard-library controller helpers for the import experiment; pytest exists only inside test environments."""

import json
import os
import signal
import subprocess
import threading
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

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
