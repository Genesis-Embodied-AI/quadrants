"""Run Mac Vulkan test files in small groups with fresh pytest controllers and workers.

Long-lived Mac Vulkan workers can spend minutes in individual full garbage collections. Keeping each process's
collection small avoids this in the measured runs while retaining the normal fixtures and interpreter shutdown.
This module also acts as a pytest plugin to select files before importing their test modules.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import Config

_SELECTED_TEST_PATHS: set[Path] | None = None


def pytest_configure(config: Config) -> None:
    # pytest_configure is a pytest startup hook. Pytest calls it automatically after parsing options and before
    # collecting tests.
    #
    # In this plugin, it:
    # 1. Reads QD_BATCH_SELECTION to locate the selection JSON file.
    # 2. Loads its file paths into the global _SELECTED_TEST_PATHS set.
    # 3. Raises an error if the selection is empty or references missing files.
    #
    # Later, pytest_ignore_collect() uses _SELECTED_TEST_PATHS to decide which test files to skip.
    #
    # config is pytest's configuration object; this implementation does not use it. Each xdist worker runs this
    # setup for itself.
    global _SELECTED_TEST_PATHS
    _SELECTED_TEST_PATHS = {Path(path) for path in json.loads(Path(os.environ["QD_BATCH_SELECTION"]).read_text())}
    if not _SELECTED_TEST_PATHS or any(not path.is_file() for path in _SELECTED_TEST_PATHS):
        raise RuntimeError("Batch selection is empty or references missing files")


def pytest_ignore_collect(collection_path: Path, config: Config) -> bool | None:
    if collection_path.is_file() and collection_path.suffix in (".py", ".ipynb"):
        return collection_path.resolve() not in _SELECTED_TEST_PATHS
    return None


def run_batches(root: Path, output: Path, runner_args: list[str], batch_size: int = 12) -> int:
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    test_root = root / "tests" / "python"
    files = sorted({*test_root.rglob("test_*.py"), *test_root.rglob("*_test.py"), *test_root.rglob("*.ipynb")})
    if not files or batch_size < 1:
        raise ValueError("A nonempty test manifest and positive batch size are required")
    file_groups = [files[start : start + batch_size] for start in range(0, len(files), batch_size)]
    manifest_contents = [[str(path.relative_to(root)) for path in group] for group in file_groups]
    (output / "manifest.json").write_text(json.dumps(manifest_contents, indent=2) + "\n")
    results = []
    suites_xml_nodes = ET.Element("testsuites")
    failed = False
    for index, group in enumerate(file_groups):
        batch_label = f"batch-{index:02d}"
        selection_path = output / f"{batch_label}-selection.json"
        selection_path.write_text(json.dumps([str(path) for path in group]) + "\n")
        batch_test_output_path = output / f"{batch_label}.xml"
        batch_test_output_path.unlink(missing_ok=True)
        env = dict(os.environ, QD_BATCH_SELECTION=str(selection_path))
        # Add this script's parent directory to PYTHONPATH so pytest can find it as a plugin.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent) + os.pathsep + env.get("PYTHONPATH", "")
        # Each subprocess needs its own temporary directory and test report. Other pytest settings are preserved.
        # Insert new options into existing PYTEST_ADDOPTS.
        options = shlex.split(env.get("PYTEST_ADDOPTS", ""))
        options += [
            "-p",
            "vulkan_batches",
            f"--basetemp={output / (batch_label + '-tmp')}",
            f"--junitxml={batch_test_output_path}",
        ]
        env["PYTEST_ADDOPTS"] = shlex.join(options)
        print(f"Batch {index + 1}/{len(file_groups)}: {manifest_contents[index]}", flush=True)
        start = time.monotonic()
        result = subprocess.run([sys.executable, "tests/run_tests.py", *runner_args], cwd=root, env=env)
        batch_failed = result.returncode != 0
        if not batch_failed:
            record = dict(batch=index, seconds=time.monotonic() - start, returncode=result.returncode)
            results.append(record)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print("Batch result:", json.dumps(record), flush=True)
            batch_failed |= not batch_test_output_path.exists()
        failed |= batch_failed
        if not batch_failed:
            report = ET.parse(batch_test_output_path).getroot()
            if report.tag == "testsuite":
                suites_xml_nodes.append(report)
            else:
                suites_xml_nodes.extend(report)
    ET.ElementTree(suites_xml_nodes).write(output / "results.xml", encoding="utf-8", xml_declaration=True)
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    return run_batches(Path.cwd(), args.output, runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
