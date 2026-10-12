"""GPU-free validation of full import retention, batch-only execution and separate census diagnostics."""

import json
import os
import subprocess
import sys
from pathlib import Path

import import_batch
import pytest


@pytest.mark.parametrize("census", [False, True])
def test_import_scope_and_outcomes(tmp_path, census):
    root = tmp_path / "project"
    tests = root / "tests/python"
    tests.mkdir(parents=True)
    (root / "tests/pytest.ini").write_text("[pytest]\n")
    for name in import_batch.FILES:
        (tests / name).write_text(
            "import os, sys, pytest\n"
            "@pytest.mark.parametrize('value', [0, 1])\ndef test_case(value):\n"
            "    assert ('test_outside' in sys.modules) == (os.environ['QD_IMPORT_SCOPE'] == 'all')\n"
            "@pytest.mark.skip(reason='expected')\ndef test_skip():\n    pass\n"
        )
    (tests / "test_outside.py").write_text(
        "STATE = [object() for _ in range(100)]\ndef test_not_selected():\n    assert False\n"
    )
    (tests / "test_skip_outside.py").write_text(
        "import pytest\npytest.skip('outside module', allow_module_level=True)\n"
    )
    output = tmp_path / "output"
    output.mkdir()
    for label in ("small", "all"):
        env = dict(
            os.environ,
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            PYTEST_ADDOPTS="",
            PYTHONPATH=str(Path(import_batch.__file__).parent),
            QD_GC_PROFILE_DIR=str(output / (label + "-gc")),
        )
        import_batch.prepare(root, output, label, env, census)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/python",
                "-q",
                "-p",
                "xdist.plugin",
                "-n",
                "3",
                "-p",
                "import_batch",
                f"--junitxml={output / (label + '.xml')}",
                f"--basetemp={output / (label + '-tmp')}",
            ],
            cwd=root,
            env=env,
            check=True,
        )
        profiles = sorted((output / (label + "-gc")).glob("population-*.jsonl"))
        assert len(profiles) == 3
        for profile in profiles:
            records = [json.loads(line) for line in profile.read_text().splitlines()]
            collection = next(record for record in records if record["kind"] == "collection")
            assert collection["selected_items"] == 36
            assert collection["retained_items"] == (37 if label == "all" else 0)
            assert ("test_outside" in collection["modules"]) == (label == "all")
            scans = [record for record in records if record["kind"] == "census"]
            assert bool(scans) == census
            assert all(record["total_tracked"] > 0 and record["automatic_gc_restored"] for record in scans)
            assert records[-1]["kind"] == "end" and records[-1]["exitstatus"] == 0
    import_batch.verify(output)
    coverage = json.loads((output / "import-coverage.json").read_text())
    assert coverage["small_cases"] == coverage["all_cases"] == 36
    assert coverage["differences"] == []
    assert coverage["outside_batch_collection_skips"] == {
        "small": [],
        "all": [{"name": "python.test_skip_outside", "outcome": "skipped"}],
    }
