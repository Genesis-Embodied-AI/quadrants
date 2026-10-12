"""GPU-free integration checks for file selection and failure propagation through fresh xdist sessions."""

import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest

_runner_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "scripts_new" / "macosx" / "vulkan_batches.py"
_spec = importlib.util.spec_from_file_location("vulkan_batches", _runner_path)
assert _spec is not None and _spec.loader is not None
vulkan_batches = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vulkan_batches)


@pytest.fixture(autouse=True)
def isolate_nested_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nested sessions must not inherit the enclosing Vulkan batch selection or xdist worker identity.
    for key in ("QD_BATCH_SELECTION", "PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-q -p xdist.plugin -n 3")


def make_project(root: Path) -> None:
    tests = root / "tests" / "python"
    tests.mkdir(parents=True)
    (root / "tests" / "run_tests.py").write_text(
        "import pytest, sys\nraise SystemExit(pytest.main(['tests/python', *sys.argv[1:]]))\n"
    )
    (tests / "conftest.py").write_text(
        "import pytest\n@pytest.fixture(autouse=True)\ndef fixture_ran(monkeypatch: pytest.MonkeyPatch) -> None:\n"
        "    monkeypatch.setenv('BATCH_FIXTURE_RAN', 'yes')\n"
    )
    for index in range(13):
        folder = tests if index < 12 else tests / "nested"
        folder.mkdir(exist_ok=True)
        name = f"test_case_{index:02d}.py" if index < 12 else "nested_test.py"
        (folder / name).write_text(
            "import json, os, pytest\nfrom pathlib import Path\n"
            "if 'QD_BATCH_SELECTION' in os.environ:\n"
            "    assert str(Path(__file__).resolve()) in "
            "json.loads(Path(os.environ['QD_BATCH_SELECTION']).read_text())\n"
            "@pytest.mark.parametrize('value', [0, 1])\ndef test_case(value: int) -> None:\n"
            "    assert os.environ.get('BATCH_FIXTURE_RAN') == 'yes'\n"
            "@pytest.mark.skip(reason='expected')\ndef test_skip() -> None:\n    pass\n"
        )


def cases(path: Path) -> Counter[tuple[str | None, str | None, tuple[str, ...]]]:
    return Counter(
        (case.get("classname"), case.get("name"), tuple(child.tag for child in case))
        for case in ET.parse(path).iter("testcase")
    )


def test_complete_case_outcomes_and_fresh_selection(tmp_path: Path) -> None:
    make_project(tmp_path)
    baseline = tmp_path / "full.xml"
    subprocess.run(
        [sys.executable, "tests/run_tests.py", f"--junitxml={baseline}", f"--basetemp={tmp_path / 'full-tmp'}"],
        cwd=tmp_path,
        check=True,
    )
    output = tmp_path / "batches"
    assert vulkan_batches.run_batches(tmp_path, output, []) == 0
    assert cases(output / "results.xml") == cases(baseline)
    assert sum(cases(baseline).values()) == 39
    manifest = json.loads((output / "manifest.json").read_text())
    assert [len(group) for group in manifest] == [12, 1]


def test_failure_is_not_hidden_by_later_success(tmp_path: Path) -> None:
    make_project(tmp_path)
    (tmp_path / "tests" / "python" / "nested" / "nested_test.py").write_text("def test_failure() -> None:\n    assert False\n")
    output = tmp_path / "batches"
    assert vulkan_batches.run_batches(tmp_path, output, []) == 1
    records = json.loads((output / "results.json").read_text())
    assert [record["returncode"] for record in records] == [1, 0]
    assert len(list(ET.parse(output / "results.xml").iter("failure"))) == 1
