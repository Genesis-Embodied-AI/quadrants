"""GPU-free integration checks for file selection and failure propagation through fresh xdist sessions."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from textwrap import dedent

import pytest

from tests import run_test_batches


@pytest.fixture(autouse=True)
def isolate_nested_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nested sessions must not inherit the enclosing Vulkan batch selection or xdist worker identity.
    for key in ("QD_BATCH_SELECTION", "PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-q -p xdist.plugin -n 3")


def make_fake_tests_project(root: Path) -> None:
    tests_path = root / "tests" / "python"
    tests_path.mkdir(parents=True)
    (root / "tests" / "run_tests.py").write_text(
        dedent(
            """\
            import pytest, sys
            raise SystemExit(pytest.main(['tests/python', *sys.argv[1:]]))
            """
        )
    )
    (tests_path / "conftest.py").write_text(
        dedent(
            """\
            import pytest
            # Set a marker so tests can verify that conftest fixtures still run when tests are batched.
            @pytest.fixture(autouse=True)
            def fixture_ran(monkeypatch: pytest.MonkeyPatch) -> None:
                monkeypatch.setenv('BATCH_FIXTURE_RAN', 'yes')
            """
        )
    )
    for index in range(13):
        folder = tests_path if index < 12 else tests_path / "nested"
        folder.mkdir(exist_ok=True)
        name = f"test_case_{index:02d}.py" if index < 12 else "nested_test.py"
        (folder / name).write_text(
            dedent(
                """\
                import json, os, pytest
                from pathlib import Path
                if 'QD_BATCH_SELECTION' in os.environ:
                    assert str(Path(__file__).resolve()) in json.loads(Path(os.environ['QD_BATCH_SELECTION']).read_text())
                @pytest.mark.parametrize('value', [0, 1])
                def test_case(value: int) -> None:
                    assert os.environ.get('BATCH_FIXTURE_RAN') == 'yes'
                @pytest.mark.skip(reason='expected')
                def test_skip() -> None:
                    pass
                """
            )
        )


def cases(path: Path) -> Counter[tuple[str | None, str | None, tuple[str, ...]]]:
    return Counter(
        (case.get("classname"), case.get("name"), tuple(child.tag for child in case))
        for case in ET.parse(path).iter("testcase")
    )


def test_complete_case_outcomes_and_fresh_selection(tmp_path: Path) -> None:
    make_fake_tests_project(tmp_path)
    baseline = tmp_path / "full.xml"
    subprocess.run(
        [sys.executable, "tests/run_tests.py", f"--junitxml={baseline}", f"--basetemp={tmp_path / 'full-tmp'}"],
        cwd=tmp_path,
        check=True,
    )
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, []) == 0
    assert cases(output / "results.xml") == cases(baseline)
    assert sum(cases(baseline).values()) == 39
    manifest_contents = json.loads((output / "manifest.json").read_text())
    assert [len(group) for group in manifest_contents] == [12, 1]


def test_failure_is_not_hidden_by_later_success(tmp_path: Path) -> None:
    make_fake_tests_project(tmp_path)
    (tmp_path / "tests" / "python" / "nested" / "nested_test.py").write_text(
        "def test_failure() -> None:\n    assert False\n"
    )
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, []) == 1
    records = json.loads((output / "results.json").read_text())
    assert [(record["batch"], record["returncode"]) for record in records] == [(1, 0)]
    assert cases(output / "results.xml") == cases(output / "batch-01.xml")
    assert len(list(ET.parse(output / "batch-00.xml").iter("failure"))) == 1
    later_report = ET.parse(output / "batch-01.xml")
    assert len(list(later_report.iter("testcase"))) == 3
    assert not list(later_report.iter("failure"))


@pytest.mark.parametrize("ignore_option", ["--ignore", "--ignore-glob"])
def test_selected_files_respect_pytest_ignore_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_option: str
) -> None:
    make_fake_tests_project(tmp_path)
    ignored = tmp_path / "tests" / "python" / "test_case_00.py"
    ignored.write_text("raise AssertionError('Ignored test module must not be imported')\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", f"-q -p xdist.plugin -n 3 {ignore_option}=tests/python/test_case_00.py")
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, []) == 0
    report = ET.parse(output / "results.xml")
    assert len(list(report.iter("testcase"))) == 36
    assert all("test_case_00" not in case.get("classname", "") for case in report.iter("testcase"))
