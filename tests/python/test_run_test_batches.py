"""GPU-free integration checks for file selection and failure propagation through fresh xdist sessions."""

import json
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Literal

import pytest

from tests import run_test_batches


@pytest.fixture(autouse=True)
def isolate_nested_pytest(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nested sessions must not inherit the enclosing Vulkan batch selection or xdist worker identity.
    for key in ("QD_BATCH_SELECTION", "PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-q -p xdist.plugin -n 3")


@dataclass
class FakeTestsProject:
    runner_path: Path
    test_case_counts: dict[Path, int]

    @property
    def total_cases(self) -> int:
        return sum(self.test_case_counts.values())


def make_fake_tests_project(root: Path, num_test_files: int, passing_cases_per_file: int = 2) -> FakeTestsProject:
    """Create the requested number of test files and return their paths and case counts, including skipped cases."""
    tests_path = root / "tests" / "python"
    tests_path.mkdir(parents=True)
    runner_path = root / "tests" / "run_tests.py"
    runner_path.write_text(
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
    test_case_counts = {}
    for index in range(num_test_files):
        folder = tests_path if index < num_test_files - 1 else tests_path / "nested"
        folder.mkdir(exist_ok=True)
        name = f"test_case_{index:02d}.py" if index < num_test_files - 1 else "nested_test.py"
        test_path = folder / name
        test_path.write_text(
            dedent(
                f"""\
                import json, os, pytest
                from pathlib import Path

                # Verify at import time that this file belongs to the current batch; skip for the unbatched baseline.
                if 'QD_BATCH_SELECTION' in os.environ:
                    assert str(Path(__file__).resolve()) in json.loads(Path(os.environ['QD_BATCH_SELECTION']).read_text())

                # Verify that batching preserves parametrized tests without loss or duplication.
                @pytest.mark.parametrize('value', {list(range(passing_cases_per_file))!r})
                def test_case(value: int) -> None:
                    assert os.environ.get('BATCH_FIXTURE_RAN') == 'yes'

                # Include a skipped case to verify that batching preserves skipped tests and their reported outcomes.
                @pytest.mark.skip(reason='expected')
                def test_skip() -> None:
                    pass
                """
            )
        )
        test_case_counts[test_path] = passing_cases_per_file + 1  # Include the skipped case.
    return FakeTestsProject(runner_path=runner_path, test_case_counts=test_case_counts)


def parse_test_results(
    path: Path,
) -> dict[tuple[str | None, str | None], Literal["pass", "failure", "skipped", "error"]]:
    # Parse JUnit XML into a mapping from (classname, name) to outcome; assert and fail the test on duplicate identities.
    outcomes: dict[tuple[str | None, str | None], Literal["pass", "failure", "skipped", "error"]] = {}
    for case in ET.parse(path).iter("testcase"):
        key = (case.get("classname"), case.get("name"))
        assert key not in outcomes, f"Duplicate test case in {path}: {key}"
        for tag in ("error", "failure", "skipped"):
            if case.find(tag) is not None:
                outcomes[key] = tag
                break
        else:
            outcomes[key] = "pass"
    return outcomes


def test_complete_case_outcomes_and_fresh_selection(tmp_path: Path) -> None:
    num_test_files = 13
    batch_size = 12
    expected_batch_sizes = [12, 1]
    project = make_fake_tests_project(tmp_path, num_test_files=num_test_files)
    results_without_batching = tmp_path / "full.xml"
    # Run tests without batching.
    subprocess.run(
        [
            sys.executable,
            str(project.runner_path),
            f"--junitxml={results_without_batching}",
            f"--basetemp={tmp_path / 'full-tmp'}",
        ],
        cwd=tmp_path,
        check=True,
    )
    output_with_batching = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output_with_batching, [], batch_size=batch_size) == 0
    assert parse_test_results(output_with_batching / "results.xml") == parse_test_results(results_without_batching)
    manifest_contents = json.loads((output_with_batching / "manifest.json").read_text())
    assert [len(group) for group in manifest_contents] == expected_batch_sizes
    assert sorted(path for group in manifest_contents for path in group) == sorted(
        str(path.relative_to(tmp_path)) for path in project.test_case_counts
    )


def test_failure_is_not_hidden_by_later_success(tmp_path: Path) -> None:
    num_test_files = 13
    batch_size = 12
    project = make_fake_tests_project(tmp_path, num_test_files=num_test_files)
    test_paths = sorted(project.test_case_counts)
    test_paths[0].write_text("def test_failure() -> None:\n    assert False\n")
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, [], batch_size=batch_size) == 1
    records = json.loads((output / "results.json").read_text())
    assert [(record["batch"], record["returncode"]) for record in records] == [(1, 0)]
    assert parse_test_results(output / "results.xml") == parse_test_results(output / "batch-01.xml")
    assert len(list(ET.parse(output / "batch-00.xml").iter("failure"))) == 1
    later_report = ET.parse(output / "batch-01.xml")
    expected_later_cases = sum(project.test_case_counts[path] for path in test_paths[batch_size:])
    assert len(list(later_report.iter("testcase"))) == expected_later_cases
    assert not list(later_report.iter("failure"))


@pytest.mark.parametrize("ignore_option", ["--ignore", "--ignore-glob"])
def test_selected_files_respect_pytest_ignore_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_option: str
) -> None:
    num_test_files = 13
    batch_size = 12
    project = make_fake_tests_project(tmp_path, num_test_files=num_test_files)
    ignored = sorted(project.test_case_counts)[0]
    ignored.write_text("raise AssertionError('Ignored test module must not be imported')\n")
    monkeypatch.setenv(
        "PYTEST_ADDOPTS", shlex.join(["-q", "-p", "xdist.plugin", "-n", "3", f"{ignore_option}={ignored}"])
    )
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, [], batch_size=batch_size) == 0
    report = ET.parse(output / "results.xml")
    assert len(list(report.iter("testcase"))) == project.total_cases - project.test_case_counts[ignored]
