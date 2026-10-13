"""GPU-free integration checks for file selection and failure propagation through fresh xdist sessions."""

import ast
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
    for key in (
        "QD_BATCH_SELECTION",
        "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT",
        "PYTEST_XDIST_TESTRUNUID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-q -p xdist.plugin -n 3")


@dataclass
class FakeTestsProject:
    runner_path: Path
    test_case_count_by_test_file_path: dict[Path, int]

    @property
    def total_cases(self) -> int:
        return sum(self.test_case_count_by_test_file_path.values())


def make_fake_tests_project(root: Path, num_test_files: int, passing_cases_per_file: int = 2) -> FakeTestsProject:
    """Create a fake test project with the requested number of test files.

    Return a FakeTestsProject containing runner_path (the generated run_tests.py path) and
    test_case_count_by_test_file_path (a mapping from each generated test file path to its number of test cases,
    including the skipped case).
    Its total_cases property gives the total number of test cases across those files.
    """
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
    test_case_count_by_test_file_path = {}
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
        test_case_count_by_test_file_path[test_path] = passing_cases_per_file + 1  # Include the skipped case.
    return FakeTestsProject(
        runner_path=runner_path, test_case_count_by_test_file_path=test_case_count_by_test_file_path
    )


def parse_test_results(
    path: Path,
) -> dict[tuple[str | None, str | None], Literal["pass", "failure", "skipped", "error"]]:
    # Parse JUnit XML into a mapping from (test module/class, test case name) to outcome;
    # assert and fail the test on duplicate identities.
    # Each parametrized instance has a distinct test case name, such as test_case[0] and test_case[1].
    outcomes: dict[tuple[str | None, str | None], Literal["pass", "failure", "skipped", "error"]] = {}
    for case in ET.parse(path).iter("testcase"):
        classname = case.get("classname")
        test_name = case.get("name")
        key = (classname, test_name)
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
    fake_tests_project = make_fake_tests_project(tmp_path, num_test_files=num_test_files)
    results_without_batching = tmp_path / "full.xml"
    # Run tests without batching.
    subprocess.run(
        [
            sys.executable,
            str(fake_tests_project.runner_path),
            f"--junitxml={results_without_batching}",
            f"--basetemp={tmp_path / 'full-tmp'}",
        ],
        cwd=tmp_path,
        check=True,
    )
    output_with_batching = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output_with_batching, [], batch_size=batch_size) == 0
    assert parse_test_results(output_with_batching / "results.xml") == parse_test_results(results_without_batching)
    records = json.loads((output_with_batching / "results.json").read_text())
    assert all(record["test_runner_exit_code"] == 0 for record in records)
    combined_report = ET.parse(output_with_batching / "results.xml").getroot()
    baseline_suite = ET.parse(results_without_batching).getroot().find("testsuite")
    for attribute in ("tests", "failures", "errors", "skipped"):
        assert combined_report.get(attribute) == baseline_suite.get(attribute)
    manifest_contents = json.loads((output_with_batching / "manifest.json").read_text())
    assert [len(group) for group in manifest_contents] == expected_batch_sizes
    assert sorted(path for group in manifest_contents for path in group) == sorted(
        str(path.relative_to(tmp_path)) for path in fake_tests_project.test_case_count_by_test_file_path
    )


def test_failed_tests(tmp_path: Path) -> None:
    num_test_files = 13
    batch_size = 12
    fake_tests_project = make_fake_tests_project(tmp_path, num_test_files=num_test_files)
    test_paths = sorted(fake_tests_project.test_case_count_by_test_file_path)
    test_paths[0].write_text(
        dedent(
            """\
            def test_failure() -> None:
                assert False
            """
        )
    )
    results_without_batching = tmp_path / "full.xml"
    # Run tests without batching; a failing test should produce exit code 1.
    unbatched_process_result = subprocess.run(
        [
            sys.executable,
            str(fake_tests_project.runner_path),
            f"--junitxml={results_without_batching}",
            f"--basetemp={tmp_path / 'full-tmp'}",
        ],
        cwd=tmp_path,
        check=False,
    )
    assert unbatched_process_result.returncode == 1
    output_with_batching = tmp_path / "batches"
    assert (
        run_test_batches.run_batches(tmp_path, output_with_batching, [], batch_size=batch_size)
        == unbatched_process_result.returncode
    )
    assert parse_test_results(output_with_batching / "results.xml") == parse_test_results(results_without_batching)


def test_intentional_worker_retirement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tests_project = make_fake_tests_project(tmp_path, num_test_files=3)
    test_paths = sorted(fake_tests_project.test_case_count_by_test_file_path)
    test_paths[0].write_text(
        dedent(
            """\
            def test_failure() -> None:
                assert False
            """
        )
    )
    # Load the real retirement hooks without importing the GPU fixtures in the same conftest.
    source = (Path(__file__).parent / "conftest.py").read_text()
    hook_names = {
        "_exit_marker_dir",
        "pytest_configure",
        "pytest_unconfigure",
        "pytest_runtest_logreport",
        "pytest_handlecrashitem",
    }
    hooks = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in hook_names]
    assert {hook.name for hook in hooks} == hook_names
    conftest_path = tmp_path / "tests" / "python" / "conftest.py"
    conftest_path.write_text(
        dedent(
            """\
            import os, tempfile
            from pathlib import Path
            """
        )
        + conftest_path.read_text()
        + "\n"
        + "\n\n".join(ast.unparse(hook) for hook in hooks)
        + dedent(
            """

            from xdist.workermanage import WorkerController

            def pytest_testnodedown(node: WorkerController, error: object | None) -> None:
                # Verify that the worker really exited with an intentional-retirement marker.
                if error is not None and (Path(_exit_marker_dir()) / node.gateway.id).exists():
                    with Path('retired_workers.txt').open('a') as retired_workers:
                        print(node.gateway.id, file=retired_workers)
            """
        )
    )
    monkeypatch.delenv("_QD_XDIST_EXIT_MARKER_DIR", raising=False)
    # One xdist worker exercises replacement without introducing a concurrent work-stealing race.
    monkeypatch.setenv("PYTEST_ADDOPTS", "-q -p xdist.plugin -n 1 --dist=worksteal --max-worker-restart=999999")
    results_without_batching = tmp_path / "full.xml"
    # Run tests without batching, including the intentional retirement after the failure.
    unbatched_process_result = subprocess.run(
        [
            sys.executable,
            str(fake_tests_project.runner_path),
            f"--junitxml={results_without_batching}",
            f"--basetemp={tmp_path / 'full-tmp'}",
        ],
        cwd=tmp_path,
        check=False,
    )
    assert unbatched_process_result.returncode == 1
    retired_workers_path = tmp_path / "retired_workers.txt"
    assert len(retired_workers_path.read_text().splitlines()) == 1
    output_with_batching = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output_with_batching, [], batch_size=2) == 1
    # One retirement in the unbatched run and one in the batched run.
    assert len(retired_workers_path.read_text().splitlines()) == 2
    combined_test_results = parse_test_results(output_with_batching / "results.xml")
    assert combined_test_results == parse_test_results(results_without_batching)
    assert list(combined_test_results.values()).count("failure") == 1
    assert "error" not in combined_test_results.values()
    assert "crashed while running" not in (output_with_batching / "results.xml").read_text()


def test_worker_crash_is_reported_and_later_batch_runs(tmp_path: Path) -> None:
    fake_tests_project = make_fake_tests_project(tmp_path, num_test_files=2)
    test_paths = sorted(fake_tests_project.test_case_count_by_test_file_path)
    test_paths[0].write_text(
        dedent(
            """\
            import os

            def test_worker_crash() -> None:
                # Terminate the worker process without raising a normal test assertion failure.
                os._exit(1)
            """
        )
    )
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, [], batch_size=1) == 1
    batch_execution_records = json.loads((output / "results.json").read_text())
    assert [(record["batch"], record["test_runner_exit_code"]) for record in batch_execution_records] == [
        (0, 1),
        (1, 0),
    ]
    crashed_batch_test_results = parse_test_results(output / "batch-00.xml")
    later_batch_test_results = parse_test_results(output / "batch-01.xml")
    assert any(outcome in ("failure", "error") for outcome in crashed_batch_test_results.values())
    assert "test_worker_crash" in (test_name for _, test_name in crashed_batch_test_results)
    assert len(later_batch_test_results) == fake_tests_project.test_case_count_by_test_file_path[test_paths[1]]
    assert set(later_batch_test_results.values()) == {"pass", "skipped"}
    assert parse_test_results(output / "results.xml") == crashed_batch_test_results | later_batch_test_results
    combined_report = ET.parse(output / "results.xml").getroot()
    assert "crashed while running" in ET.tostring(combined_report, encoding="unicode")
    for attribute in ("tests", "failures", "errors", "skipped"):
        batch_total = sum(
            int(suite.get(attribute))
            for batch in ("batch-00.xml", "batch-01.xml")
            for suite in ET.parse(output / batch).iter("testsuite")
        )
        assert int(combined_report.get(attribute)) == batch_total


@pytest.mark.parametrize("ignore_option", ["--ignore", "--ignore-glob"])
def test_selected_files_respect_pytest_ignore_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ignore_option: str
) -> None:
    num_test_files = 13
    batch_size = 12
    project = make_fake_tests_project(tmp_path, num_test_files=num_test_files)
    ignored = sorted(project.test_case_count_by_test_file_path)[0]
    ignored.write_text("raise AssertionError('Ignored test module must not be imported')\n")
    monkeypatch.setenv(
        "PYTEST_ADDOPTS", shlex.join(["-q", "-p", "xdist.plugin", "-n", "3", f"{ignore_option}={ignored}"])
    )
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, [], batch_size=batch_size) == 0
    report = ET.parse(output / "results.xml")
    assert (
        len(list(report.iter("testcase"))) == project.total_cases - project.test_case_count_by_test_file_path[ignored]
    )


@pytest.mark.parametrize("use_xdist", [False, True])
def test_collection_error_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_xdist: bool) -> None:
    if not use_xdist:
        monkeypatch.setenv("PYTEST_ADDOPTS", "-q")
    fake_tests_project = make_fake_tests_project(tmp_path, num_test_files=1)
    test_path = next(iter(fake_tests_project.test_case_count_by_test_file_path))
    test_path.write_text("raise RuntimeError('Cannot collect tests')\n")
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, []) == 1
    record = json.loads((output / "results.json").read_text())[0]
    assert record["test_runner_exit_code"] != 0
    assert "error" in parse_test_results(output / "results.xml").values()


def test_controller_crash_returns_failure(tmp_path: Path) -> None:
    make_fake_tests_project(tmp_path, num_test_files=1)
    conftest_path = tmp_path / "tests" / "python" / "conftest.py"
    conftest_path.write_text(
        dedent(
            """\
            import os, pytest

            def pytest_sessionstart(session: pytest.Session) -> None:
                if not hasattr(session.config, 'workerinput'):
                    os._exit(3)
            """
        )
    )
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, []) == 1
    record = json.loads((output / "results.json").read_text())[0]
    assert record["test_runner_exit_code"] == 3
    assert not parse_test_results(output / "results.xml")


@pytest.mark.parametrize("report_contents", [None, "<broken"])
def test_unusable_report_fails_batch_even_when_tests_pass(tmp_path: Path, report_contents: str | None) -> None:
    fake_tests_project = make_fake_tests_project(tmp_path, num_test_files=1)
    fake_tests_project.runner_path.write_text(
        dedent(
            f"""\
            import os, pytest, sys
            from pathlib import Path

            exit_code = pytest.main(['tests/python', *sys.argv[1:]])
            report_path = Path(os.environ['QD_BATCH_SELECTION']).with_name('batch-00.xml')
            report_contents = {report_contents!r}
            if report_contents is None:
                report_path.unlink()
            else:
                report_path.write_text(report_contents)
            raise SystemExit(exit_code)
            """
        )
    )
    output = tmp_path / "batches"
    assert run_test_batches.run_batches(tmp_path, output, []) == 1
    record = json.loads((output / "results.json").read_text())[0]
    assert record["test_runner_exit_code"] == 0
    assert not parse_test_results(output / "results.xml")


@pytest.mark.parametrize("exit_code", [0, 3])
def test_runner_exit_before_pytest_is_not_success(tmp_path: Path, exit_code: int) -> None:
    fake_tests_project = make_fake_tests_project(tmp_path, num_test_files=1)
    fake_tests_project.runner_path.write_text(f"raise SystemExit({exit_code})\n")
    output = tmp_path / "batches"
    output.mkdir()
    # An old report must not conceal a run that never reached pytest.
    (output / "batch-00.xml").write_text('<testsuites><testsuite tests="1" failures="0"/></testsuites>')
    assert run_test_batches.run_batches(tmp_path, output, []) == 1
    record = json.loads((output / "results.json").read_text())[0]
    assert record["test_runner_exit_code"] == exit_code
    assert not parse_test_results(output / "results.xml")
