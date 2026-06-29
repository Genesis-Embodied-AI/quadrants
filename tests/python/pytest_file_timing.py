"""Pytest plugin that reports wall-clock time spent per test file.

Activated by the environment variable QD_FILE_TIMING=1.  Collects the "call" phase duration of each test item and
prints a sorted summary at the end of the session.

Works correctly with pytest-xdist: the controller process receives forwarded reports from all workers and aggregates
them here.

Set QD_FILE_TIMING_OUTPUT to a file path to also write the results as markdown (useful for GitHub Actions job
summaries).
"""

import os
from collections import defaultdict

_active = os.environ.get("QD_FILE_TIMING", "0") == "1"

_file_durations: dict[str, float] = defaultdict(float)
_file_test_counts: dict[str, int] = defaultdict(int)


def pytest_runtest_logreport(report):
    if not _active:
        return
    if report.when == "call":
        fspath = report.fspath
        _file_durations[fspath] += report.duration
        _file_test_counts[fspath] += 1


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _active:
        return
    if not _file_durations:
        return

    tw = terminalreporter._tw
    tw.sep("=", "per-file timing summary")
    tw.line(f"{'Duration (s)':>12}  {'Tests':>6}  File")
    tw.sep("-")

    sorted_files = sorted(_file_durations.items(), key=lambda x: -x[1])

    total = 0.0
    for fspath, duration in sorted_files:
        count = _file_test_counts[fspath]
        tw.line(f"{duration:12.2f}  {count:6d}  {fspath}")
        total += duration

    tw.sep("-")
    total_tests = sum(_file_test_counts.values())
    tw.line(f"{total:12.2f}  {total_tests:6d}  TOTAL (sum of per-file call durations)")
    tw.sep("=")

    output_path = os.environ.get("QD_FILE_TIMING_OUTPUT")
    if output_path:
        _write_markdown(sorted_files, total, total_tests, output_path)


def _write_markdown(sorted_files, total, total_tests, path):
    lines = [
        "### Per-file test timing",
        "",
        "| Duration (s) | Tests | File |",
        "|---:|---:|:---|",
    ]
    for fspath, duration in sorted_files:
        count = _file_test_counts[fspath]
        basename = os.path.basename(fspath)
        lines.append(f"| {duration:.2f} | {count} | `{basename}` |")
    lines.append(f"| **{total:.2f}** | **{total_tests}** | **TOTAL** |")
    lines.append("")

    with open(path, "a") as f:
        f.write("\n".join(lines))
