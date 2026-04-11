#!/usr/bin/env python3
"""Generate a coverage report showing diff coverage for changed lines.

Usage:
    # Run tests with coverage and show diff report (default: compare against main)
    python tests/coverage_report.py

    # Compare against a different branch
    python tests/coverage_report.py --compare-branch origin/my-branch

    # Skip running tests, just generate report from existing coverage data
    python tests/coverage_report.py --report-only

    # Run only specific tests
    python tests/coverage_report.py -k "test_kernel_coverage"

    # CI mode: output markdown for PR comment
    python tests/coverage_report.py --ci --coverage-xml coverage-cpu/coverage.xml coverage-cuda/coverage.xml
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

NO_KCOV_TESTS = (
    "test_offline_cache or test_concurrent_kernels"
    " or test_src_ll_cache_with_corruption or test_fe_ll_observations"
)

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _clean_coverage_files():
    for pattern in [".coverage", ".coverage.*", "_qd_kcov.*", "coverage.xml", "pytest-coverage.txt"]:
        for f in glob.glob(str(REPO_ROOT / pattern)):
            os.remove(f)


def _run(cmd, **kwargs):
    print(f"{DIM}$ {cmd}{RESET}")
    return subprocess.run(cmd, shell=True, cwd=REPO_ROOT, **kwargs)


def run_tests(args):
    """Run tests in phases with coverage collection."""
    _clean_coverage_files()

    extra = ""
    if args.keys:
        kcov_filter = f"not ({NO_KCOV_TESTS})" if args.keys is None else args.keys
    else:
        kcov_filter = None

    threads = f"-t {args.threads}" if args.threads else ""
    arch = f"--arch {args.arch}" if args.arch else ""
    verbose = "-v" if args.verbose else ""
    rerun = f"-r {args.rerun}" if args.rerun else ""

    if kcov_filter:
        # User specified -k, run everything in one phase with KCOV
        os.environ["QD_KERNEL_COVERAGE"] = "1"
        _run(
            f"python tests/run_tests.py {verbose} {rerun} {threads} {arch}"
            f' -k "{kcov_filter}" --coverage'
        )
    else:
        # Phase 1: coverage-incompatible tests (no kernel coverage)
        _run(
            f"python tests/run_tests.py {verbose} {rerun} {threads} {arch}"
            f' -k "{NO_KCOV_TESTS}" --coverage'
        )

        # Phase 2: remaining tests with kernel coverage
        os.environ["QD_KERNEL_COVERAGE"] = "1"
        _run(
            f"python tests/run_tests.py {verbose} {rerun} {threads} {arch}"
            f' -m "not needs_torch" -k "not ({NO_KCOV_TESTS})" --coverage --cov-append'
        )

    # Combine coverage data (triggers path remapping via pyproject.toml)
    pytest_cov = REPO_ROOT / ".coverage"
    if pytest_cov.exists():
        pytest_cov.rename(REPO_ROOT / ".coverage.pytest")
        kcov_files = glob.glob(str(REPO_ROOT / "_qd_kcov.*"))
        combine_args = [".coverage.pytest"] + [os.path.basename(f) for f in kcov_files]
        _run(f"coverage combine {' '.join(combine_args)}")

    _run("coverage xml -o coverage.xml --ignore-errors")
    _run("coverage report --show-missing --skip-covered --ignore-errors > pytest-coverage.txt")


def get_diff_lines(compare_branch):
    """Return {filename: [(lineno, text)]} for added/modified lines."""
    result = subprocess.run(
        ["git", "diff", "-U0", compare_branch],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    diff_lines = {}
    current_file = None
    current_lineno = 0
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            plus_part = [p for p in line.split() if p.startswith("+")][0][1:]
            if "," in plus_part:
                start, count = plus_part.split(",")
                start, count = int(start), int(count)
            else:
                start, count = int(plus_part), 1
            current_lineno = start
        elif line.startswith("+") and not line.startswith("+++"):
            if current_file and current_file.endswith(".py"):
                diff_lines.setdefault(current_file, []).append(
                    (current_lineno, line[1:])
                )
            current_lineno += 1
        elif not line.startswith("-"):
            current_lineno += 1
    return diff_lines


def get_covered_lines(xml_paths):
    """Return {filename: {lineno: hits}} from one or more coverage.xml files."""
    result = {}
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        for cls in tree.getroot().findall(".//class"):
            fn = cls.get("filename")
            for line in cls.findall(".//line"):
                lineno = int(line.get("number"))
                hits = int(line.get("hits", 0))
                result.setdefault(fn, {})
                result[fn][lineno] = result[fn].get(lineno, 0) + hits
    return result


def generate_report(compare_branch, coverage_xmls, output_format="terminal"):
    """Generate the diff coverage report."""
    diff_lines = get_diff_lines(compare_branch)
    coverage = get_covered_lines(coverage_xmls)

    files_report = []
    total_hit = 0
    total_miss = 0
    total_lines = 0

    for filename in sorted(diff_lines):
        lines = diff_lines[filename]
        if not lines:
            continue
        file_cov = coverage.get(filename, {})

        hit = miss = no_data = 0
        line_details = []
        for lineno, text in lines:
            hits = file_cov.get(lineno)
            if hits is None:
                no_data += 1
                status = "no_data"
            elif hits > 0:
                hit += 1
                status = "hit"
            else:
                miss += 1
                status = "miss"
            line_details.append((lineno, text, status))

        measurable = hit + miss
        if measurable == 0:
            continue

        pct = (hit / measurable * 100) if measurable else 0
        total_hit += hit
        total_miss += miss
        total_lines += measurable
        missing = [ln for ln, _, s in line_details if s == "miss"]

        files_report.append({
            "filename": filename,
            "hit": hit,
            "miss": miss,
            "no_data": no_data,
            "pct": pct,
            "missing": missing,
            "lines": line_details,
        })

    total_pct = (total_hit / (total_hit + total_miss) * 100) if (total_hit + total_miss) else 0

    if output_format == "terminal":
        _print_terminal(files_report, total_hit, total_miss, total_pct)
    elif output_format == "annotated":
        _print_annotated(files_report, total_hit, total_miss, total_pct)
    elif output_format == "markdown":
        _print_markdown(files_report, total_hit, total_miss, total_pct)

    return total_pct


def _print_terminal(files_report, total_hit, total_miss, total_pct):
    print(f"\n{BOLD}Diff Coverage Report{RESET}")
    print("=" * 70)
    for fr in files_report:
        color = GREEN if fr["pct"] >= 80 else RED
        missing_str = f"  Missing: {_format_ranges(fr['missing'])}" if fr["missing"] else ""
        print(f"  {fr['filename']}: {color}{fr['pct']:.0f}%{RESET}{missing_str}")
    print("-" * 70)
    color = GREEN if total_pct >= 80 else RED
    print(f"  {BOLD}Total: {total_hit + total_miss} lines, {total_miss} missing, {color}{total_pct:.0f}%{RESET}")


def _print_annotated(files_report, total_hit, total_miss, total_pct):
    _print_terminal(files_report, total_hit, total_miss, total_pct)
    print()
    for fr in files_report:
        print(f"\n{BOLD}=== {fr['filename']} ({fr['pct']:.0f}%) ==={RESET}")
        for lineno, text, status in fr["lines"]:
            if status == "hit":
                print(f"{GREEN} \u2713 {lineno:4d}{RESET} {GREEN}{text}{RESET}")
            elif status == "miss":
                print(f"{RED} \u2717 {lineno:4d}{RESET} {RED}{text}{RESET}")
            else:
                print(f"{DIM}   {lineno:4d}{RESET} {DIM}{text}{RESET}")


def _print_markdown(files_report, total_hit, total_miss, total_pct):
    overall = _get_overall_coverage()
    print("## Coverage Report\n")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| **Diff coverage** (changed lines only) | **{total_pct:.0f}%** |")
    if overall:
        print(f"| Overall project coverage | {overall} |")
    print()
    print("### Changed files\n")
    for fr in files_report:
        missing_str = f": Missing lines {_format_ranges(fr['missing'])}" if fr["missing"] else ""
        print(f"- {fr['filename']} ({fr['pct']:.0f}%){missing_str}")
    print(f"\n**Total**: {total_hit + total_miss} lines, {total_miss} missing, {total_pct:.0f}% covered")


def _get_overall_coverage():
    """Extract overall coverage % from pytest-coverage.txt if it exists."""
    for path in [REPO_ROOT / "pytest-coverage.txt", REPO_ROOT / "coverage-cpu" / "pytest-coverage.txt"]:
        if path.exists():
            for line in reversed(path.read_text().splitlines()):
                if line.startswith("TOTAL"):
                    match = re.search(r"(\d+%)", line)
                    if match:
                        return match.group(1)
    return None


def _format_ranges(numbers):
    """Format [1,2,3,5,7,8,9] as '1-3,5,7-9'."""
    if not numbers:
        return ""
    ranges = []
    start = prev = numbers[0]
    for n in numbers[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def main():
    parser = argparse.ArgumentParser(description="Generate diff coverage report")
    parser.add_argument(
        "--compare-branch", default="origin/main",
        help="Branch to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip running tests, use existing coverage.xml",
    )
    parser.add_argument(
        "--coverage-xml", nargs="+", default=None,
        help="Path(s) to coverage.xml file(s). Default: coverage.xml in repo root",
    )
    parser.add_argument(
        "--format", dest="output_format", default="annotated",
        choices=["terminal", "annotated", "markdown"],
        help="Output format (default: annotated)",
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: output markdown format",
    )
    parser.add_argument("-k", dest="keys", default=None, help="Test filter expression")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--threads", default=None)
    parser.add_argument("-r", "--rerun", default=None)
    parser.add_argument("--arch", default=None)

    args = parser.parse_args()

    if args.ci:
        args.output_format = "markdown"

    if not args.report_only:
        run_tests(args)

    xml_paths = args.coverage_xml or [str(REPO_ROOT / "coverage.xml")]
    xml_paths = [p for p in xml_paths if os.path.exists(p)]
    if not xml_paths:
        print("No coverage.xml found. Run tests first or specify --coverage-xml.", file=sys.stderr)
        sys.exit(1)

    pct = generate_report(args.compare_branch, xml_paths, args.output_format)
    return 0


if __name__ == "__main__":
    sys.exit(main())
