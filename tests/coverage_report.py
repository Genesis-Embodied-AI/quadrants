#!/usr/bin/env python3
"""Combine kernel coverage data and generate diff coverage reports.

Run tests first with run_tests.py --coverage, then use this script:

  # Local dev: combine coverage data and generate HTML diff report
  python tests/coverage_report.py

  # CI: combine coverage data and generate coverage.xml (no diff report)
  python tests/coverage_report.py --collect-only

  # CI: generate diff report from previously collected coverage.xml files
  python tests/coverage_report.py --report-only --format markdown \\
      --coverage-xml coverage-cpu/coverage.xml coverage-cuda/coverage.xml
"""

import argparse
import glob
import html as html_mod
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _run(cmd, **kwargs):
    print(f"{DIM}$ {cmd}{RESET}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=REPO_ROOT, **kwargs)


def combine_coverage():
    """Combine pytest-cov and kernel coverage data, applying path remapping."""
    pytest_cov = REPO_ROOT / ".coverage"
    if not pytest_cov.exists():
        return
    pytest_cov.rename(REPO_ROOT / ".coverage.pytest")
    kcov_files = glob.glob(str(REPO_ROOT / "_qd_kcov.*"))
    combine_args = [".coverage.pytest"] + [os.path.basename(f) for f in kcov_files]
    result = _run(f"coverage combine {' '.join(combine_args)}")
    if result.returncode != 0 and kcov_files:
        _run("coverage combine .coverage.pytest")


def generate_artifacts():
    """Generate coverage.xml and pytest-coverage.txt from the combined .coverage."""
    _run("coverage xml -o coverage.xml --ignore-errors")
    _run("coverage report --show-missing --skip-covered --ignore-errors > pytest-coverage.txt")


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


class _Renderer:
    """Base class for coverage report renderers."""

    def begin(self, total_hit, total_miss, total_pct):
        pass

    def begin_file(self, filename, pct, missing):
        pass

    def write_line(self, lineno, text, status):
        pass

    def end_file(self):
        pass

    def finish(self):
        pass

    def output(self):
        return None


class _TerminalRenderer(_Renderer):
    def begin(self, total_hit, total_miss, total_pct):
        self._total_hit, self._total_miss, self._total_pct = total_hit, total_miss, total_pct
        print(f"\n{BOLD}Diff Coverage Report{RESET}")
        print("=" * 70)

    def begin_file(self, filename, pct, missing):
        color = GREEN if pct >= 80 else RED
        missing_str = f"  Missing: {_format_ranges(missing)}" if missing else ""
        print(f"  {filename}: {color}{pct:.0f}%{RESET}{missing_str}")

    def finish(self):
        print("-" * 70)
        color = GREEN if self._total_pct >= 80 else RED
        total = self._total_hit + self._total_miss
        print(f"  {BOLD}Total: {total} lines, {self._total_miss} missing, {color}{self._total_pct:.0f}%{RESET}")


class _AnnotatedRenderer(_TerminalRenderer):
    """Print grouped summary table first, then all annotated file blocks."""

    _STATUS_FMT = {"hit": (GREEN, "\u2713"), "miss": (RED, "\u2717"), "no_data": (DIM, " ")}

    def begin(self, total_hit, total_miss, total_pct):
        super().begin(total_hit, total_miss, total_pct)
        self._file_blocks: list[tuple[str, float, list[tuple[int, str, str]]]] = []

    def begin_file(self, filename, pct, missing):
        super().begin_file(filename, pct, missing)
        self._cur_filename, self._cur_pct = filename, pct
        self._cur_lines: list[tuple[int, str, str]] = []

    def write_line(self, lineno, text, status):
        self._cur_lines.append((lineno, text, status))

    def end_file(self):
        if self._cur_lines:
            self._file_blocks.append((self._cur_filename, self._cur_pct, self._cur_lines))

    def finish(self):
        super().finish()
        for filename, pct, lines in self._file_blocks:
            print(f"\n{BOLD}=== {filename} ({pct:.0f}%) ==={RESET}")
            for lineno, text, status in lines:
                color, marker = self._STATUS_FMT[status]
                print(f"{color} {marker} {lineno:4d}{RESET} {color}{text}{RESET}")


_COMMIT_HASH_OVERRIDE = None


def _get_commit_hash():
    if _COMMIT_HASH_OVERRIDE:
        return _COMMIT_HASH_OVERRIDE
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    ).stdout.strip()


class _MarkdownRenderer(_Renderer):
    """Full annotated markdown report (for GitHub Check body)."""

    _STATUS_MARKER = {"hit": "🟢", "miss": "🔴", "no_data": "  "}

    def __init__(self):
        self._buf = []

    def _print(self, text=""):
        self._buf.append(text)

    def begin(self, total_hit, total_miss, total_pct):
        commit = _get_commit_hash()
        heading = f"## Coverage Report (`{commit}`)\n" if commit else "## Coverage Report\n"
        self._print(heading)
        self._print("| Metric | Value |")
        self._print("|--------|-------|")
        self._print(f"| **Diff coverage** (changed lines only) | **{total_pct:.0f}%** |")
        overall = _get_overall_coverage()
        if overall:
            self._print(f"| Overall project coverage | {overall} |")
        self._print()
        self._print(f"**Total**: {total_hit + total_miss} lines, {total_miss} missing, {total_pct:.0f}% covered\n")

    def begin_file(self, filename, pct, missing):
        icon = "🟢" if pct >= 80 else "🔴"
        self._print(f"<details><summary>{icon} <code>{filename}</code> ({pct:.0f}%)</summary>\n")
        self._print("```")

    def write_line(self, lineno, text, status):
        self._print(f"{self._STATUS_MARKER[status]} {lineno:4d}  {text}")

    def end_file(self):
        self._print("```\n</details>\n")

    def output(self):
        return "\n".join(self._buf)


class _MarkdownSummaryRenderer(_Renderer):
    """One-line summary for the PR comment (turned into a link by the workflow)."""

    def __init__(self):
        self._total_hit = self._total_miss = 0
        self._total_pct = 0.0

    def begin(self, total_hit, total_miss, total_pct):
        self._total_hit, self._total_miss, self._total_pct = total_hit, total_miss, total_pct

    def begin_file(self, filename, pct, missing):
        pass

    def finish(self):
        pass

    def output(self):
        total = self._total_hit + self._total_miss
        return f"Diff coverage: {self._total_pct:.0f}% · {total} lines, {self._total_miss} missing"


_HTML_CSS = """\
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
       max-width: 960px; margin: 2rem auto; padding: 0 1rem; background: #1e1e1e; color: #d4d4d4; }
h1 { color: #e0e0e0; }
table.summary { border-collapse: collapse; margin: 1rem 0; }
table.summary td, table.summary th { padding: 0.4rem 1rem; border: 1px solid #444; }
table.summary th { background: #2d2d2d; text-align: left; }
details { margin: 0.5rem 0; }
summary { cursor: pointer; padding: 0.4rem; background: #2d2d2d; border-radius: 4px; }
summary:hover { background: #363636; }
.file-header { font-weight: bold; }
.pct-good { color: #4ec9b0; }
.pct-bad { color: #f44747; }
pre { margin: 0; padding: 0.5rem; background: #1a1a1a; border-radius: 4px; overflow-x: auto;
      font-size: 13px; line-height: 1.5; }
.line { display: block; }
.hit { background: #1e3a1e; }
.miss { background: #3a1e1e; }
.no-data { opacity: 0.5; }
.lineno { display: inline-block; width: 4em; text-align: right; color: #858585;
          margin-right: 1em; user-select: none; }
.status { display: inline-block; width: 1.5em; text-align: center; }
.status-hit { color: #4ec9b0; }
.status-miss { color: #f44747; }"""

_HTML_STATUS = {
    "hit": ("hit", '<span class="status status-hit">&#10003;</span>'),
    "miss": ("miss", '<span class="status status-miss">&#10007;</span>'),
    "no_data": ("no-data", '<span class="status"> </span>'),
}


class _HtmlRenderer(_Renderer):
    def __init__(self, output_path=None):
        self._out_path = Path(output_path) if output_path else REPO_ROOT / "coverage-report.html"
        self._parts = []

    def begin(self, total_hit, total_miss, total_pct):
        overall = _get_overall_coverage()
        self._parts.append(
            f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>Diff Coverage Report</title>\n'
            f"<style>\n{_HTML_CSS}\n</style></head><body>\n<h1>Diff Coverage Report</h1>"
        )
        pct_cls = "pct-good" if total_pct >= 80 else "pct-bad"
        self._parts.append('<table class="summary"><tr><th>Metric</th><th>Value</th></tr>')
        self._parts.append(
            f'<tr><td>Diff coverage (changed lines)</td><td class="{pct_cls}"><b>{total_pct:.0f}%</b></td></tr>'
        )
        if overall:
            self._parts.append(f"<tr><td>Overall project coverage</td><td>{overall}</td></tr>")
        self._parts.append(
            f"<tr><td>Total lines</td><td>{total_hit + total_miss} ({total_miss} missing)</td></tr></table>"
        )

    def begin_file(self, filename, pct, missing):
        pct_cls = "pct-good" if pct >= 80 else "pct-bad"
        missing_str = f" &mdash; missing: {_format_ranges(missing)}" if missing else ""
        self._parts.append(
            f'<details><summary><span class="file-header">{html_mod.escape(filename)}</span>'
            f' <span class="{pct_cls}">{pct:.0f}%</span>{missing_str}</summary><pre>'
        )
        self._line_parts = []

    def write_line(self, lineno, text, status):
        cls, icon = _HTML_STATUS[status]
        escaped = html_mod.escape(text)
        self._line_parts.append(f'<span class="line {cls}"><span class="lineno">{lineno}</span>{icon}{escaped}</span>')

    def end_file(self):
        self._parts.append("".join(self._line_parts) + "</pre></details>")

    def finish(self):
        self._parts.append("</body></html>")
        self._out_path.write_text("\n".join(self._parts))
        print(f"Coverage report written to {self._out_path}")


_RENDERERS = {
    "terminal": _TerminalRenderer,
    "annotated": _AnnotatedRenderer,
    "markdown": _MarkdownRenderer,
    "markdown-summary": _MarkdownSummaryRenderer,
    "html": _HtmlRenderer,
}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def get_diff_lines(compare_branch):
    """Return {filename: [(lineno, text)]} for added/modified lines."""
    result = subprocess.run(
        ["git", "diff", "-U0", compare_branch],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
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
                diff_lines.setdefault(current_file, []).append((current_lineno, line[1:]))
            current_lineno += 1
        elif line.startswith("\\"):
            continue
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


def generate_report(compare_branch, coverage_xmls, output_format="terminal", output_path=None):
    """Generate the diff coverage report."""
    diff_lines = get_diff_lines(compare_branch)
    coverage = get_covered_lines(coverage_xmls)

    files_report = []
    total_hit = 0
    total_miss = 0

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
        missing = [ln for ln, _, s in line_details if s == "miss"]

        files_report.append(
            {
                "filename": filename,
                "pct": pct,
                "missing": missing,
                "lines": line_details,
            }
        )

    total_pct = (total_hit / (total_hit + total_miss) * 100) if (total_hit + total_miss) else 0

    renderer_cls = _RENDERERS[output_format]
    renderer = renderer_cls(output_path=output_path) if output_format == "html" else renderer_cls()
    renderer.begin(total_hit, total_miss, total_pct)
    for fr in files_report:
        renderer.begin_file(fr["filename"], fr["pct"], fr["missing"])
        for lineno, text, status in fr["lines"]:
            renderer.write_line(lineno, text, status)
        renderer.end_file()
    renderer.finish()

    rendered = renderer.output()
    if rendered is not None:
        print(rendered)

    return total_pct


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
    parser = argparse.ArgumentParser(
        description="Combine kernel coverage data and generate diff coverage reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect-only",
        action="store_true",
        help="Combine coverage data and generate coverage.xml, but skip the diff report",
    )
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="Skip combining, generate report from existing coverage.xml",
    )

    parser.add_argument(
        "--compare-branch",
        default="origin/main",
        help="Branch to compare against (default: origin/main)",
    )
    parser.add_argument(
        "--coverage-xml",
        nargs="+",
        default=None,
        help="Path(s) to coverage.xml file(s). Default: coverage.xml in repo root",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="html",
        choices=["html", "terminal", "annotated", "markdown", "markdown-summary"],
        help="Output format (default: html)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path for HTML format (default: coverage-report.html in repo root)",
    )
    parser.add_argument(
        "--commit-hash",
        default=None,
        help="Override the commit hash shown in the report heading (default: git rev-parse HEAD)",
    )

    args = parser.parse_args()

    global _COMMIT_HASH_OVERRIDE
    _COMMIT_HASH_OVERRIDE = args.commit_hash

    if not args.report_only:
        combine_coverage()
        generate_artifacts()

    if args.collect_only:
        return 0

    xml_paths = args.coverage_xml or [str(REPO_ROOT / "coverage.xml")]
    xml_paths = [p for p in xml_paths if os.path.exists(p)]
    if not xml_paths:
        print("No coverage.xml found. Run tests first or specify --coverage-xml.", file=sys.stderr)
        sys.exit(1)

    total_pct = generate_report(args.compare_branch, xml_paths, args.output_format, output_path=args.output)
    return 0 if total_pct >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
