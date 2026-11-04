"""Tests for report formatting; these do not import or build Quadrants."""

import json

from build_inputs import FileSummary, group_summaries, render_comment_markdown
from render_report import render


def test_groups_use_top_level_folders_and_preserve_order():
    paths = [
        "docs/conf.py",
        "quadrants/a.cpp",
        "cmake/helper.py",
        "python/a.py",
        "tests/a.py",
        ".github/helper.py",
        "setup.py",
        "other/python/a.py",
        "docs_extra/a.py",
        "docs",
        "quadrants/b.cpp",
    ]
    summaries = [FileSummary(path, "py", 10, 2, 1) for path in paths]
    groups = group_summaries(summaries)
    assert list(groups) == ["docs", "c++", "python", "tests", "github", "misc"]
    assert {name: [s.path for s in files] for name, files in groups.items()} == {
        "docs": ["docs/conf.py"],
        "c++": ["quadrants/a.cpp", "cmake/helper.py", "quadrants/b.cpp"],
        "python": ["python/a.py"],
        "tests": ["tests/a.py"],
        "github": [".github/helper.py"],
        "misc": ["setup.py", "other/python/a.py", "docs_extra/a.py", "docs"],
    }


def test_markdown_totals_include_deleted_and_zero_change_files():
    summaries = [
        FileSummary("quadrants/new.cpp", "cpp", 0, 7, 0),
        FileSummary("cmake/deleted.py", "py", 5, 0, 5, is_deleted=True),
        FileSummary("tests/comment.py", "py", 3, 0, 0),
        FileSummary("setup.py", "py", 10, 2, 1),
    ]
    report = render_comment_markdown(summaries, "abc123")
    assert "| c++ | 2 | 5 | +7 | -5 |" in report
    assert "| docs | 0 | 0 | +0 | -0 |" in report
    assert "| tests | 1 | 3 | +0 | -0 |" in report
    assert "| misc | 1 | 10 | +2 | -1 |" in report
    assert "| **Total** | 4 | 18 | +9 | -6 |" in report
    assert "**Group total**: 2 file(s) changed, +7 -5 code lines." in report
    assert report.endswith("**Total**: 4 file(s) changed, +9 -6 code lines.\n")
    assert "### docs" not in report
    assert report.index("### c++") < report.index("### tests") < report.index("### misc")
    for summary in summaries:
        assert report.count(f"`{summary.path}`") == 1


def test_detailed_report_groups_files_and_preserves_attribution(tmp_path):
    summaries = [
        FileSummary("tests/deleted.py", "py", 2, 0, 2, is_deleted=True),
        FileSummary("python/new.py", "py", 0, 2, 0),
    ]
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps([vars(s) for s in summaries]))
    head = tmp_path / "head/python/new.py"
    head.parent.mkdir(parents=True)
    head.write_text("def hello():\n    return 1\n")
    diff = tmp_path / "diffs/python/new.py.diff"
    diff.parent.mkdir(parents=True)
    diff.write_text("@@ -0,0 +1,2 @@\n+def hello():\n+    return 1\n")
    ranges = tmp_path / "ranges.jsonl"
    ranges.write_text(json.dumps({"path": "python/new.py", "name": "hello", "head": [1, 2], "base": None}))
    report = render(summary_path, ranges, tmp_path)
    assert report.index("python: 1 file(s)") < report.index("tests: 1 file(s)")
    assert "hello()" in report
    assert "# entire file deleted" in report
    assert "Total: 2 file(s) changed, +2 -2 code lines." in report
    # Deletion-only reports do not require any agent output.
    summary_path.write_text(json.dumps([vars(summaries[0])]))
    report = render(summary_path, tmp_path / "missing.jsonl", tmp_path)
    assert "tests: 1 file(s) changed, +0 -2 code lines." in report
    assert "Total: 1 file(s) changed, +0 -2 code lines." in report


def test_empty_report(tmp_path):
    assert "No source files" in render_comment_markdown([], "")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("[]")
    assert render(summary_path, tmp_path / "missing.jsonl", tmp_path) == ""
