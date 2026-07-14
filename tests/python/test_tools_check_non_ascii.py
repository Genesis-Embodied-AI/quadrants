import importlib.util
import sys
from pathlib import Path

# python/tools is intentionally not a package, so load the tool module by file path
# (mirrors tests/python/test_tools_markdown_check.py).
_tool_path = Path(__file__).parent.parent.parent / "python" / "tools" / "check_non_ascii.py"
_spec = importlib.util.spec_from_file_location("check_non_ascii", _tool_path)
check_non_ascii = importlib.util.module_from_spec(_spec)
# Register before exec so the module's @dataclass can resolve its (stringized, due to
# `from __future__ import annotations`) field annotations via sys.modules[cls.__module__].
sys.modules["check_non_ascii"] = check_non_ascii
_spec.loader.exec_module(check_non_ascii)

find_violations = check_non_ascii.find_violations

# Non-ASCII characters are written as escapes so this test file itself stays pure ASCII (otherwise
# the check_non_ascii CI job would flag its own test). \u2019 = right single quote, \u2014 = em dash,
# \u00a0 = non-breaking space, \u00e9 = latin small e with acute, \u0080 = C1 control (no name).


def _diff(*lines):
    return "\n".join(lines) + "\n"


def test_clean_diff_has_no_violations():
    diff = _diff(
        "diff --git a/foo.py b/foo.py",
        "--- a/foo.py",
        "+++ b/foo.py",
        "@@ -1,1 +1,2 @@",
        " existing line",
        "+a brand new ascii line",
    )
    assert find_violations(diff) == []


def test_detects_non_ascii_with_location():
    diff = _diff(
        "diff --git a/foo.py b/foo.py",
        "--- a/foo.py",
        "+++ b/foo.py",
        "@@ -10,1 +10,2 @@",
        " context",
        "+x = \u2019quote\u2019",
    )
    violations = find_violations(diff)
    assert len(violations) == 2
    first, second = violations
    # Context " context" is new line 10, so the added line is line 11.
    assert (first.path, first.line, first.char) == ("foo.py", 11, "\u2019")
    # Content after '+' is `x = 'quote'`: quotes sit at 0-based indices 4 and 10 -> 1-based cols 5, 11.
    assert first.col == 5
    assert second.col == 11


def test_line_numbers_account_for_removed_lines():
    # Removed ("-") lines carry a non-ASCII char but must not be flagged, and must not advance the
    # new-file line counter.
    diff = _diff(
        "diff --git a/foo.py b/foo.py",
        "--- a/foo.py",
        "+++ b/foo.py",
        "@@ -5,2 +5,2 @@",
        " keep",
        "-removed \u2014",
        "+added \u2014",
    )
    violations = find_violations(diff)
    assert len(violations) == 1
    assert violations[0].line == 6
    assert violations[0].char == "\u2014"


def test_multiple_files_and_new_file_paths():
    diff = _diff(
        "diff --git a/a.py b/a.py",
        "--- a/a.py",
        "+++ b/a.py",
        "@@ -1,0 +1,1 @@",
        "+alpha \u00a0",
        "diff --git a/b.txt b/b.txt",
        "new file mode 100644",
        "--- /dev/null",
        "+++ b/b.txt",
        "@@ -0,0 +1,1 @@",
        "+beta \u00e9",
    )
    violations = find_violations(diff)
    assert [(v.path, v.line) for v in violations] == [("a.py", 1), ("b.txt", 1)]
    assert violations[0].char == "\u00a0"
    assert violations[1].char == "\u00e9"


def test_hunk_header_without_line_counts():
    diff = _diff(
        "--- a/foo.py",
        "+++ b/foo.py",
        "@@ -1 +1 @@",
        "+lonely \u2014",
    )
    violations = find_violations(diff)
    assert len(violations) == 1
    assert violations[0].line == 1


def test_no_newline_marker_is_ignored():
    diff = _diff(
        "--- a/foo.py",
        "+++ b/foo.py",
        "@@ -1,1 +1,1 @@",
        "-old",
        "+new \u2014",
        "\\ No newline at end of file",
    )
    violations = find_violations(diff)
    assert len(violations) == 1
    assert violations[0].line == 1


def test_added_line_starting_with_pluses_is_scanned_not_treated_as_header():
    # An added source line whose content starts with "++ " (e.g. C++ prefix decrement of a deref)
    # is emitted by git as "+++ ...". It must be scanned as an added line, not parsed as a "+++"
    # file header (which would skip it and let its non-ASCII slip through). Regression for
    # PR #778 review comment r3571557923.
    diff = _diff(
        "diff --git a/foo.cpp b/foo.cpp",
        "--- a/foo.cpp",
        "+++ b/foo.cpp",
        "@@ -1,1 +1,2 @@",
        " int main() {",
        "+++ *p; // caf\u00e9",
    )
    violations = find_violations(diff)
    assert len(violations) == 1
    v = violations[0]
    # Context " int main() {" is new line 1, so the added line is line 2. Content after the diff's
    # leading "+" is "++ *p; // caf" + U+00E9 (e-acute), which sits at 0-based index 13 -> col 14.
    assert (v.path, v.line, v.col, v.char) == ("foo.cpp", 2, 14, "\u00e9")


def test_pluses_content_does_not_corrupt_following_file_path():
    # If the "+++ ..." added line were mistaken for a header it would also clobber the tracked path
    # (and drop the violation), corrupting attribution for the next file. Verify both files report.
    diff = _diff(
        "diff --git a/a.cpp b/a.cpp",
        "--- a/a.cpp",
        "+++ b/a.cpp",
        "@@ -0,0 +1,1 @@",
        "+++ added \u00e9",
        "diff --git a/b.cpp b/b.cpp",
        "--- a/b.cpp",
        "+++ b/b.cpp",
        "@@ -0,0 +1,1 @@",
        "+plain \u2014 here",
    )
    violations = find_violations(diff)
    assert [(v.path, v.line, v.char) for v in violations] == [
        ("a.cpp", 1, "\u00e9"),
        ("b.cpp", 1, "\u2014"),
    ]


def test_removed_line_starting_with_dashes_is_not_a_header():
    # A removed source line starting with "-- " is emitted as "--- ...". It must not be parsed as a
    # "--- " file header, and it must not advance the new-file line counter.
    diff = _diff(
        "diff --git a/foo.py b/foo.py",
        "--- a/foo.py",
        "+++ b/foo.py",
        "@@ -1,2 +1,2 @@",
        " keep",
        "--- dashes removed",
        "+kept \u2014 added",
    )
    violations = find_violations(diff)
    assert len(violations) == 1
    v = violations[0]
    # " keep" is new line 1; the removed line does not advance the new counter, so the added line
    # is new line 2. Content after "+" is "kept " + U+2014 (em dash) + " added": index 5 -> col 6.
    assert (v.path, v.line, v.col, v.char) == ("foo.py", 2, 6, "\u2014")


def test_describe_format():
    v = check_non_ascii.Violation(path="foo.py", line=3, col=7, char="\u2014")
    text = v.describe()
    assert text.startswith("foo.py:3:7:")
    assert "U+2014" in text
    assert "EM DASH" in text
    assert repr("\u2014") in text


def test_describe_unnamed_char():
    v = check_non_ascii.Violation(path="f", line=1, col=1, char="\u0080")
    text = v.describe()
    assert "U+0080" in text
    assert "<unnamed>" in text


def test_main_diff_file_reports_and_fails(tmp_path, capsys):
    patch = tmp_path / "d.patch"
    patch.write_text(
        _diff(
            "--- a/foo.py",
            "+++ b/foo.py",
            "@@ -1,0 +1,1 @@",
            "+bad \u2019",
        ),
        encoding="utf-8",
    )
    rc = check_non_ascii.main(["--diff-file", str(patch)])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.startswith("FAIL")
    assert "foo.py:1:5:" in out


def test_main_diff_file_passes_on_clean(tmp_path, capsys):
    patch = tmp_path / "d.patch"
    patch.write_text(
        _diff(
            "--- a/foo.py",
            "+++ b/foo.py",
            "@@ -1,0 +1,1 @@",
            "+totally fine",
        ),
        encoding="utf-8",
    )
    rc = check_non_ascii.main(["--diff-file", str(patch)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("PASS")
