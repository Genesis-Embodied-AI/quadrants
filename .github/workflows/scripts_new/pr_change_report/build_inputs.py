#!/usr/bin/env python3
"""Build inputs for the PR change report agent.

For each source file changed in the PR (.py / .c / .cc / .cpp / .h / .hpp / .cu) this writes:

  ``<output_dir>/summary.json``        list of ``{path, language, total, added, removed}``
  ``<output_dir>/file_list.txt``       one path per line (same order as ``summary.json``)
  ``<output_dir>/report_header.md``    pre-formatted file-header lines (verbatim for the report)
  ``<output_dir>/diffs/<path>.diff``   per-file unified diff vs. merge-base
  ``<output_dir>/base/<path>``         file content at merge-base (absent for newly added files)

``total`` is the number of code lines in the HEAD version of the file. ``added`` and ``removed`` are
the number of code lines added or removed by this PR (vs. merge-base). A "code line" excludes blank
lines and lines whose only non-whitespace content is a comment. C/C++ ``/* ... */`` block comments
are stripped before counting.

The agent consumes these inputs to produce the per-function breakdown.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PY_EXTS = {".py"}
CPP_EXTS = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu"}


def run_git(args: list[str], *, check: bool = True) -> str:
    """Run a git command and return stdout. Empty string on failure when ``check=False``."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if check:
            sys.stderr.write(f"git {' '.join(args)} failed:\n{result.stderr}")
            raise SystemExit(result.returncode)
        return ""
    return result.stdout


def language_for(path: str) -> str | None:
    suffix = Path(path).suffix
    if suffix in PY_EXTS:
        return "py"
    if suffix in CPP_EXTS:
        return "cpp"
    return None


def strip_cpp_comments(src: str) -> str:
    """Replace C/C++ comments with whitespace, preserving line numbering.

    Handles ``// ...`` line comments, ``/* ... */`` block comments (including multi-line), and
    leaves contents of string and char literals intact.
    """
    out: list[str] = []
    i = 0
    n = len(src)
    state = "code"  # one of: code, line_comment, block_comment, dq_string, sq_string
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                out.append("  ")
                i += 2
                state = "line_comment"
                continue
            if c == "/" and nxt == "*":
                out.append("  ")
                i += 2
                state = "block_comment"
                continue
            if c == '"':
                state = "dq_string"
                out.append(c)
                i += 1
                continue
            if c == "'":
                state = "sq_string"
                out.append(c)
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                state = "code"
                out.append(c)
            else:
                out.append(" ")
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and nxt == "/":
                out.append("  ")
                i += 2
                state = "code"
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        # dq_string / sq_string: preserve content; escape sequences pass through.
        out.append(c)
        if c == "\\" and i + 1 < n:
            out.append(src[i + 1])
            i += 2
            continue
        if state == "dq_string" and c == '"':
            state = "code"
        elif state == "sq_string" and c == "'":
            state = "code"
        i += 1
    return "".join(out)


def code_line_set(src: str, language: str) -> set[int]:
    """Return 1-indexed line numbers in ``src`` whose content is "code" (not blank, not a comment)."""
    if language == "cpp":
        src = strip_cpp_comments(src)
    code_lines: set[int] = set()
    for idx, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if language == "py" and stripped.startswith("#"):
            continue
        # After ``strip_cpp_comments`` line comments have been blanked, but defensively also skip
        # any residual ``//`` (e.g. a line that was already in a malformed state).
        if language == "cpp" and stripped.startswith("//"):
            continue
        code_lines.add(idx)
    return code_lines


HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def diff_added_removed(diff_text: str, head_codes: set[int], base_codes: set[int]) -> tuple[int, int]:
    """Count code-line additions and removals in a unified diff.

    For each ``+`` line, the line number in the new file must be in ``head_codes``; for each ``-``
    line, the line number in the old file must be in ``base_codes``. This way block comments are
    excluded correctly because the code-line sets were computed with full block-comment stripping.
    """
    added = 0
    removed = 0
    new_no = 0
    old_no = 0
    in_hunk = False
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            m = HUNK_RE.match(line)
            if not m:
                in_hunk = False
                continue
            old_no = int(m.group(1))
            new_no = int(m.group(3))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            if new_no in head_codes:
                added += 1
            new_no += 1
        elif line.startswith("-"):
            if old_no in base_codes:
                removed += 1
            old_no += 1
        elif line.startswith(" "):
            new_no += 1
            old_no += 1
        elif line.startswith("\\"):
            continue
    return added, removed


@dataclass
class FileSummary:
    path: str
    language: str
    total: int
    added: int
    removed: int


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def format_header(s: FileSummary) -> str:
    parts = [s.path, str(s.total)]
    if s.added or not s.removed:
        parts.append(f"+{s.added}")
    if s.removed:
        parts.append(f"-{s.removed}")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="PR base ref name (e.g. main)")
    parser.add_argument("--head-ref", default="HEAD", help="PR head ref / sha (default: HEAD)")
    parser.add_argument("--output-dir", required=True, help="Where to write summary.json + diffs/ + base/")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    diffs_dir = output_dir / "diffs"
    base_dir = output_dir / "base"
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_base = run_git(["merge-base", f"origin/{args.base_ref}", args.head_ref]).strip()
    if not merge_base:
        sys.stderr.write(f"could not determine merge-base with origin/{args.base_ref}\n")
        return 1

    name_status = run_git(["diff", "--name-status", f"{merge_base}...{args.head_ref}"])

    summaries: list[FileSummary] = []
    for line in name_status.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("D"):
            continue
        # For renames / copies the new path is the last field; for plain modifications it is the only path.
        path = parts[-1]
        lang = language_for(path)
        if lang is None:
            continue

        diff_text = run_git(["diff", f"{merge_base}...{args.head_ref}", "--", path])
        if not diff_text.strip():
            continue

        head_content = run_git(["show", f"{args.head_ref}:{path}"], check=False)
        base_content = run_git(["show", f"{merge_base}:{path}"], check=False)

        head_codes = code_line_set(head_content, lang)
        base_codes = code_line_set(base_content, lang) if base_content else set()
        added, removed = diff_added_removed(diff_text, head_codes, base_codes)

        write_file(diffs_dir / f"{path}.diff", diff_text)
        if base_content:
            write_file(base_dir / path, base_content)

        summaries.append(FileSummary(path=path, language=lang, total=len(head_codes), added=added, removed=removed))

    summary_dicts = [
        {"path": s.path, "language": s.language, "total": s.total, "added": s.added, "removed": s.removed}
        for s in summaries
    ]
    (output_dir / "summary.json").write_text(json.dumps(summary_dicts, indent=2) + "\n")

    file_list = "\n".join(s.path for s in summaries)
    (output_dir / "file_list.txt").write_text(file_list + ("\n" if file_list else ""))

    headers = "\n".join(format_header(s) for s in summaries)
    (output_dir / "report_header.md").write_text(headers + ("\n" if headers else ""))

    print(f"Wrote summaries for {len(summaries)} file(s) to {output_dir}")
    if summaries:
        print()
        print(headers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
