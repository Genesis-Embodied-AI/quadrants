#!/usr/bin/env python3
"""Build inputs for the PR change report agent.

For each source file changed in the PR (.py / .c / .cc / .cpp / .h / .hpp / .cu) this writes:

  ``<output_dir>/summary.json``        list of ``{path, language, total, added, removed, is_deleted}``
  ``<output_dir>/file_list.txt``       one path per line; SAME order as ``summary.json`` but EXCLUDES
                                       deleted files (they have no HEAD content for the agent to attribute)
  ``<output_dir>/report_header.md``    pre-formatted file-header lines (verbatim for the report)
  ``<output_dir>/report_comment.md``   compact PR-comment markdown (table + totals line)
  ``<output_dir>/diffs/<path>.diff``   per-file unified diff vs. merge-base
  ``<output_dir>/head/<path>``         HEAD content of the file (always present)
  ``<output_dir>/base/<path>``         file content at merge-base (absent for newly added files)

``total`` is the number of code lines in the BASE (pre-PR, merge-base) version of the file, i.e.
the size of the file before this PR's changes. For newly-added files this is 0. ``added`` and
``removed`` are the number of code lines added or removed by this PR (vs. merge-base). A "code
line" excludes blank lines, lines whose only non-whitespace content is a comment, and (in Python)
lines whose only token content is a string literal (i.e. docstrings and continuation lines of
multi-line strings). C/C++ ``/* ... */`` block comments are stripped before counting.

The agent consumes these inputs to produce the per-function breakdown.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tokenize
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


# Python tokens that DO NOT count as code on their own: comments, string literals (incl. docstrings
# and continuation lines of multi-line strings), f-string string-portion tokens, the encoding
# declaration, and structural newlines / indents that produce no executable text.
_PY_NON_CODE_TOKEN_TYPES: set[int] = {
    tokenize.COMMENT,
    tokenize.STRING,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.ENCODING,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
}
for _name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
    if hasattr(tokenize, _name):
        _PY_NON_CODE_TOKEN_TYPES.add(getattr(tokenize, _name))


def _python_code_line_set(src: str) -> set[int]:
    """Lines of ``src`` (1-indexed) that contain at least one non-string, non-comment token.

    Multi-line strings, docstrings, and continuation lines of multi-line strings are excluded.
    A line that contains a string literal alongside real code (e.g. ``x = "foo"``) is still code,
    because the ``=`` and ``x`` are non-excluded tokens on that line.
    """
    code_lines: set[int] = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        # Malformed or partial source: fall back to a coarse heuristic that treats any non-blank
        # non-``#`` line as code. Multi-line string lines may be over-counted in this fallback.
        for idx, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code_lines.add(idx)
        return code_lines

    for tok in tokens:
        if tok.type in _PY_NON_CODE_TOKEN_TYPES:
            continue
        for line_no in range(tok.start[0], tok.end[0] + 1):
            code_lines.add(line_no)
    return code_lines


def _cpp_code_line_set(src: str) -> set[int]:
    """Lines of ``src`` (1-indexed) that contain at least one non-comment, non-blank character.

    String literal contents are kept (a string literal is a code expression). ``// ...`` line
    comments and ``/* ... */`` block comments are stripped first, so a line that holds only a
    block comment becomes blank and is excluded.
    """
    src = strip_cpp_comments(src)
    code_lines: set[int] = set()
    for idx, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        # Defensive: if ``strip_cpp_comments`` left a ``//`` (malformed source edge case), skip it.
        if stripped.startswith("//"):
            continue
        code_lines.add(idx)
    return code_lines


def code_line_set(src: str, language: str) -> set[int]:
    """Return 1-indexed line numbers in ``src`` that contain code (not blank, not comment-only,
    and -- for Python -- not string-literal-only)."""
    if not src:
        return set()
    if language == "py":
        return _python_code_line_set(src)
    if language == "cpp":
        return _cpp_code_line_set(src)
    return set()


HUNK_RE = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def diff_changed_line_numbers(diff_text: str) -> tuple[list[int], list[int]]:
    """Walk a unified diff and return ``(added_new_line_nos, removed_old_line_nos)``.

    Each list contains the 1-indexed line number that each ``+`` (or ``-``) line maps to in the
    new (or old) file -- one entry per diff line. Hunk headers, file headers and the
    ``\\ No newline at end of file`` marker are skipped.
    """
    added: list[int] = []
    removed: list[int] = []
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
            added.append(new_no)
            new_no += 1
        elif line.startswith("-"):
            removed.append(old_no)
            old_no += 1
        elif line.startswith(" "):
            new_no += 1
            old_no += 1
        elif line.startswith("\\"):
            continue
    return added, removed


def diff_added_removed(diff_text: str, head_codes: set[int], base_codes: set[int]) -> tuple[int, int]:
    """Count code-line additions and removals in a unified diff.

    A ``+`` line is counted if its new-file line number is in ``head_codes``; a ``-`` line is
    counted if its old-file line number is in ``base_codes``. This excludes block-comment
    content correctly because the code-line sets were computed with full comment stripping.
    """
    added_lines, removed_lines = diff_changed_line_numbers(diff_text)
    added = sum(1 for ln in added_lines if ln in head_codes)
    removed = sum(1 for ln in removed_lines if ln in base_codes)
    return added, removed


def diff_touched_ranges(diff_text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return the (head_ranges, base_ranges) covered by hunks of ``diff_text``.

    Each range is ``(start_line, end_line)`` 1-indexed inclusive. The head ranges cover the
    new-file line numbers spanned by each hunk's ``+``/context lines; base ranges cover the
    old-file line numbers spanned by ``-``/context lines. Useful as a hint to the agent about
    where to look for changed functions without scanning the whole file.
    """
    head_ranges: list[tuple[int, int]] = []
    base_ranges: list[tuple[int, int]] = []
    new_no = 0
    old_no = 0
    head_start = base_start = None
    head_last = base_last = None
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            if head_start is not None:
                head_ranges.append((head_start, head_last))
            if base_start is not None:
                base_ranges.append((base_start, base_last))
            m = HUNK_RE.match(line)
            if not m:
                head_start = base_start = head_last = base_last = None
                continue
            old_no = int(m.group(1))
            new_no = int(m.group(3))
            head_start = new_no
            base_start = old_no
            head_last = new_no - 1
            base_last = old_no - 1
            continue
        if line.startswith("+"):
            head_last = new_no
            new_no += 1
        elif line.startswith("-"):
            base_last = old_no
            old_no += 1
        elif line.startswith(" "):
            head_last = new_no
            base_last = old_no
            new_no += 1
            old_no += 1
        elif line.startswith("\\"):
            continue
    if head_start is not None and head_last >= head_start:
        head_ranges.append((head_start, head_last))
    if base_start is not None and base_last >= base_start:
        base_ranges.append((base_start, base_last))
    return head_ranges, base_ranges


@dataclass
class FileSummary:
    path: str
    language: str
    total: int
    added: int
    removed: int
    is_deleted: bool = False


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


def render_comment_markdown(summaries: list[FileSummary], commit: str) -> str:
    """Render the compact PR-comment markdown: heading + per-file table + totals line.

    The comment intentionally does NOT include the per-function breakdown; that lives in the
    Check page (which the comment links to).
    """
    head = f"## PR change report (`{commit}`)\n" if commit else "## PR change report\n"
    lines = [head]
    if not summaries:
        lines.append("No source files (.py, .c, .cc, .cpp, .h, .hpp, .cu) changed in this PR.")
        return "\n".join(lines) + "\n"
    lines.append(
        "LoC = code lines in the file BEFORE this PR (0 for newly-added files). "
        "Excludes blank lines, comment-only lines, and Python multi-line strings."
    )
    lines.append("")
    lines.append("| File | LoC | Added | Removed |")
    lines.append("|------|-----|-------|---------|")
    total_added = 0
    total_removed = 0
    for s in summaries:
        added_cell = f"+{s.added}" if s.added else ""
        removed_cell = f"-{s.removed}" if s.removed else ""
        lines.append(f"| `{s.path}` | {s.total} | {added_cell} | {removed_cell} |")
        total_added += s.added
        total_removed += s.removed
    lines.append("")
    lines.append(f"**Total**: {len(summaries)} file(s) changed, +{total_added} -{total_removed} code lines.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True, help="PR base ref name (e.g. main)")
    parser.add_argument("--head-ref", default="HEAD", help="PR head ref / sha (default: HEAD)")
    parser.add_argument("--output-dir", required=True, help="Where to write summary.json + diffs/ + base/")
    parser.add_argument("--commit-hash", default="", help="Short commit hash to embed in the comment heading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    diffs_dir = output_dir / "diffs"
    head_dir = output_dir / "head"
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
        is_deleted = status.startswith("D")
        # For renames / copies the new path is the last field; for plain modifications it is the only path.
        path = parts[-1]
        lang = language_for(path)
        if lang is None:
            continue

        diff_text = run_git(["diff", f"{merge_base}...{args.head_ref}", "--", path])
        if not diff_text.strip():
            continue

        if is_deleted:
            head_content = ""
            base_content = run_git(["show", f"{merge_base}:{path}"], check=False)
            head_codes: set[int] = set()
            base_codes = code_line_set(base_content, lang) if base_content else set()
            added = 0
            removed = len(base_codes)
        else:
            head_content = run_git(["show", f"{args.head_ref}:{path}"], check=False)
            base_content = run_git(["show", f"{merge_base}:{path}"], check=False)
            head_codes = code_line_set(head_content, lang)
            base_codes = code_line_set(base_content, lang) if base_content else set()
            added, removed = diff_added_removed(diff_text, head_codes, base_codes)

        write_file(diffs_dir / f"{path}.diff", diff_text)
        if head_content:
            write_file(head_dir / path, head_content)
        if base_content:
            write_file(base_dir / path, base_content)

        summaries.append(
            FileSummary(
                path=path, language=lang, total=len(base_codes), added=added, removed=removed, is_deleted=is_deleted
            )
        )

    # Sort files by descending lines added, then descending lines removed, then path. This is the
    # order used for every downstream artifact (summary.json, file_list.txt, report_header.md,
    # report_comment.md) so that the report and PR comment surface the largest-impact files first.
    summaries.sort(key=lambda s: (-s.added, -s.removed, s.path))

    summary_dicts = [
        {
            "path": s.path,
            "language": s.language,
            "total": s.total,
            "added": s.added,
            "removed": s.removed,
            "is_deleted": s.is_deleted,
        }
        for s in summaries
    ]
    (output_dir / "summary.json").write_text(json.dumps(summary_dicts, indent=2) + "\n")

    # file_list.txt and touched_ranges.txt feed the agent. Deleted files have no HEAD content
    # and no per-function attribution to compute, so they are excluded from both. They still
    # appear in summary.json / report_comment.md / report.txt with correct totals.
    file_list = "\n".join(s.path for s in summaries if not s.is_deleted)
    (output_dir / "file_list.txt").write_text(file_list + ("\n" if file_list else ""))

    headers = "\n".join(format_header(s) for s in summaries)
    (output_dir / "report_header.md").write_text(headers + ("\n" if headers else ""))

    (output_dir / "report_comment.md").write_text(render_comment_markdown(summaries, args.commit_hash))

    touched_lines = []
    for s in summaries:
        if s.is_deleted:
            continue
        diff_text = (diffs_dir / f"{s.path}.diff").read_text()
        head_ranges, base_ranges = diff_touched_ranges(diff_text)
        head_str = ", ".join(f"{a}-{b}" for a, b in head_ranges) or "(none)"
        base_str = ", ".join(f"{a}-{b}" for a, b in base_ranges) or "(none)"
        touched_lines.append(f"{s.path}\n  head hunks: {head_str}\n  base hunks: {base_str}")
    (output_dir / "touched_ranges.txt").write_text("\n".join(touched_lines) + ("\n" if touched_lines else ""))

    print(f"Wrote summaries for {len(summaries)} file(s) to {output_dir}")
    if summaries:
        print()
        print(headers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
