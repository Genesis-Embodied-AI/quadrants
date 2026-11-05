#!/usr/bin/env python3
"""Render all PR change report artifacts from prepared inputs and identified function ranges.

Phase 1 is ``build_inputs.py``; phase 2 (this script) consumes those outputs plus the agent's JSONL output. The agent
supplies function names and line ranges; counts are computed deterministically from the diffs and HEAD/base contents
using the same counting rules as ``build_inputs.py``.

JSONL input (one object per line) -- emitted by the agent::

  {"path": "<path>", "name": "<func>", "head": [<start>, <end>] | null, "base": [<start>, <end>] | null}

Output ``report_comment.md`` contains group and file tables; ``report_header.md`` contains compact file headers.
``report_check.md`` combines the summary and detailed report for the workflow summary page.

Output ``report.md`` contains folder-group headings, file totals, and per-function tables rendered by the shared
``render_table`` helper. Within each file, functions are split into New, Existing, and Deleted sections and sorted by
added lines descending, then removed lines descending.

The LoC column is the BASE (pre-PR) counted-line total: file size before the PR or function body size before the PR.
Newly added files and functions have LoC 0; deleted files and functions retain their original size. Added and Removed
columns contain nonnegative counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow importing helper functions from ``build_inputs.py`` (in the same directory).
sys.path.insert(0, str(Path(__file__).parent))
from build_inputs import (  # noqa: E402  (import after sys.path tweak)
    FileSummary,
    code_line_set,
    diff_changed_line_numbers,
)


COUNTING_NOTE = (
    "Counts exclude blank lines. Python/C++ exclude comment-only lines; Python also excludes "
    "string-literal-only lines. Markdown counts all nonblank lines. Shell/YAML skip lines starting "
    "with # after whitespace (shell shebangs count); this heuristic also skips #-prefixed data "
    "inside multiline strings, YAML block scalars, and shell heredocs."
)


def format_header(s: FileSummary) -> str:
    parts = [s.path, str(s.total)]
    if s.added or not s.removed:
        parts.append(f"+{s.added}")
    if s.removed:
        parts.append(f"-{s.removed}")
    return " ".join(parts)


MISC_GROUP_NAME = "misc"
GROUPS = ("docs", "c++", "python", "tests", "github", MISC_GROUP_NAME)
_GROUP_BY_FOLDER_NAME = {
    "docs": "docs",
    "quadrants": "c++",
    "cmake": "c++",
    "python": "python",
    "tests": "tests",
    ".github": "github",
}


def get_group_summaries_by_name(summaries: list[FileSummary]) -> dict[str, list[FileSummary]]:
    """Group by top-level folder, keeping the existing impact order within each group."""
    summaries_by_group_name: dict[str, list[FileSummary]] = {name: [] for name in GROUPS}
    for summary in summaries:
        folder, separator, _ = summary.path.partition("/")
        group = _GROUP_BY_FOLDER_NAME.get(folder, MISC_GROUP_NAME) if separator else MISC_GROUP_NAME
        summaries_by_group_name[group].append(summary)
    return summaries_by_group_name


def render_table(columns: list[str], rows: list[dict[str, str | int | float]]) -> list[str]:
    """Render dictionary rows as a Markdown table in the specified column order."""
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]).replace("|", r"\|") for column in columns) + " |")
    return lines


def render_comment_markdown(summaries: list[FileSummary], commit: str) -> str:
    """Render group totals and grouped per-file tables for the report summary."""
    head = f"## PR change report (`{commit}`)\n" if commit else "## PR change report\n"
    lines = [head]
    if not summaries:
        lines.append(
            "No supported files (.py, .c, .cc, .cpp, .h, .hpp, .cu, .md, .sh, .yaml, .yml) changed in this PR."
        )
        return "\n".join(lines) + "\n"
    lines.append("LoC = counted lines in the file BEFORE this PR (0 for newly-added files). " + COUNTING_NOTE)
    lines.append("")
    group_summaries_by_name = get_group_summaries_by_name(summaries)
    group_rows: list[dict[str, str | int | float]] = []
    for group_name, group_files in group_summaries_by_name.items():
        group_rows.append(
            {
                "Group": group_name,
                "Files": len(group_files),
                "Added": sum(s.added for s in group_files),
                "Removed": sum(s.removed for s in group_files),
            }
        )
    total_added = sum(s.added for s in summaries)
    total_removed = sum(s.removed for s in summaries)
    group_rows.append(
        {
            "Group": "**Total**",
            "Files": len(summaries),
            "Added": total_added,
            "Removed": total_removed,
        }
    )
    lines.extend(render_table(["Group", "Files", "Added", "Removed"], group_rows))

    for group_name, group_files in group_summaries_by_name.items():
        if not group_files:
            continue
        lines.append("")
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append(
            f"**Group total**: {len(group_files)} file(s) changed, "
            f"+{sum(s.added for s in group_files)} -{sum(s.removed for s in group_files)} counted lines."
        )
        lines.append("")
        file_rows: list[dict[str, str | int | float]] = []
        for s in group_files:
            file_rows.append(
                {
                    "File": f"`{s.path}`",
                    "LoC": s.total,
                    "Added": s.added,
                    "Removed": s.removed,
                }
            )
        lines.extend(render_table(["File", "LoC", "Added", "Removed"], file_rows))
    lines.append("")
    lines.append(f"**Total**: {len(summaries)} file(s) changed, +{total_added} -{total_removed} counted lines.")
    return "\n".join(lines) + "\n"


@dataclass
class _FunctionEntry:
    name: str
    head_range: tuple[int, int] | None
    base_range: tuple[int, int] | None


@dataclass
class _FileBookkeeping:
    path: str
    language: str
    total: int
    added: int
    removed: int
    head_codes: set[int]
    base_codes: set[int]
    added_line_nos: list[int]
    removed_line_nos: list[int]
    is_deleted: bool = False


def _load_function_entries(jsonl_path: Path) -> dict[str, list[_FunctionEntry]]:
    """Parse the agent's JSONL and group entries by file path. Malformed lines are skipped.

    Identical ``(path, name, head, base)`` tuples are deduplicated -- some agents (e.g.
    composer-2) occasionally emit the entire JSONL stream twice, which would otherwise
    cause the coalesce-by-name step to double the per-function counts.
    """
    by_path: dict[str, list[_FunctionEntry]] = {}
    seen: set[tuple] = set()
    duplicates = 0
    if not jsonl_path.exists():
        return by_path
    for raw in jsonl_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            sys.stderr.write(f"warning: skipping malformed JSONL line: {raw!r}\n")
            continue
        path = obj.get("path")
        name = obj.get("name")
        if not isinstance(path, str) or not isinstance(name, str):
            sys.stderr.write(f"warning: skipping JSONL entry missing path/name: {obj!r}\n")
            continue
        head_range = _coerce_range(obj.get("head"))
        base_range = _coerce_range(obj.get("base"))
        if head_range is None and base_range is None:
            sys.stderr.write(f"warning: skipping JSONL entry with no head/base range: {obj!r}\n")
            continue
        key = (path, name, head_range, base_range)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        by_path.setdefault(path, []).append(_FunctionEntry(name=name, head_range=head_range, base_range=base_range))
    if duplicates:
        sys.stderr.write(f"info: deduplicated {duplicates} identical JSONL entry/entries\n")
    return by_path


def _coerce_range(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
        and value[0] <= value[1]
    ):
        return (int(value[0]), int(value[1]))
    return None


def _read_or_empty(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _build_file_bookkeeping(summary: dict, output_dir: Path) -> _FileBookkeeping:
    path = summary["path"]
    language = summary["language"]
    head_text = _read_or_empty(output_dir / "head" / path)
    base_text = _read_or_empty(output_dir / "base" / path)
    diff_text = _read_or_empty(output_dir / "diffs" / f"{path}.diff")
    head_codes = code_line_set(head_text, language)
    base_codes = code_line_set(base_text, language)
    added_line_nos, removed_line_nos = diff_changed_line_numbers(diff_text)
    return _FileBookkeeping(
        path=path,
        language=language,
        total=summary["total"],
        added=summary["added"],
        removed=summary["removed"],
        head_codes=head_codes,
        base_codes=base_codes,
        added_line_nos=added_line_nos,
        removed_line_nos=removed_line_nos,
        is_deleted=bool(summary.get("is_deleted", False)),
    )


def _attribute_function(entry: _FunctionEntry, book: _FileBookkeeping) -> tuple[int, int, int, bool, bool]:
    """Return ``(total, added, removed, is_new, is_deleted)`` for a single function entry.

    ``total`` is the number of code lines in the function's BASE (pre-PR) body, i.e. how big
    this function was before this PR. For NEW functions (no base body) this is 0. ``added`` /
    ``removed`` are the number of ``+`` / ``-`` diff lines that fall inside the function's
    HEAD / base range AND are themselves code lines (excluding comments / blank lines / Python
    multi-line strings).
    """
    total = 0
    added = 0
    removed = 0
    if entry.base_range is not None:
        c, d = entry.base_range
        total = sum(1 for ln in book.base_codes if c <= ln <= d)
        removed = sum(1 for ln in book.removed_line_nos if c <= ln <= d and ln in book.base_codes)
    if entry.head_range is not None:
        a, b = entry.head_range
        added = sum(1 for ln in book.added_line_nos if a <= ln <= b and ln in book.head_codes)
    is_new = entry.head_range is not None and entry.base_range is None
    is_deleted = entry.head_range is None and entry.base_range is not None
    return total, added, removed, is_new, is_deleted


@dataclass
class _AttributedEntry:
    """A ``_FunctionEntry`` plus its computed total / added / removed counts."""

    entry: _FunctionEntry
    total: int
    added: int
    removed: int
    is_new: bool
    is_deleted: bool


def _attribute_all(entries: list[_FunctionEntry], book: _FileBookkeeping) -> list[_AttributedEntry]:
    out: list[_AttributedEntry] = []
    for entry in entries:
        total, added, removed, is_new, is_deleted = _attribute_function(entry, book)
        if added == 0 and removed == 0:
            continue
        out.append(_AttributedEntry(entry, total, added, removed, is_new, is_deleted))
    return _coalesce_by_name(out)


def _coalesce_by_name(attributed: list[_AttributedEntry]) -> list[_AttributedEntry]:
    """Merge attributed entries that share the same ``name`` by summing counts.

    The agent legitimately emits multiple ``<module>`` entries when module-level changes
    are scattered between function definitions; collapsing them into one row per file is
    cleaner than showing the same name several times. Counts are summed (assumes the
    agent's ranges within a file are disjoint, per the prompt's overlap rule).
    """
    by_name: dict[str, _AttributedEntry] = {}
    order: list[str] = []
    for a in attributed:
        existing = by_name.get(a.entry.name)
        if existing is None:
            by_name[a.entry.name] = a
            order.append(a.entry.name)
            continue
        by_name[a.entry.name] = _AttributedEntry(
            entry=existing.entry,
            total=existing.total + a.total,
            added=existing.added + a.added,
            removed=existing.removed + a.removed,
            is_new=existing.is_new and a.is_new,
            is_deleted=existing.is_deleted and a.is_deleted,
        )
    return [by_name[n] for n in order]


def _impact_sort_key(a: _AttributedEntry) -> tuple[int, int, str]:
    """Sort by added desc, then removed desc, then name asc (stable across runs)."""
    return (-a.added, -a.removed, a.entry.name)


def _function_label(name: str) -> str:
    """Append ``()`` only when the agent did not already supply a parenthesized signature.

    For overloads like ``Foo::visit(RangeForStmt*)`` the agent's parameter list is the
    disambiguating part; we keep it verbatim instead of producing ``Foo::visit(...)()``.
    """
    return name if name.rstrip().endswith(")") else f"{name}()"


def _emit_file_section(book: _FileBookkeeping, attributed: list[_AttributedEntry]) -> tuple[list[str], int, int]:
    """Render file totals and grouped function tables; return lines and per-function addition/removal totals."""
    lines = [f"#### `{book.path}`", ""]
    lines.extend(
        render_table(["LoC", "Added", "Removed"], [{"LoC": book.total, "Added": book.added, "Removed": book.removed}])
    )
    if book.is_deleted:
        lines.append("")
        lines.append("Entire file deleted (per-function breakdown skipped).")
        return lines, 0, 0
    if book.language not in {"py", "cpp"}:
        return lines, 0, 0
    if not attributed:
        if book.added or book.removed:
            lines.append("")
            lines.append("Note: no per-function attribution available.")
        return lines, 0, 0

    new_group = sorted([a for a in attributed if a.is_new], key=_impact_sort_key)
    deleted_group = sorted([a for a in attributed if a.is_deleted], key=_impact_sort_key)
    existing_group = sorted([a for a in attributed if not a.is_new and not a.is_deleted], key=_impact_sort_key)

    for group_name, group_functions in [("New", new_group), ("Existing", existing_group), ("Deleted", deleted_group)]:
        if not group_functions:
            continue
        lines.append("")
        lines.append(f"##### {group_name}")
        lines.append("")
        rows: list[dict[str, str | int | float]] = []
        for function in group_functions:
            rows.append(
                {
                    "Function": f"`{_function_label(function.entry.name)}`",
                    "LoC": function.total,
                    "Added": function.added,
                    "Removed": function.removed,
                }
            )
        lines.extend(render_table(["Function", "LoC", "Added", "Removed"], rows))

    per_added = sum(a.added for a in attributed)
    per_removed = sum(a.removed for a in attributed)
    added_drift = book.added - per_added
    removed_drift = book.removed - per_removed
    if abs(added_drift) > 5 or abs(removed_drift) > 5:
        lines.append("")
        lines.append(
            "Note: per-function counts differ from file totals by "
            f"added_drift={added_drift:+d} removed_drift={removed_drift:+d}"
        )
    return lines, per_added, per_removed


_FOOTER = (
    "### Notes\n\n"
    "- LoC counts lines in the BASE (pre-PR) version: file size or function body size before this PR "
    "(0 for new files/functions; original size for deleted files/functions).\n"
    "- Added and Removed are nonnegative counts of lines added and removed by this PR.\n"
    f"- {COUNTING_NOTE}"
)


def render(summary_json: Path, jsonl_path: Path, output_dir: Path) -> str:
    summaries: list[dict] = json.loads(summary_json.read_text())
    by_path = _load_function_entries(jsonl_path)

    out_lines: list[str] = []
    group_summaries_by_name = get_group_summaries_by_name([FileSummary(**s) for s in summaries])
    summaries_by_path = {s["path"]: s for s in summaries}
    total_added = 0
    total_removed = 0
    for group_name, group_files in group_summaries_by_name.items():
        if not group_files:
            continue
        added = sum(s.added for s in group_files)
        removed = sum(s.removed for s in group_files)
        total_added += added
        total_removed += removed
        out_lines.append(f"### {group_name}")
        out_lines.append("")
        out_lines.append(f"{len(group_files)} file(s) changed, +{added} -{removed} counted lines.")
        out_lines.append("")
        for s in group_files:
            book = _build_file_bookkeeping(summaries_by_path[s.path], output_dir)
            attributed = _attribute_all(by_path.get(book.path, []), book)
            lines, _, _ = _emit_file_section(book, attributed)
            out_lines.extend(lines)
            out_lines.append("")
    if summaries:
        out_lines.append(f"Total: {len(summaries)} file(s) changed, +{total_added} -{total_removed} counted lines.")
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    if out_lines:
        out_lines.append("")
        out_lines.append(_FOOTER)
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Phase-1 output directory")
    parser.add_argument("--function-ranges", required=True, help="Path to JSONL of function ranges")
    parser.add_argument("--output", required=True, help="Where to write report.md")
    parser.add_argument("--commit-hash", default="", help="Short commit hash to embed in the report heading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary_json = output_dir / "summary.json"
    summaries = [FileSummary(**s) for s in json.loads(summary_json.read_text())]
    headers = "\n".join(format_header(s) for s in summaries)
    (output_dir / "report_header.md").write_text(headers + ("\n" if headers else ""))
    summary_markdown = render_comment_markdown(summaries, args.commit_hash)
    (output_dir / "report_comment.md").write_text(summary_markdown)
    text = render(
        summary_json=summary_json,
        jsonl_path=Path(args.function_ranges),
        output_dir=output_dir,
    )
    Path(args.output).write_text(text)
    (output_dir / "report_check.md").write_text(
        summary_markdown + "\n## Per-function breakdown (Python/C++ only)\n\n" + text
    )
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
