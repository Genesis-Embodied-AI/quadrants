#!/usr/bin/env python3
"""Render the per-file / per-function PR change report from phase-1 outputs and the agent's
identified function ranges.

Phase 1 is ``build_inputs.py``; phase 2 (this script) consumes those outputs plus the agent's
JSONL output and produces the human-readable report. Everything here is deterministic -- the
agent only contributes function NAMES and LINE RANGES; line counts and the +/- attribution are
computed from the diff and the HEAD/base file contents using the same code-line definition
``build_inputs.py`` uses.

JSONL input (one object per line) -- emitted by the agent::

  {"path": "<path>", "name": "<func>", "head": [<start>, <end>] | null, "base": [<start>, <end>] | null}

Output ``report.txt`` shape::

  <path> <total> +<added> -<removed>
      New:
        <func>()                              <total>  +<added>
        ...
      Existing:
        <func>()                              <total>  +<added>  -<removed>
        ...
      Deleted:
        <func>()                              <total>            -<removed>
        ...

The ``<total>`` column at every level is the BASE (pre-PR) code-line count: file size before
the PR (or 0 for newly-added files); function body size before the PR (or 0 for new functions;
the original body size for deleted functions). Within each file, functions are split into a
``New:`` group (added by this PR), an ``Existing:`` group (modified by this PR), and a
``Deleted:`` group (removed by this PR), and within each group sorted by added lines descending,
then removed lines descending. Function-name and numeric columns are padded with spaces so the
columns line up within a file.
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
    code_line_set,
    diff_changed_line_numbers,
    format_header,
)


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


# Column widths chosen to fit the longest expected token in each column without crowding adjacent
# columns. Five-digit ``+``/``-`` counts are the longest expected numeric tokens; the total
# column holds a base (pre-PR) line count so it never exceeds a few digits in practice.
_TOTAL_WIDTH = 6
_ADDED_WIDTH = 7
_REMOVED_WIDTH = 7
_FUNC_INDENT = "      "
_GROUP_INDENT = "    "


def _format_total(a: _AttributedEntry) -> str:
    return str(a.total)


def _function_label(name: str) -> str:
    """Append ``()`` only when the agent did not already supply a parenthesized signature.

    For overloads like ``Foo::visit(RangeForStmt*)`` the agent's parameter list is the
    disambiguating part; we keep it verbatim instead of producing ``Foo::visit(...)()``.
    """
    return name if name.rstrip().endswith(")") else f"{name}()"


def _format_aligned_function(a: _AttributedEntry, name_width: int) -> str:
    name = _function_label(a.entry.name)
    total_str = _format_total(a)
    added_str = f"+{a.added}" if a.added else ""
    removed_str = f"-{a.removed}" if a.removed else ""
    return (
        f"{_FUNC_INDENT}{name:<{name_width}}  "
        f"{total_str:>{_TOTAL_WIDTH}}  "
        f"{added_str:>{_ADDED_WIDTH}}  "
        f"{removed_str:>{_REMOVED_WIDTH}}"
    ).rstrip()


def _emit_file_section(book: _FileBookkeeping, attributed: list[_AttributedEntry]) -> tuple[list[str], int, int]:
    """Render a single file's section (header + grouped/sorted/aligned function rows).

    Returns ``(lines, per_func_added_total, per_func_removed_total)``. Trailing blank line is
    appended by the caller, not here.
    """
    lines: list[str] = [format_header(_HeaderProxy(book))]
    if book.is_deleted:
        lines.append(f"{_GROUP_INDENT}# entire file deleted (per-function breakdown skipped)")
        return lines, 0, 0
    if not attributed:
        if book.added or book.removed:
            lines.append(f"{_GROUP_INDENT}# note: no per-function attribution available")
        return lines, 0, 0

    new_group = sorted([a for a in attributed if a.is_new], key=_impact_sort_key)
    deleted_group = sorted([a for a in attributed if a.is_deleted], key=_impact_sort_key)
    existing_group = sorted([a for a in attributed if not a.is_new and not a.is_deleted], key=_impact_sort_key)

    # Pad function names to the longest name across ALL groups so the New: / Existing: /
    # Deleted: subsections line up with each other.
    all_names = [_function_label(a.entry.name) for a in new_group + existing_group + deleted_group]
    name_width = max((len(n) for n in all_names), default=0)

    if new_group:
        lines.append(f"{_GROUP_INDENT}New:")
        for a in new_group:
            lines.append(_format_aligned_function(a, name_width))
    if existing_group:
        lines.append(f"{_GROUP_INDENT}Existing:")
        for a in existing_group:
            lines.append(_format_aligned_function(a, name_width))
    if deleted_group:
        lines.append(f"{_GROUP_INDENT}Deleted:")
        for a in deleted_group:
            lines.append(_format_aligned_function(a, name_width))

    per_added = sum(a.added for a in attributed)
    per_removed = sum(a.removed for a in attributed)
    added_drift = book.added - per_added
    removed_drift = book.removed - per_removed
    if abs(added_drift) > 5 or abs(removed_drift) > 5:
        lines.append(
            f"{_GROUP_INDENT}# note: per-function +/- differs from file totals by "
            f"added_drift={added_drift:+d} removed_drift={removed_drift:+d}"
        )
    return lines, per_added, per_removed


_FOOTER = (
    "Notes:\n"
    "  * The number columns (without a + or - sign) are code-line counts in the "
    "BASE (pre-PR) version: file size before this PR (0 for newly-added files), "
    "function body size before this PR (0 for new functions; original body size "
    "for deleted functions).\n"
    "  * +<n> / -<n> are code lines added / removed by this PR.\n"
    "  * Code lines exclude blank lines, comment-only lines, and Python "
    "multi-line strings."
)


def render(summary_json: Path, jsonl_path: Path, output_dir: Path) -> str:
    summaries: list[dict] = json.loads(summary_json.read_text())
    by_path = _load_function_entries(jsonl_path)

    out_lines: list[str] = []
    for s in summaries:
        book = _build_file_bookkeeping(s, output_dir)
        attributed = _attribute_all(by_path.get(book.path, []), book)
        lines, _, _ = _emit_file_section(book, attributed)
        out_lines.extend(lines)
        out_lines.append("")
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
    if out_lines:
        out_lines.append("")
        out_lines.append(_FOOTER)
    return "\n".join(out_lines) + ("\n" if out_lines else "")


class _HeaderProxy:
    """Adapter so ``format_header`` (which expects a ``FileSummary``-shaped object) can consume
    a ``_FileBookkeeping`` directly."""

    def __init__(self, book: _FileBookkeeping) -> None:
        self.path = book.path
        self.total = book.total
        self.added = book.added
        self.removed = book.removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Phase-1 output directory")
    parser.add_argument("--function-ranges", required=True, help="Path to JSONL of function ranges")
    parser.add_argument("--output", required=True, help="Where to write report.txt")
    args = parser.parse_args()

    text = render(
        summary_json=Path(args.output_dir) / "summary.json",
        jsonl_path=Path(args.function_ranges),
        output_dir=Path(args.output_dir),
    )
    Path(args.output).write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
