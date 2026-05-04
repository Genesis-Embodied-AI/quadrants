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
      <func>() NEW +<added>
      <func>() <total> +<added> -<removed>
      <func>() DELETED -<removed>
      ...
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
    language_for,
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


def _load_function_entries(jsonl_path: Path) -> dict[str, list[_FunctionEntry]]:
    """Parse the agent's JSONL and group entries by file path. Malformed lines are skipped."""
    by_path: dict[str, list[_FunctionEntry]] = {}
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
        by_path.setdefault(path, []).append(
            _FunctionEntry(name=name, head_range=head_range, base_range=base_range)
        )
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


def _build_file_bookkeeping(
    summary: dict, output_dir: Path
) -> _FileBookkeeping:
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
    )


def _attribute_function(
    entry: _FunctionEntry, book: _FileBookkeeping
) -> tuple[int, int, int, bool, bool]:
    """Return ``(total, added, removed, is_new, is_deleted)`` for a single function entry.

    ``total`` is the number of code lines in the function's HEAD body. ``added`` / ``removed``
    are the number of ``+`` / ``-`` diff lines that fall inside the function's HEAD / base range
    AND are themselves code lines (excluding comments / blank lines / Python multi-line strings).
    """
    total = 0
    added = 0
    removed = 0
    if entry.head_range is not None:
        a, b = entry.head_range
        total = sum(1 for ln in book.head_codes if a <= ln <= b)
        added = sum(1 for ln in book.added_line_nos if a <= ln <= b and ln in book.head_codes)
    if entry.base_range is not None:
        c, d = entry.base_range
        removed = sum(1 for ln in book.removed_line_nos if c <= ln <= d and ln in book.base_codes)
    is_new = entry.head_range is not None and entry.base_range is None
    is_deleted = entry.head_range is None and entry.base_range is not None
    return total, added, removed, is_new, is_deleted


def _format_function_line(name: str, total: int, added: int, removed: int, is_new: bool, is_deleted: bool) -> str | None:
    """Produce the indented per-function line, or ``None`` if the function has no real change.

    For a NEW function we never print a ``-N`` because nothing was removed by definition.
    For a DELETED function we never print ``<total>`` (it's 0) -- we use the literal ``DELETED``.
    """
    if added == 0 and removed == 0:
        return None
    label = f"{name}()"
    if is_deleted:
        return f"    {label} DELETED -{removed}"
    if is_new:
        return f"    {label} NEW +{added}"
    parts = [label, str(total)]
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    return "    " + " ".join(parts)


def _function_sort_key(entry: _FunctionEntry) -> tuple[int, int]:
    if entry.head_range is not None:
        return (0, entry.head_range[0])
    if entry.base_range is not None:
        return (1, entry.base_range[0])
    return (2, 0)


def render(summary_json: Path, jsonl_path: Path, output_dir: Path) -> str:
    summaries: list[dict] = json.loads(summary_json.read_text())
    by_path = _load_function_entries(jsonl_path)

    out_lines: list[str] = []
    file_drift_notes: list[str] = []
    for s in summaries:
        book = _build_file_bookkeeping(s, output_dir)
        header = format_header(_HeaderProxy(book))
        out_lines.append(header)
        entries = sorted(by_path.get(book.path, []), key=_function_sort_key)
        per_func_added = 0
        per_func_removed = 0
        any_func_line = False
        for entry in entries:
            total, added, removed, is_new, is_deleted = _attribute_function(entry, book)
            line = _format_function_line(entry.name, total, added, removed, is_new, is_deleted)
            if line is None:
                continue
            out_lines.append(line)
            per_func_added += added
            per_func_removed += removed
            any_func_line = True
        if not any_func_line and (book.added or book.removed):
            out_lines.append("    # note: no per-function attribution available")
        else:
            added_drift = book.added - per_func_added
            removed_drift = book.removed - per_func_removed
            if abs(added_drift) > 5 or abs(removed_drift) > 5:
                note = (
                    f"    # note: per-function +/- differs from file totals by "
                    f"+{added_drift} -{removed_drift}"
                )
                out_lines.append(note)
                file_drift_notes.append(f"{book.path}: {note.strip()}")
        out_lines.append("")
    while out_lines and out_lines[-1] == "":
        out_lines.pop()
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
