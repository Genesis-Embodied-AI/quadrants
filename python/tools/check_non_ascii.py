# type: ignore
"""Scan added lines in a unified diff for non-ASCII characters.

Used by the CI workflow ``.github/workflows/check_non_ascii.yml`` to fail a pull request when any
newly added line in a code or documentation file introduces a non-ASCII character. Only added
lines (those starting with ``+`` in the diff, excluding the ``+++`` file header) are inspected, so
pre-existing violations do not block unrelated PRs.

The report lists every violation with its filename, line number, column, and the offending
character (code point, repr, and Unicode name), followed by a total count.

Usage:
    python python/tools/check_non_ascii.py --diff-file /tmp/pr_diff.patch
    python python/tools/check_non_ascii.py --base-ref origin/main
    git diff origin/main...HEAD -- '*.py' | python python/tools/check_non_ascii.py
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass

# Matches a unified-diff hunk header. Captures the old-side line count (group 1, absent -> 1), the
# new-file start line (group 2), and the new-side line count (group 3, absent -> 1), e.g.
# "@@ -12,7 +34,9 @@ optional section heading" -> ("7", "34", "9").
_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Violation:
    path: str
    line: int
    col: int
    char: str

    def describe(self) -> str:
        code_point = ord(self.char)
        try:
            name = unicodedata.name(self.char)
        except ValueError:
            name = "<unnamed>"
        return f"{self.path}:{self.line}:{self.col}: non-ASCII character U+{code_point:04X} {self.char!r} ({name})"


def _strip_diff_prefix(target: str) -> str | None:
    """Return the file path from a ``+++`` diff header, or None for /dev/null (deleted files)."""
    target = target.strip()
    if target == "/dev/null":
        return None
    # Drop the trailing "\t<timestamp>" some diff variants append.
    target = target.split("\t", 1)[0]
    if target.startswith("b/"):
        target = target[2:]
    return target


def find_violations(diff_text: str) -> list[Violation]:
    """Parse a unified diff and return non-ASCII violations on added lines, in file order.

    ``+++ `` / ``--- `` are recognized as file headers ONLY outside a hunk body. Inside a hunk, an
    added line whose content starts with ``++ `` is emitted by git as ``+++ ...`` (and a removed
    line starting with ``-- `` as ``--- ...``); treating those as file headers would skip the line
    (letting its non-ASCII characters through) and corrupt the tracked path and line numbers. Hunk
    boundaries are tracked via the line counts declared in the ``@@`` header, so content lines are
    always classified by their diff marker rather than mistaken for headers.
    """
    violations: list[Violation] = []
    path: str | None = None
    new_lineno = 0
    # Old-file / new-file lines still expected in the current hunk body. Both <= 0 means we are
    # between hunks, i.e. in file-header territory.
    remaining_old = 0
    remaining_new = 0

    for raw in diff_text.splitlines():
        if remaining_old <= 0 and remaining_new <= 0:
            hunk = _HUNK_RE.match(raw)
            if hunk:
                remaining_old = int(hunk.group(1)) if hunk.group(1) else 1
                new_lineno = int(hunk.group(2))
                remaining_new = int(hunk.group(3)) if hunk.group(3) else 1
                continue
            if raw.startswith("+++ "):
                path = _strip_diff_prefix(raw[4:])
            # Any other line here (diff --git, index, "--- " header, mode lines, ...) is ignored.
            continue

        # Inside a hunk body: classify strictly by the leading diff marker.
        marker = raw[0] if raw else " "
        if marker == "+":
            content = raw[1:]
            if path is not None:
                for idx, ch in enumerate(content):
                    if ord(ch) > 127:
                        violations.append(Violation(path, new_lineno, idx + 1, ch))
            new_lineno += 1
            remaining_new -= 1
        elif marker == "-":
            # Removed lines exist only in the old file, so they are not scanned.
            remaining_old -= 1
        elif marker == "\\":
            # "\ No newline at end of file" carries no line of its own; do not count it.
            pass
        else:
            # Context line (marker " ", or a defensively-handled empty line): on both sides.
            new_lineno += 1
            remaining_old -= 1
            remaining_new -= 1

    return violations


def _read_diff(args: argparse.Namespace) -> str:
    if args.diff_file:
        with open(args.diff_file, "r", encoding="utf-8", errors="surrogateescape") as f:
            return f.read()
    if args.base_ref:
        cmd = ["git", "diff", f"{args.base_ref}...HEAD"]
        if args.pathspec:
            cmd += ["--", *args.pathspec]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, errors="surrogateescape")
        return result.stdout
    return sys.stdin.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--diff-file", help="Path to a unified diff file to scan.")
    source.add_argument("--base-ref", help="Git ref to diff HEAD against (runs `git diff <ref>...HEAD`).")
    parser.add_argument(
        "pathspec",
        nargs="*",
        help="Optional git pathspecs to limit the diff (only used with --base-ref).",
    )
    args = parser.parse_args(argv)

    diff_text = _read_diff(args)
    violations = find_violations(diff_text)

    if not violations:
        print("PASS: no non-ASCII characters found in added lines of code or doc changes.")
        return 0

    print(f"FAIL: found {len(violations)} non-ASCII character(s) in added lines:")
    print("")
    for v in violations:
        print(f"  {v.describe()}")
    print("")
    print(
        "Non-ASCII characters are not allowed in code or documentation changes. Replace them with "
        "their ASCII equivalents (e.g. straight quotes, hyphens, plain spaces)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
