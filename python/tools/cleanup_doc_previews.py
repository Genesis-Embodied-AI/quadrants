"""Stage deletion of gh-pages previews whose content has not changed for seven days.

Recognizes both Sphinx previews and legacy pdoc previews. The root site, main/master, and release/version directories
are preserved. Uses GitHub's commit history API so the Pages checkout can remain shallow. Requires git and gh.
"""

import argparse
import json
import re
import subprocess
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode

PROTECTED = {
    "main",
    "master",
    "stable",
    "latest",
    "release",
    "releases",
    "versions",
    "autoapi",
    "taichi",
    "user_guide",
    "contributor_guide",
}


def preview_directories(files: Iterable[str]) -> list[str]:
    # Find outermost Sphinx and pdoc previews, excluding the root site and protected documentation.
    files = set(files)
    candidates = set()
    for filename in files:
        if "/_static/" in filename:
            directory = filename.split("/_static/", 1)[0]
        elif filename.endswith("/search.js"):
            directory = filename.removesuffix("/search.js")
        else:
            continue
        parts = PurePosixPath(directory).parts
        if not parts or any(part in (".", "..") for part in parts) or directory.startswith("/"):
            continue
        if parts[0] in PROTECTED or parts[0].startswith(("_", ".")) or re.match(r"v?\d+(?:\.|$)", parts[0]):
            continue
        if f"{directory}/index.html" in files:
            candidates.add(directory)
    # A retained parent also retains nested previews; never remove a child independently of its parent.
    return sorted(
        path
        for path in candidates
        if not any(parent.as_posix() in candidates for parent in PurePosixPath(path).parents)
    )


def last_updated(repository: str, revision: str, directory: str) -> datetime:
    # Ask GitHub when this directory last changed, using the checkout's exact revision.
    query = urlencode({"sha": revision, "path": directory + "/", "per_page": 1})
    result = subprocess.run(
        ["gh", "api", f"repos/{repository}/commits?{query}"], check=True, capture_output=True, text=True
    )
    commits = json.loads(result.stdout)
    if not commits:
        raise RuntimeError(f"No commit history for {directory}; refusing to delete previews")
    return datetime.fromisoformat(commits[0]["commit"]["committer"]["date"].replace("Z", "+00:00"))


def cleanup(site: Path, repository: str, *, apply: bool = False, now: datetime | None = None) -> list[str]:
    # Report expired previews and optionally stage their deletion after validating the checkout and all dates.
    def git(*args: str) -> str:
        # Run Git in the site checkout, returning stdout and raising if the command fails.
        return subprocess.check_output(["git", "-C", str(site), *args], text=True)

    if git("branch", "--show-current").strip() != "gh-pages":
        raise RuntimeError("Expected a gh-pages checkout")
    if git("status", "--porcelain").strip():
        raise RuntimeError("Expected a clean checkout")
    revision = git("rev-parse", "HEAD").strip()
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=7)
    directories = preview_directories(git("ls-files", "-z").split("\0"))
    expired = []
    print(f"Preview cutoff: {cutoff.isoformat()}", flush=True)
    # Finish all history lookups before changing the index: API errors must leave the site untouched.
    for directory in directories:
        updated = last_updated(repository, revision, directory)
        delete = updated < cutoff
        print(f"{'DELETE' if delete else 'KEEP'} {directory} (updated {updated.isoformat()})", flush=True)
        if delete:
            expired.append(directory)
    print(f"{'Removing' if apply else 'Would remove'} {len(expired)} of {len(directories)} previews", flush=True)
    if apply and expired:
        git("--literal-pathspecs", "rm", "-r", "-q", "--", *expired)
    return expired


def main() -> None:
    # Parse command-line options and run cleanup, defaulting to a dry run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="Clean, shallow gh-pages checkout")
    parser.add_argument("--repository", required=True, help="GitHub OWNER/REPO")
    parser.add_argument("--apply", action="store_true", help="Stage deletions; default is a dry run")
    args = parser.parse_args()
    cleanup(args.site, args.repository, apply=args.apply)


if __name__ == "__main__":
    main()
