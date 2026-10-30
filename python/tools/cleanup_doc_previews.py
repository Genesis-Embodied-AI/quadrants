"""Stage deletion of gh-pages previews whose content has not changed for seven days.

Only directories containing a .preview marker are eligible; the root site and unmarked documentation are preserved.
Uses GitHub's commit history API so the Pages checkout can remain shallow. Requires git and gh.
"""

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlencode


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
    # -z returns unquoted, NUL-separated filenames, so splitting on \0 also handles names containing newlines.
    directories = [
        str(PurePosixPath(path).parent) for path in git("ls-files", "-z").split("\0") if path.endswith("/.preview")
    ]
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
    parser.add_argument("--site-dir", type=Path, required=True, help="Clean, shallow gh-pages checkout")
    parser.add_argument("--repository", required=True, help="GitHub OWNER/REPO")
    parser.add_argument("--apply", action="store_true", help="Stage deletions; default is a dry run")
    args = parser.parse_args()
    cleanup(args.site_dir, args.repository, apply=args.apply)


if __name__ == "__main__":
    main()
