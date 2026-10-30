import importlib.util
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

tool_path = Path(__file__).parents[2] / "python/tools/cleanup_doc_previews.py"
spec = importlib.util.spec_from_file_location("cleanup_doc_previews", tool_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_discovers_only_marked_previews() -> None:
    # Ignore the main site, releases, and unmarked directories; recognize previews with slash-containing names.
    files = {
        ".preview",
        "main/index.html",
        "v0.4.0/index.html",
        "unmarked/_static/theme.css",
        "hp/current/.preview",
        "simple/.preview",
    }
    assert tool.preview_directories(files) == ["hp/current", "simple"]


def make_site(tmp_path: Path) -> Callable[..., str]:
    # Commit a small gh-pages fixture containing the root site, protected docs, and previews of different ages.
    def git(*args: str) -> str:
        # Run Git in the temporary test repository and return its stdout.
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True)

    git("init", "-b", "gh-pages")
    for name in ("", "main/", "hp/old/", "hp/new/", "hp/boundary/"):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text("docs\n")
        if name.startswith("hp/"):
            (directory / ".preview").touch()
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Initial site")
    return git


@pytest.mark.parametrize("apply", [False, True])
def test_cleanup_only_stages_expired_previews(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply: bool) -> None:
    # Delete only previews older than seven days; a dry run must leave every tracked file unchanged.
    git = make_site(tmp_path)
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    ages = {"hp/old": 8, "hp/new": 1, "hp/boundary": 7}
    monkeypatch.setattr(tool, "last_updated", lambda repo, revision, directory: now - timedelta(days=ages[directory]))
    assert tool.cleanup(tmp_path, "owner/repo", apply=apply, now=now) == ["hp/old"]
    changed = git("diff", "--cached", "--name-status").splitlines()
    assert changed == (["D\thp/old/.preview", "D\thp/old/index.html"] if apply else [])
    assert git("diff", "--name-only") == ""


def test_history_error_does_not_partially_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An API failure after finding an expired preview must leave the entire checkout untouched.
    git = make_site(tmp_path)

    def updated(repo: str, revision: str, directory: str) -> datetime:
        # Simulate a history lookup failure after an earlier preview has already been classified as expired.
        if directory == "hp/new":
            raise RuntimeError("API unavailable")
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(tool, "last_updated", updated)
    with pytest.raises(RuntimeError, match="API unavailable"):
        tool.cleanup(tmp_path, "owner/repo", apply=True)
    assert git("status", "--porcelain") == ""
