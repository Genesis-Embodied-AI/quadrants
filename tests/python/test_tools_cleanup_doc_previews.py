import importlib.util
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

tool_path = Path(__file__).parents[2] / "python/tools/cleanup_doc_previews.py"
spec = importlib.util.spec_from_file_location("cleanup_doc_previews", tool_path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_discovers_sphinx_and_legacy_previews_without_main_or_releases():
    files = {"index.html", "_static/theme.css", "search.js"}
    for name in ("hp/current", "main", "master", "v0.4.0", "0.4.0", "releases/v1", "user_guide", "_sources/example"):
        files.update({f"{name}/index.html", f"{name}/_static/theme.css"})
    files.update({"hp/legacy/index.html", "hp/legacy/search.js", "hp/incomplete/_static/theme.css"})
    assert tool.preview_directories(files) == ["hp/current", "hp/legacy"]


def test_nested_previews_are_kept_with_their_parent():
    files = {f"{name}/{file}" for name in ("hp/parent", "hp/parent/child") for file in ("index.html", "search.js")}
    assert tool.preview_directories(files) == ["hp/parent"]


def make_site(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True)

    git("init", "-b", "gh-pages")
    for name in ("", "main/", "hp/old/", "hp/new/", "hp/boundary/"):
        directory = tmp_path / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.html").write_text("docs\n")
        (directory / "search.js").write_text("search\n")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Initial site")
    return git


@pytest.mark.parametrize("apply", [False, True])
def test_cleanup_only_stages_expired_previews(tmp_path, monkeypatch, apply):
    git = make_site(tmp_path)
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    ages = {"hp/old": 8, "hp/new": 1, "hp/boundary": 7}
    calls = []

    def updated(repo, revision, directory):
        calls.append((repo, revision))
        return now - timedelta(days=ages[directory])

    monkeypatch.setattr(tool, "last_updated", updated)
    head = git("rev-parse", "HEAD").strip()
    assert tool.cleanup(tmp_path, "owner/repo", apply=apply, now=now) == ["hp/old"]
    assert set(calls) == {("owner/repo", head)}
    changed = git("diff", "--cached", "--name-status").splitlines()
    assert changed == (["D\thp/old/index.html", "D\thp/old/search.js"] if apply else [])
    assert git("rev-parse", "HEAD").strip() == head
    assert (tmp_path / "index.html").is_file()
    assert (tmp_path / "main/index.html").is_file()


def test_history_error_does_not_partially_delete(tmp_path, monkeypatch):
    git = make_site(tmp_path)

    def updated(repo, revision, directory):
        if directory == "hp/new":
            raise RuntimeError("API unavailable")
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(tool, "last_updated", updated)
    with pytest.raises(RuntimeError, match="API unavailable"):
        tool.cleanup(tmp_path, "owner/repo", apply=True)
    assert git("status", "--porcelain") == ""


@pytest.mark.parametrize("condition", ["dirty", "wrong_branch"])
def test_refuses_unsafe_checkout(tmp_path, condition):
    git = make_site(tmp_path)
    if condition == "dirty":
        (tmp_path / "index.html").write_text("uncommitted\n")
    else:
        git("checkout", "-b", "main")
    with pytest.raises(RuntimeError, match="Expected"):
        tool.cleanup(tmp_path, "owner/repo", apply=True)


def test_history_query_is_pinned_and_encodes_branch_name(monkeypatch):
    def run(command, **kwargs):
        query = parse_qs(urlsplit(command[2]).query)
        assert query == {"sha": ["abc123"], "path": ["hp/example/"], "per_page": ["1"]}
        commits = [{"commit": {"committer": {"date": "2026-10-30T12:00:00Z"}}}]
        return subprocess.CompletedProcess(command, 0, json.dumps(commits))

    monkeypatch.setattr(tool.subprocess, "run", run)
    assert tool.last_updated("owner/repo", "abc123", "hp/example") == datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def test_missing_history_is_an_error(monkeypatch):
    monkeypatch.setattr(tool.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "[]"))
    with pytest.raises(RuntimeError, match="No commit history"):
        tool.last_updated("owner/repo", "abc123", "hp/example")
