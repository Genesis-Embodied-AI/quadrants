import importlib
import os
import subprocess
from pathlib import Path
from typing import Generator, Iterable

import pytest

# I don't want to make python/tools a module, and I don't want to move this tool
# into `quadrants` namespace, so that leaves temporarily importing it somehow
tools_path = Path(__file__).parent.parent.parent / "python" / "tools" / "run_modified_files.py"
spec = importlib.util.spec_from_file_location("run_modified_files", tools_path)
run_modified_files = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_modified_files)


class Runner:
    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def __call__(self, args: Iterable[str]) -> str:
        return subprocess.check_output(cwd=self.cwd, args=args).decode("utf-8")


@pytest.fixture(scope="function")
def git_repo(tmp_path) -> Generator[str, None, None]:
    parent_repo = tmp_path / "parent"
    os.makedirs(parent_repo)
    child_repo = tmp_path / "child"
    origin_runner = Runner(cwd=parent_repo)
    origin_runner(["git", "init", "-b", "main"])
    origin_runner(["git", "config", "user.email", "test@example.com"])
    origin_runner(["git", "config", "user.name", "Test User"])
    origin_runner(["touch", "foo"])
    origin_runner(["git", "add", "foo"])
    origin_runner(["git", "commit", "-m", "foo"])

    subprocess.check_output(["git", "clone", parent_repo, "child"], cwd=tmp_path).decode("utf-8")

    child_runner = Runner(cwd=child_repo)
    child_runner(["git", "config", "user.email", "test@example.com"])
    child_runner(["git", "config", "user.name", "Test User"])
    child_runner(["git", "status"])
    child_runner(["ls"])
    yield child_repo


@pytest.fixture(scope="function")
def runner(git_repo) -> Runner:
    return Runner(cwd=git_repo)


def test_get_changed_files_added_file(git_repo, runner):
    # Create a file and add it
    runner(["ls", "-lh"])
    runner(["pwd"])
    f = git_repo / "foo.py"
    f.write_text("print('hi')")
    runner(["git", "add", "foo.py"])
    runner(["git", "commit", "-m", "add foo.py"])
    # Modify file
    f.write_text("print('hello')")
    # Should show as changed
    files = run_modified_files.get_changed_files("*.py", cwd=git_repo)
    assert "foo.py" in files or str(f) in files


def test_get_changed_files_untracked(git_repo, runner):
    f = git_repo / "bar.py"
    f.write_text("print('bar')")
    # Not added to git
    files = run_modified_files.get_changed_files("*.py", cwd=git_repo)
    assert "bar.py" in files or str(f) in files


def test_get_changed_files_pattern(git_repo, runner):
    (git_repo / "a.py").write_text("a")
    (git_repo / "b.txt").write_text("b")
    runner(["git", "add", "."])
    runner(["git", "commit", "-m", "add files"])
    # Change a.py
    (git_repo / "a.py").write_text("changed")
    files = run_modified_files.get_changed_files("*.py", cwd=git_repo)
    assert any(f.endswith("a.py") for f in files)
    assert not any(f.endswith("b.txt") for f in files)


def test_get_changed_files_none(git_repo, runner):
    # No changes
    files = run_modified_files.get_changed_files("*.md", cwd=git_repo)
    assert files == []


@pytest.mark.parametrize(
    "filenames,pattern,expected",
    [
        (["foo.py"], "*.py", ["foo.py"]),
        (["foo.py", "bar.txt"], "*.py", ["foo.py"]),
        (["foo.py", "bar.py"], "*.py", ["foo.py", "bar.py"]),
        (["foo.cpp", "bar.h"], "*.py", []),
        (["foo.py", "foo.pyc"], "foo.*", ["foo.py", "foo.pyc"]),
        (["foo.md"], "*.md", ["foo.md"]),
        ([], "*.py", []),
    ],
)
def test_get_changed_files_parametrize(git_repo, filenames, pattern, expected, runner):
    # Create files and add/commit them
    for fname in filenames:
        f = git_repo / fname
        f.write_text("content")
        runner(["git", "add", fname])
    if filenames:
        runner(["git", "commit", "-m", "add files"])
    # Modify all files to make them "changed"
    for fname in filenames:
        f = git_repo / fname
        f.write_text("changed content")
    files = run_modified_files.get_changed_files(pattern, cwd=git_repo)
    # Only compare basenames for assertion
    found = {Path(f).name for f in files}
    assert set(expected) == found


@pytest.mark.parametrize(
    "untracked,pattern,expected",
    [
        (["baz.py"], "*.py", ["baz.py"]),
        (["baz.txt"], "*.py", []),
        (["baz.py", "baz.txt"], "*.txt", ["baz.txt"]),
    ],
)
def test_get_changed_files_untracked_parametrize(git_repo, untracked, pattern, expected, runner):
    # Create and do not add to git
    for fname in untracked:
        (git_repo / fname).write_text("untracked")
    files = run_modified_files.get_changed_files(pattern, cwd=git_repo)
    found = {Path(f).name for f in files}
    assert set(expected) == found
