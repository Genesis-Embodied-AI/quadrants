import importlib
import os
import tempfile
from pathlib import Path

import pytest

# I don't want to make python/tools a module, and I don't want to move this tool
# into `quadrants` namespace, so that leaves temporarily importing it somehow
tools_path = Path(__file__).parent.parent.parent / "python" / "tools" / "markdown_link_check.py"
spec = importlib.util.spec_from_file_location("markdown_link_check", tools_path)
markdown_link_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(markdown_link_check)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def write_md(base_dir, filename, content):
    path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


def test_find_markdown_files(temp_dir):
    write_md(temp_dir, "a.md", "# Title")
    write_md(temp_dir, "b.txt", "not markdown")
    os.mkdir(os.path.join(temp_dir, "sub"))
    write_md(temp_dir, "sub/c.md", "# Sub")
    files = markdown_link_check.find_markdown_files(temp_dir)
    assert len(files) == 2
    assert any(f.endswith("a.md") for f in files)
    assert any(f.endswith("c.md") for f in files)


def test_check_markdown_links_valid(temp_dir, capsys):
    md = "# Title\n[Link](other.md)\n"
    other = "# Other"
    write_md(temp_dir, "main.md", md)
    write_md(temp_dir, "other.md", other)
    markdown_link_check.check_markdown_links(os.path.join(temp_dir, "main.md"), temp_dir)
    out = capsys.readouterr().out
    assert "❌" not in out


def test_check_markdown_links_broken_file(temp_dir, capsys):
    md = "# Title\n[Missing](missing.md)\n"
    write_md(temp_dir, "main.md", md)
    markdown_link_check.check_markdown_links(os.path.join(temp_dir, "main.md"), temp_dir)
    out = capsys.readouterr().out
    assert "❌ Broken link" in out


def test_check_anchor_found(temp_dir, capsys):
    md = "# My Header\n"
    path = write_md(temp_dir, "doc.md", md)
    markdown_link_check.check_anchor(path, "my-header")
    out = capsys.readouterr().out
    assert "❌" not in out


def test_check_anchor_not_found(temp_dir, capsys):
    md = "# My Header\n"
    path = write_md(temp_dir, "doc.md", md)
    markdown_link_check.check_anchor(path, "not-present")
    out = capsys.readouterr().out
    assert "❌ Broken anchor" in out


def test_check_anchor_symbol_removal(temp_dir, capsys):
    md = "# My `Header`.\n"
    path = write_md(temp_dir, "doc.md", md)
    markdown_link_check.check_anchor(path, "my-header")
    out = capsys.readouterr().out
    assert "❌" not in out


def test_external_and_mailto_links(temp_dir, capsys):
    md = "# Title\n[Google](https://google.com)\n[Email](mailto:test@example.com)\n"
    path = write_md(temp_dir, "main.md", md)
    markdown_link_check.check_markdown_links(path, temp_dir)
    out = capsys.readouterr().out
    assert "External link" in out


def test_anchor_only_link(temp_dir, capsys):
    md = "# Section 1\n[Go](#section-1)\n"
    path = write_md(temp_dir, "main.md", md)
    markdown_link_check.check_markdown_links(path, temp_dir)
    out = capsys.readouterr().out
    assert "❌" not in out


def test_pr_review(temp_dir, capsys):
    md = """

- [PR review & merging checklist](#pr-review-merging-checklist)

### PR review & merging checklist

Follow this checklist during PR review or merging:
"""
    path = write_md(temp_dir, "main.md", md)
    markdown_link_check.check_markdown_links(path, temp_dir)
    out = capsys.readouterr().out
    assert "❌" not in out
