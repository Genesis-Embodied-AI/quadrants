#!/usr/bin/env python3
"""Create or update an internal CI PR using an existing `gh auth login` session.

Usage: python .github/workflows/scripts/parallel_pr/parallel_pr.py https://github.com/OWNER/REPO/pull/NUMBER

No checkout or Python dependencies are needed. Each invocation copies the source head observed at startup; invoking this
script authorizes that revision to run with internal CI credentials. Rebased source branches are copied with a merge
commit whose tree exactly matches the source, preserving published history without force pushing. Closed parallel PRs
are never reopened. Parallel PRs are drafts with a [DONOTMERGE] title prefix. The original PR is not modified.
"""

import argparse
import json
import re
import subprocess
import sys
from urllib.parse import quote


class Error(Exception):
    """An actionable CLI error."""


def api(endpoint, method="GET", data=None):
    command = ["gh", "api", "--hostname", "github.com", "--method", method, endpoint]
    if data is not None:
        command += ["--input", "-"]
    result = subprocess.run(
        command, input=json.dumps(data) if data is not None else None, text=True, capture_output=True
    )
    if result.returncode:
        raise Error(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout) if result.stdout.strip() else None


def parse_url(url):
    match = re.fullmatch(r"https://github\.com/([\w.-]+)/([\w.-]+)/pull/([1-9][0-9]*)/?", url)
    if not match:
        raise Error("Expected https://github.com/OWNER/REPO/pull/NUMBER")
    owner, repo, number = match.groups()
    return owner, repo, int(number)


def update_branch(root, branch, old_sha, source_sha):
    """Fast-forward, or append a merge with the source tree when histories diverge."""
    if old_sha == source_sha:
        return False
    old = api(f"{root}/git/commits/{old_sha}")
    source = api(f"{root}/git/commits/{source_sha}")
    marker = f"Parallel-Source-SHA: {source_sha}"
    if marker in old["message"].splitlines() and old["tree"]["sha"] == source["tree"]["sha"]:
        return False
    comparison = api(f"{root}/compare/{old_sha}...{source_sha}")
    new_sha = source_sha
    if comparison["status"] not in ("ahead", "identical"):
        commit = api(
            f"{root}/git/commits",
            "POST",
            {
                "message": f"Sync parallel CI branch to {source_sha}\n\n{marker}",
                "tree": source["tree"]["sha"],
                "parents": [old_sha, source_sha],
            },
        )
        new_sha = commit["sha"]
    # GitHub rejects non-fast-forward updates, including conflicting concurrent invocations.
    api(f"{root}/git/refs/heads/{branch}", "PATCH", {"sha": new_sha, "force": False})
    return True


def sync(url):
    owner, repo, number = parse_url(url)
    root = f"repos/{owner}/{repo}"
    source = api(f"{root}/pulls/{number}")
    if source["state"] != "open":
        raise Error("The source PR is closed or merged; nothing changed.")
    branch = f"hp/parallel-pr-{number}"
    source_sha = source["head"]["sha"]
    base = source["base"]["ref"]
    if source["head"]["ref"] == branch and source["head"]["repo"]["full_name"].lower() == f"{owner}/{repo}".lower():
        raise Error("The source PR already uses the parallel branch name.")
    marker = f"<!-- parallel-pr-source: {owner}/{repo}#{number} -->"
    prs = []
    page = 1
    while True:
        batch = api(f"{root}/pulls?state=all&head={quote(owner + ':' + branch, safe='')}&per_page=100&page={page}")
        prs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    if len(prs) > 1:
        raise Error(f"Multiple PRs use {branch}; resolve the ambiguity manually.")
    parallel = prs[0] if prs else None
    if parallel:
        if marker not in (parallel.get("body") or ""):
            raise Error(f"Existing PR {parallel['html_url']} is not managed by this script; nothing changed.")
        if parallel["state"] != "open":
            raise Error(f"Parallel PR is closed or merged: {parallel['html_url']}; nothing changed.")
        if not parallel["draft"]:
            result = api(
                "graphql",
                "POST",
                {
                    "query": "mutation($id: ID!) { convertPullRequestToDraft(input: {pullRequestId: $id}) "
                    "{ pullRequest { isDraft } } }",
                    "variables": {"id": parallel["node_id"]},
                },
            )
            if result.get("errors") or not result["data"]["convertPullRequestToDraft"]["pullRequest"]["isDraft"]:
                raise Error(f"Could not convert parallel PR to draft: {result}")

    # matching-refs returns an empty list for an absent branch, without hiding authentication/API failures as 404s.
    refs = api(f"{root}/git/matching-refs/heads/{branch}")
    ref = next((item for item in refs if item["ref"] == f"refs/heads/{branch}"), None)
    if ref:
        old_sha = ref["object"]["sha"]
        if not parallel and old_sha != source_sha:
            raise Error(f"Branch {branch} already exists without a managed PR; nothing changed.")
        changed = update_branch(root, branch, old_sha, source_sha)
    else:
        if parallel:
            raise Error("The parallel PR's branch is missing; nothing changed.")
        api(f"{root}/git/refs", "POST", {"ref": f"refs/heads/{branch}", "sha": source_sha})
        changed = True

    body = (
        "This parallel PR exists in order to run internal CI workflows. **Do not merge this PR.**\n\n"
        f"{marker}\n"
        f"CI-only parallel of {source['html_url']}. Review and merge the original PR.\n\n"
        f"Source commit: `{source_sha}`\n"
        f"Source branch: `{source['head']['label']}`\n"
        f"Target branch: `{base}`\n\n"
        "Rerun `parallel_pr.py` with the original PR URL to copy its latest revision here. "
        "The internal branch preserves its history; after a source rebase, its tip may be a synthetic merge commit "
        "with exactly the source commit's file tree.\n\n"
        "---\n\nOriginal PR description:\n\n" + (source.get("body") or "(No description.)")
    )
    payload = {"title": f"[DONOTMERGE] [CI parallel #{number}] {source['title']}"[:256], "body": body, "base": base}
    if parallel:
        if any(parallel.get(key) != payload[key] for key in ("title", "body")) or parallel["base"]["ref"] != base:
            api(f"{root}/pulls/{parallel['number']}", "PATCH", payload)
        print("Updated parallel branch." if changed else "Source unchanged; parallel branch already current.")
    else:
        parallel = api(f"{root}/pulls", "POST", {**payload, "head": branch, "draft": True})
        print("Created parallel PR.")
    print(parallel["html_url"])
    return parallel["html_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pr_url", help="Original GitHub PR URL")
    args = parser.parse_args()
    try:
        sync(args.pr_url)
    except (Error, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
