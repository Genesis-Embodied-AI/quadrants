"""Authorize /make_parallel using current repository permissions, never author_association."""

import json
import os

from parallel_pr import api


def authorized(event):
    if event.get("action") != "created" or not event.get("issue", {}).get("pull_request"):
        return False
    comment = event["comment"]
    if comment["body"].strip() != "/make_parallel" or comment["user"]["type"] != "User":
        return False
    repository = event["repository"]["full_name"]
    username = comment["user"]["login"]
    permission = api(f"repos/{repository}/collaborators/{username}/permission")
    # GitHub maps maintain to write in this field, including access inherited through teams.
    return permission["permission"] in ("write", "admin")


def main():
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        allowed = authorized(json.load(stream))
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"authorized={str(allowed).lower()}\n")
    print("Authorized repository maintainer." if allowed else "Ignoring unauthorized or unrelated comment.")


if __name__ == "__main__":
    main()
