# Parallel CI PR command

Post `/make_parallel` as a standalone comment in an original PR's conversation to create or update an internal draft PR for running internal CI workflows. The title starts with `[DONOTMERGE]`; review and merge the original PR. The bot replies with the parallel PR link or a link to the failed workflow run.

The commenter must currently have write, maintain, or admin access to this repository, including access through a team. The workflow queries GitHub's repository permission API; contributor status and organization membership alone do not authorize the command. Comments from bots, edited comments, review comments on individual lines, and comments on issues do not invoke it.

## Repository setup

Merge `.github/workflows/make_parallel.yml` and these scripts into the default branch. The `issue_comment` trigger does not activate from an unmerged PR.

Create a repository Actions secret named `PARALLEL_PR_TOKEN` containing a fine-grained personal access token scoped to this repository, with Contents and Pull requests read/write permissions. Include Workflows write permission to support source revisions that change workflow files. Complete organization approval for the token if required. Keep the token valid and rotate it before expiration. No access to `perso_hugh` is needed.

The token is used only by the authorized synchronization step. The built-in `GITHUB_TOKEN` checks permissions and posts the reply. A separate token lets the parallel PR's CI run automatically, without the additional approval required for PR events generated using `GITHUB_TOKEN`. See GitHub's [workflow triggering documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow) and [Git reference API permissions](https://docs.github.com/en/rest/git/refs#create-a-reference).

## Behavior and trust

Posting the command authorizes the source head observed when the script runs to execute as internal CI code. The command workflow itself checks out only its default-branch SHA, never the external PR's code. PR titles, descriptions, and comment bodies are not interpolated into shell commands. Updates for the same PR are serialized.

The script reuses `hp/parallel-pr-N` and the existing parallel PR. An unchanged source does not push another commit. Source rebases or rewinds produce a merge commit with the exact source tree, preserving internal history without force pushes. A closed or merged parallel PR stops the command; it is not reopened. Existing parallel PRs are converted to draft on update. Workflows that explicitly skip draft PRs remain skipped.

The source PR receives only the command reply; its branch and description are unchanged. Failures may leave a created branch or an updated parallel branch behind; rerunning retries PR creation or metadata updates.

## Maintenance and validation

`parallel_pr.py` is a bundled copy of `Genesis-Embodied-AI/perso_hugh`'s `bin/parallel_pr.py` at commit `43054532`, with its usage paths adapted for this repository. Keep behavior changes synchronized deliberately; no script is downloaded at runtime.

Run the offline API and authorization tests with Python 3.10 from the repository root; no Quadrants build, credentials, or GitHub mutations are needed:

```sh
python -m unittest discover -s .github/workflows/scripts/parallel_pr -p 'test_*.py' -v
```
