"""Offline API contract tests; no GitHub mutations or credentials required."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("parallel", Path(__file__).with_name("parallel_pr.py"))
parallel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel)


class ParallelTests(unittest.TestCase):
    def test_url_validation(self):
        self.assertEqual(parallel.parse_url("https://github.com/org/repo/pull/950/"), ("org", "repo", 950))
        for url in ("http://github.com/org/repo/pull/1", "https://evil.test/org/repo/pull/1", "x; echo bad"):
            with self.assertRaises(parallel.Error):
                parallel.parse_url(url)

    def test_same_head_does_not_call_api(self):
        with patch.object(parallel, "api") as api:
            self.assertFalse(parallel.update_branch("root", "branch", "same", "same"))
            api.assert_not_called()

    def test_fast_forward(self):
        with patch.object(
            parallel,
            "api",
            side_effect=[
                {"message": "old", "tree": {"sha": "old-tree"}},
                {"tree": {"sha": "new-tree"}},
                {"status": "ahead"},
                {},
            ],
        ) as api:
            self.assertTrue(parallel.update_branch("root", "branch", "old", "new"))
            api.assert_called_with("root/git/refs/heads/branch", "PATCH", {"sha": "new", "force": False})

    def test_rebase_or_rewind_preserves_history_and_exact_tree(self):
        for status in ("diverged", "behind"):
            with (
                self.subTest(status=status),
                patch.object(
                    parallel,
                    "api",
                    side_effect=[
                        {"message": "old", "tree": {"sha": "old-tree"}},
                        {"tree": {"sha": "source-tree"}},
                        {"status": status},
                        {"sha": "synthetic"},
                        {},
                    ],
                ) as api,
            ):
                parallel.update_branch("root", "branch", "old", "source")
                payload = api.call_args_list[3].args[2]
                self.assertEqual(payload["parents"], ["old", "source"])
                self.assertEqual(payload["tree"], "source-tree")
                api.assert_called_with("root/git/refs/heads/branch", "PATCH", {"sha": "synthetic", "force": False})

    def test_repeated_synthetic_head_is_noop(self):
        with patch.object(
            parallel,
            "api",
            side_effect=[
                {"message": "Sync\n\nParallel-Source-SHA: source", "tree": {"sha": "tree"}},
                {"tree": {"sha": "tree"}},
            ],
        ) as api:
            self.assertFalse(parallel.update_branch("root", "branch", "synthetic", "source"))
            self.assertEqual(api.call_count, 2)

    def source(self):
        return {
            "state": "open",
            "head": {"sha": "source", "ref": "feature", "label": "fork:feature"},
            "base": {"ref": "main"},
            "title": "Feature",
            "body": "Details",
            "html_url": "original",
        }

    def test_create(self):
        with patch.object(
            parallel,
            "api",
            side_effect=[
                self.source(),
                [],
                [],
                {},
                {"html_url": "parallel"},
            ],
        ) as api:
            self.assertEqual(parallel.sync("https://github.com/org/repo/pull/950"), "parallel")
            self.assertEqual(api.call_args_list[3].args[2], {"ref": "refs/heads/hp/parallel-pr-950", "sha": "source"})
            self.assertEqual(api.call_args_list[4].args[2]["head"], "hp/parallel-pr-950")
            self.assertTrue(api.call_args_list[4].args[2]["draft"])
            self.assertTrue(api.call_args_list[4].args[2]["title"].startswith("[DONOTMERGE] "))

    def test_closed_parallel_stops_before_branch_mutation(self):
        with patch.object(
            parallel,
            "api",
            side_effect=[
                self.source(),
                [
                    {
                        "state": "closed",
                        "body": "<!-- parallel-pr-source: org/repo#950 -->",
                        "html_url": "parallel",
                    }
                ],
            ],
        ) as api:
            with self.assertRaisesRegex(parallel.Error, "closed or merged"):
                parallel.sync("https://github.com/org/repo/pull/950")
            self.assertEqual(api.call_count, 2)

    def test_update_reuses_pr(self):
        pr = {
            "state": "open",
            "body": "<!-- parallel-pr-source: org/repo#950 -->",
            "html_url": "parallel",
            "number": 951,
            "title": "Old",
            "base": {"ref": "main"},
            "draft": True,
        }
        with (
            patch.object(
                parallel,
                "api",
                side_effect=[
                    self.source(),
                    [pr],
                    [{"ref": "refs/heads/hp/parallel-pr-950", "object": {"sha": "old"}}],
                    {},
                ],
            ) as api,
            patch.object(parallel, "update_branch", return_value=True) as update,
        ):
            parallel.sync("https://github.com/org/repo/pull/950")
            update.assert_called_once_with("repos/org/repo", "hp/parallel-pr-950", "old", "source")
            self.assertEqual(api.call_args.args[:2], ("repos/org/repo/pulls/951", "PATCH"))
            self.assertTrue(api.call_args.args[2]["title"].startswith("[DONOTMERGE] "))

    def test_existing_ready_pr_is_converted_before_sync(self):
        pr = {
            "state": "open",
            "body": "<!-- parallel-pr-source: org/repo#950 -->",
            "html_url": "parallel",
            "number": 951,
            "title": "Old",
            "base": {"ref": "main"},
            "draft": False,
            "node_id": "PR_node",
        }
        with patch.object(
            parallel,
            "api",
            side_effect=[
                self.source(),
                [pr],
                {"data": {"convertPullRequestToDraft": {"pullRequest": {"isDraft": True}}}},
                [{"ref": "refs/heads/hp/parallel-pr-950", "object": {"sha": "source"}}],
                {},
            ],
        ) as api:
            parallel.sync("https://github.com/org/repo/pull/950")
            self.assertEqual(api.call_args_list[2].args[:2], ("graphql", "POST"))
            self.assertEqual(api.call_args_list[2].args[2]["variables"], {"id": "PR_node"})

    def test_failed_draft_conversion_stops_sync(self):
        pr = {
            "state": "open",
            "body": "<!-- parallel-pr-source: org/repo#950 -->",
            "html_url": "parallel",
            "draft": False,
            "node_id": "PR_node",
        }
        with patch.object(
            parallel,
            "api",
            side_effect=[
                self.source(),
                [pr],
                {"errors": [{"message": "Permission denied"}]},
            ],
        ) as api:
            with self.assertRaisesRegex(parallel.Error, "Could not convert"):
                parallel.sync("https://github.com/org/repo/pull/950")
            self.assertEqual(api.call_count, 3)


if __name__ == "__main__":
    unittest.main()
