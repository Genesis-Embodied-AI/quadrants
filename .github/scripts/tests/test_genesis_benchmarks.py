"""Exercise benchmark lifecycle decisions without GitHub or W&B writes."""

import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import genesis_benchmarks as benchmarks


class BenchmarkLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(
            record_id="100-1",
            run_id=100,
            attempt=1,
            activation="100",
            pr=12,
            genesis_sha="genesis",
            compatibility_key="environment",
            bootstrap=False,
            head_sha="head",
            wheel=dict(tree="tree", commit="tested"),
            baseline_sha=None,
            main_cutoff="main",
        )

    def test_baseline_uses_main_order_and_excludes_candidate(self):
        older = dict(self.manifest, record_id="90-1", run_id=90, baseline_sha="older")
        newer = dict(self.manifest, record_id="80-1", run_id=80, baseline_sha="newer")
        candidate = dict(self.manifest, baseline_sha="candidate")
        wrong_environment = dict(newer, compatibility_key="other", baseline_sha="tip")
        records = [older, candidate, newer, wrong_environment]
        history = ["tip", "candidate", "newer", "older"]
        self.assertEqual(benchmarks.select_baseline(self.manifest, records, history, "candidate"), newer)
        self.assertIsNone(benchmarks.select_baseline(self.manifest, [candidate], history, "candidate"))

    def test_merge_requires_head_and_tree_match(self):
        pull = dict(merged=True, head=dict(sha="head"), merge_commit_sha="landed")
        with patch.object(benchmarks, "github", side_effect=[pull, dict(tree=dict(sha="tree"))]):
            self.assertEqual(benchmarks.promote(self.manifest), "landed")
        with patch.object(benchmarks, "github", side_effect=[pull, dict(tree=dict(sha="different"))]):
            self.assertIsNone(benchmarks.promote(self.manifest))
        with patch.object(benchmarks, "github", return_value=dict(pull, head=dict(sha="superseded"))):
            self.assertIsNone(benchmarks.promote(self.manifest))
        with patch.object(benchmarks, "github", return_value=dict(pull, merged=False)):
            self.assertIsNone(benchmarks.promote(self.manifest))

    def test_activation_rejects_stale_request_and_is_idempotent(self):
        with (
            patch.object(benchmarks, "active_state", return_value=(dict(activation="101"), "blob")),
            patch.object(benchmarks, "github") as github,
        ):
            with self.assertRaisesRegex(ValueError, "superseded"):
                benchmarks.activate(self.manifest)
            github.assert_not_called()
        with (
            patch.object(benchmarks, "active_state", return_value=(self.manifest, "blob")),
            patch.object(benchmarks, "github") as github,
        ):
            benchmarks.activate(self.manifest)
            github.assert_not_called()

    def test_activation_updates_pointer_with_expected_blob(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, GITHUB_STEP_SUMMARY=str(Path(directory) / "summary")),
            patch.object(benchmarks, "active_state", return_value=(dict(activation="99"), "old-blob")),
            patch.object(benchmarks, "github") as github,
        ):
            benchmarks.activate(self.manifest)
            payload = github.call_args.args[1]
            self.assertEqual(payload["sha"], "old-blob")
            self.assertEqual(json.loads(base64.b64decode(payload["content"])), self.manifest)

    def test_merge_during_publication_is_promoted(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(benchmarks, "collect_results", return_value=self.manifest),
            patch.object(benchmarks, "promote", side_effect=[None, "landed"]),
            patch.object(benchmarks.wandb, "init") as initialize,
            patch.object(benchmarks.wandb, "Artifact"),
            patch.object(benchmarks.wandb, "Api") as api,
        ):
            run = initialize.return_value.__enter__.return_value
            run.log_artifact.return_value.wait.return_value.name = "measurements-100-1:v0"
            published = api.return_value.run.return_value
            published.config = {}
            result = benchmarks.publish(Path(directory), self.manifest)
            self.assertEqual(result["baseline_sha"], "landed")
            self.assertEqual(published.config["baseline_sha"], "landed")
            published.update.assert_called_once()

    def test_merge_handler_promotes_results_that_finished_first(self):
        run = MagicMock()
        run.config = self.manifest.copy()
        with (
            patch.object(benchmarks, "active_state", return_value=(self.manifest, "blob")),
            patch.object(benchmarks.wandb, "Api") as api,
            patch.object(benchmarks, "promote", return_value="landed"),
        ):
            api.return_value.runs.return_value = [run]
            benchmarks.reconcile(12)
            benchmarks.reconcile(12)
            self.assertEqual(run.config["baseline_sha"], "landed")
            run.update.assert_called_once()

    def test_results_from_different_attempts_are_rejected(self):
        jobs = [
            dict(name=f"production-benchmarks-{label}", conclusion="success", started_at="2026-11-20T12:00:00Z")
            for label in ("field", "ndarray")
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(benchmarks, "pages", return_value=iter(jobs)),
            patch.object(benchmarks, "artifact_files", side_effect=ValueError("earlier benchmark attempt")),
        ):
            with self.assertRaisesRegex(ValueError, "earlier benchmark attempt"):
                benchmarks.collect_results(Path(directory), self.manifest)

    def test_failed_mode_cannot_publish(self):
        jobs = [dict(name="production-benchmarks-field", conclusion="failure")]
        with tempfile.TemporaryDirectory() as directory, patch.object(benchmarks, "pages", return_value=iter(jobs)):
            with self.assertRaisesRegex(ValueError, "Both benchmark modes"):
                benchmarks.collect_results(Path(directory), self.manifest)

    def test_prepare_retry_keeps_original_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(
                self.manifest,
                genesis_sha="original",
                image_version="1_25",
                constraints="numpy==2.0",
                wheel=dict(run_id=20, artifact_id=30),
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            with (
                patch.object(benchmarks, "pages", return_value=iter([dict(name="benchmark-input")])),
                patch.object(benchmarks, "artifact_files"),
                patch.object(benchmarks, "active_state") as active,
                patch.dict(os.environ, GITHUB_OUTPUT=str(root / "outputs")),
            ):
                benchmarks.prepare(root, {}, 100, 2)
                saved = json.loads((root / "manifest.json").read_text())
                self.assertEqual(saved["genesis_sha"], "original")
                self.assertEqual(saved["record_id"], "100-2")
                active.assert_not_called()


if __name__ == "__main__":
    unittest.main()
