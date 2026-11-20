"""Exercise benchmark lifecycle decisions without GitHub or W&B writes."""

# Currently run manually, not as part of CI:
# python -m pytest .github/scripts/tests/test_genesis_benchmarks.py -v

import base64
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import genesis_benchmarks as benchmarks


@pytest.fixture
def manifest():
    return dict(
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


def test_baseline_uses_main_order_and_excludes_candidate(manifest):
    older = dict(manifest, record_id="90-1", run_id=90, baseline_sha="older")
    newer = dict(manifest, record_id="80-1", run_id=80, baseline_sha="newer")
    candidate = dict(manifest, baseline_sha="candidate")
    wrong_environment = dict(newer, compatibility_key="other", baseline_sha="tip")
    records = [older, candidate, newer, wrong_environment]
    history = ["tip", "candidate", "newer", "older"]
    assert benchmarks.select_baseline(manifest, records, history, "candidate") == newer
    assert benchmarks.select_baseline(manifest, [candidate], history, "candidate") is None


def test_merge_requires_head_and_tree_match(manifest):
    pull = dict(merged=True, head=dict(sha="head"), base=dict(ref="main"), merge_commit_sha="landed")
    with patch.object(benchmarks, "github", side_effect=[pull, dict(tree=dict(sha="tree"))]):
        assert benchmarks.promote(manifest) == "landed"
    with patch.object(benchmarks, "github", side_effect=[pull, dict(tree=dict(sha="different"))]):
        assert benchmarks.promote(manifest) is None
    with patch.object(benchmarks, "github", return_value=dict(pull, head=dict(sha="superseded"))):
        assert benchmarks.promote(manifest) is None
    with patch.object(benchmarks, "github", return_value=dict(pull, merged=False)):
        assert benchmarks.promote(manifest) is None
    with patch.object(benchmarks, "github", return_value=dict(pull, base=dict(ref="stack"))):
        assert benchmarks.promote(manifest) is None


def test_activation_rejects_stale_request_and_is_idempotent(manifest):
    with (
        patch.object(benchmarks, "active_state", return_value=(dict(activation="101"), "blob")),
        patch.object(benchmarks, "github") as github,
    ):
        with pytest.raises(ValueError, match="superseded"):
            benchmarks.activate(manifest)
        github.assert_not_called()
    with (
        patch.object(benchmarks, "active_state", return_value=(manifest, "blob")),
        patch.object(benchmarks, "github") as github,
    ):
        benchmarks.activate(manifest)
        github.assert_not_called()


def test_activation_updates_pointer_with_expected_blob(manifest, tmp_path):
    with (
        patch.dict(os.environ, GITHUB_STEP_SUMMARY=str(tmp_path / "summary")),
        patch.object(benchmarks, "active_state", return_value=(dict(activation="99", record_id="99-1"), "old-blob")),
        patch.object(benchmarks, "github") as github,
    ):
        benchmarks.activate(manifest)
        payload = github.call_args.args[1]
        assert payload["sha"] == "old-blob"
        assert json.loads(base64.b64decode(payload["content"])) == manifest


def test_merge_during_publication_is_promoted(manifest, tmp_path):
    with (
        patch.object(benchmarks, "collect_results", return_value=manifest),
        patch.object(benchmarks, "promote", side_effect=[None, "landed"]),
        patch.object(benchmarks.wandb, "init") as initialize,
        patch.object(benchmarks.wandb, "Artifact"),
        patch.object(benchmarks.wandb, "Api") as api,
    ):
        run = initialize.return_value.__enter__.return_value
        run.log_artifact.return_value.wait.return_value.name = "measurements-100-1:v0"
        published = api.return_value.run.return_value
        published.config = {}
        result = benchmarks.publish(tmp_path, manifest)
        assert result["baseline_sha"] == "landed"
        assert published.config["baseline_sha"] == "landed"
        published.update.assert_called_once()


def test_merge_handler_promotes_results_that_finished_first(manifest):
    run = MagicMock()
    run.config = manifest.copy()
    with (
        patch.object(benchmarks, "active_state", return_value=(manifest, "blob")),
        patch.object(benchmarks.wandb, "Api") as api,
        patch.object(benchmarks, "promote", return_value="landed"),
    ):
        api.return_value.runs.return_value = [run]
        benchmarks.reconcile(12)
        benchmarks.reconcile(12)
        assert run.config["baseline_sha"] == "landed"
        run.update.assert_called_once()


def test_results_from_different_attempts_are_rejected(manifest, tmp_path):
    jobs = [
        dict(name=f"production-benchmarks-{label}", conclusion="success", started_at="2026-11-20T12:00:00Z")
        for label in ("field", "ndarray")
    ]
    with (
        patch.object(benchmarks, "pages", return_value=iter(jobs)),
        patch.object(benchmarks, "artifact_files", side_effect=ValueError("earlier benchmark attempt")),
    ):
        with pytest.raises(ValueError, match="earlier benchmark attempt"):
            benchmarks.collect_results(tmp_path, manifest)


def test_failed_mode_cannot_publish(manifest, tmp_path):
    jobs = [dict(name="production-benchmarks-field", conclusion="failure")]
    with patch.object(benchmarks, "pages", return_value=iter(jobs)):
        with pytest.raises(ValueError, match="Both benchmark modes"):
            benchmarks.collect_results(tmp_path, manifest)


def test_prepare_retry_keeps_original_configuration(manifest, tmp_path):
    manifest = dict(
        manifest,
        genesis_sha="original",
        image_version="1_25",
        constraints="numpy==2.0",
        wheel=dict(run_id=20, artifact_id=30),
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with (
        patch.object(benchmarks, "pages", return_value=iter([dict(name="benchmark-input")])),
        patch.object(benchmarks, "artifact_files"),
        patch.object(benchmarks, "active_state") as active,
        patch.dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "outputs")),
    ):
        benchmarks.prepare(tmp_path, {}, 100, 2)
        saved = json.loads((tmp_path / "manifest.json").read_text())
        assert saved["genesis_sha"] == "original"
        assert saved["record_id"] == "100-2"
        active.assert_not_called()


def test_comparison_retry_reuses_complete_published_attempt(manifest, tmp_path):
    published = dict(manifest, artifact="project/measurements:v0")
    run = MagicMock()
    run.config = published
    with (
        patch.object(benchmarks, "collect_results", return_value=None),
        patch.object(benchmarks.wandb, "Api") as api,
        patch.object(benchmarks.wandb, "init") as initialize,
    ):
        api.return_value.runs.return_value = [run]
        result = benchmarks.publish(tmp_path, dict(manifest, attempt=2, record_id="100-2"))
        assert result["record_id"] == "100-1"
        initialize.assert_not_called()
        assert json.loads((tmp_path / "manifest.json").read_text()) == published


def test_complete_modes_produce_a_bootstrap_key(manifest, tmp_path):
    jobs = [
        dict(name=f"production-benchmarks-{label}", conclusion="success", started_at="now")
        for label in ("field", "ndarray")
    ]
    for mode in ("0", "1"):
        for name in ("speed-test", "mem-test", "benchmark-environment"):
            (tmp_path / f"{name}-{mode}").mkdir()
        (tmp_path / f"speed-test-{mode}" / "speed_test.txt").write_text(
            f"env=anymal | dtype={mode} | runtime_fps=100 | compile_time=10 | realtime_factor=1\n"
        )
        (tmp_path / f"mem-test-{mode}" / "mem_test.txt").write_text(f"env=anymal | dtype={mode} | max_mem_mb=100\n")
        (tmp_path / f"benchmark-environment-{mode}" / "environment.json").write_text(
            json.dumps(dict(packages=["numpy==2.0"], image_version="1_25"))
        )
    with patch.object(benchmarks, "pages", return_value=iter(jobs)), patch.object(benchmarks, "artifact_files"):
        result = benchmarks.collect_results(tmp_path, dict(manifest, bootstrap=True))
        assert result["baseline_sha"] == "tested"
        assert result["constraints"] == "numpy==2.0"
        assert len(result["compatibility_key"]) == 64
    with patch.object(benchmarks, "pages", return_value=iter(jobs)), patch.object(benchmarks, "artifact_files"):
        with pytest.raises(ValueError, match="environment changed"):
            benchmarks.collect_results(tmp_path, manifest)
