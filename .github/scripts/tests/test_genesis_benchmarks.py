"""Exercise benchmark lifecycle decisions without GitHub or W&B writes."""

# Currently run manually, not as part of CI:
# python -m pytest .github/scripts/tests/test_genesis_benchmarks.py -v

import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with patch.dict(
    os.environ,
    BENCHMARK_REPOSITORY="Genesis-Embodied-AI/quadrants",
    GENESIS_REPOSITORY="Genesis-Embodied-AI/genesis-world",
    BENCHMARK_PROJECT="genesis-ai-company/quadrants-genesis-benchmarks",
    BENCHMARK_STAGING_PROJECT="genesis-ai-company/quadrants-genesis-benchmarks-staging",
    BENCHMARK_STATE_BRANCH="genesis-benchmark-state",
    BENCHMARK_WHEEL_ARTIFACT="manylinux_wheel_cp312_ubuntu-22.04",
):
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
    other_genesis = dict(newer, genesis_sha="other", baseline_sha="tip")
    records = [older, candidate, newer, other_genesis]
    history = ["tip", "candidate", "newer", "older"]
    assert benchmarks.select_baseline(manifest, records, history, "candidate") == newer
    assert benchmarks.select_baseline(manifest, [candidate], history, "candidate") is None


def test_comparison_exports_baseline_and_provenance(manifest, tmp_path):
    record = dict(
        manifest, record_id="90-1", run_id=90, baseline_sha="older", artifact="measurements:v0", publication="chosen"
    )
    with (
        patch.object(benchmarks.subprocess, "run"),
        patch.object(benchmarks.subprocess, "check_output", return_value="main\nolder\n"),
        patch.object(benchmarks, "github", return_value=dict(merged=False)),
        patch.object(benchmarks.wandb, "Api") as api,
        patch.dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "outputs"), GITHUB_STEP_SUMMARY=str(tmp_path / "summary")),
    ):
        api.return_value.runs.return_value = [MagicMock(config=record)]
        benchmarks.prepare_comparison(dict(manifest, genesis_ref="v1.0"))
        outputs = (tmp_path / "outputs").read_text()
        assert "wandb_filters<<" in outputs
        assert '"config.publication": "chosen"' in outputs
        assert '"config.genesis_sha": "genesis"' in outputs
        summary = (tmp_path / "summary").read_text()
        assert "Genesis: `v1.0` (`genesis`)" in summary
        assert "Quadrants candidate: `tested`" in summary
        api.return_value.artifact.assert_not_called()


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
        patch.object(benchmarks, "promote", return_value="landed"),
        patch.object(benchmarks, "store_measurements", return_value=dict(manifest, artifact="staging:v0")) as store,
        patch.object(benchmarks, "publish_baseline") as publish_baseline,
    ):
        result = benchmarks.publish(tmp_path, manifest)
        assert result["baseline_sha"] == "landed"
        assert store.call_args.args[2] == benchmarks.STAGING_PROJECT
        assert publish_baseline.call_args.args[1]["baseline_sha"] == "landed"


def test_merge_handler_promotes_results_that_finished_first(manifest):
    run = MagicMock(config=dict(manifest, artifact="staging:v0"))
    with (
        patch.object(benchmarks.wandb, "Api") as api,
        patch.object(benchmarks, "promote", return_value="landed"),
        patch.object(benchmarks, "publish_baseline") as publish_baseline,
    ):
        api.return_value.runs.return_value = [run]
        benchmarks.reconcile(12)
        assert api.return_value.runs.call_args.args == (benchmarks.STAGING_PROJECT,)
        assert publish_baseline.call_args.args[1]["baseline_sha"] == "landed"
        api.return_value.artifact.assert_called_once_with("staging:v0")


def test_baseline_publication_is_idempotent(manifest, tmp_path):
    with patch.object(benchmarks.wandb, "Api") as api, patch.object(benchmarks, "store_measurements") as store:
        api.return_value.runs.return_value = [MagicMock()]
        benchmarks.publish_baseline(tmp_path, manifest)
        store.assert_not_called()
        api.return_value.runs.return_value = []
        benchmarks.publish_baseline(tmp_path, manifest)
        store.assert_called_once_with(tmp_path, manifest, benchmarks.PROJECT)


@pytest.mark.parametrize("from_record", [False, True])
def test_baseline_copies_bytes_and_publishes_summaries_before_record(manifest, tmp_path, from_record):
    if from_record:
        manifest = dict(manifest, kind="record", artifact="staging:v0")
    for kind, data in (("speed", "runtime_fps=100 | compile_time=10 | realtime_factor=1"), ("mem", "max_mem_mb=100")):
        for mode in (0, 1):
            directory = tmp_path / f"{kind}-test-{mode}"
            directory.mkdir()
            (directory / f"{kind}_test.txt").write_text(f"env=anymal | dtype={mode} | {data}\n")
    with patch.object(benchmarks.wandb, "init") as initialize, patch.object(benchmarks.wandb, "Artifact") as artifact:
        run = initialize.return_value.__enter__.return_value
        run.log_artifact.return_value.wait.return_value.name = "copy:v0"
        result = benchmarks.store_measurements(tmp_path, dict(manifest, baseline_sha="landed"), benchmarks.PROJECT)
        configs = [call.kwargs for call in initialize.call_args_list]
        assert [config["project"] for config in configs] == ["quadrants-genesis-benchmarks"] * 3
        assert configs[0]["name"].startswith("speed-")
        assert configs[1]["name"].startswith("mem-")
        assert all("kind" not in config["config"] for config in configs[:2])
        assert all(config["config"].get("artifact") != "staging:v0" for config in configs)
        assert configs[2]["config"]["kind"] == "record"
        assert all(config["config"]["revision"] == "landed@Genesis-Embodied-AI/quadrants/main" for config in configs)
        assert len({config["config"]["publication"] for config in configs}) == 1
        summaries = [call.args[0] for call in run.summary.update.call_args_list]
        assert summaries[0]["runtime_fps-dtype=0-env=anymal"] == 100
        assert summaries[1]["max_mem_mb-dtype=1-env=anymal"] == 100
        artifact.return_value.add_dir.assert_called_once_with(str(tmp_path))
        assert result["artifact"] == f"{benchmarks.PROJECT}/copy:v0"


def test_failed_summary_cannot_publish_baseline_record(manifest, tmp_path):
    with patch.object(benchmarks.wandb, "init") as initialize:
        with pytest.raises(ValueError, match="Missing speed"):
            benchmarks.store_measurements(tmp_path, manifest, benchmarks.PROJECT)
        initialize.assert_not_called()


@pytest.mark.parametrize("timestamp_format", ["naive", "offset", "z"])
def test_cleanup_deletes_only_expired_staging_runs_and_artifacts(timestamp_format):
    now = datetime.now(timezone.utc)
    if timestamp_format == "naive":
        now = now.replace(tzinfo=None)
    old = MagicMock(created_at=(now - timedelta(days=8)).isoformat())
    recent = MagicMock(created_at=(now - timedelta(days=6)).isoformat())
    if timestamp_format == "z":
        old.created_at = old.created_at.replace("+00:00", "Z")
        recent.created_at = recent.created_at.replace("+00:00", "Z")
    with patch.object(benchmarks.wandb, "Api") as api:
        api.return_value.runs.return_value = [old, recent]
        benchmarks.cleanup()
        assert api.return_value.runs.call_args.args == (benchmarks.STAGING_PROJECT,)
        old.delete.assert_called_once_with(delete_artifacts=True)
        recent.delete.assert_not_called()


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
        patch.object(benchmarks, "promote", return_value=None),
    ):
        api.return_value.runs.return_value = [run]
        result = benchmarks.publish(tmp_path, dict(manifest, attempt=2, record_id="100-2"))
        assert result["record_id"] == "100-1"
        initialize.assert_not_called()
        assert json.loads((tmp_path / "manifest.json").read_text()) == published


def test_complete_modes_preserve_the_selected_genesis_revision(manifest, tmp_path):
    jobs = [
        dict(name=f"production-benchmarks-{label}", conclusion="success", started_at="now")
        for label in ("field", "ndarray")
    ]
    for mode in ("0", "1"):
        for name in ("speed-test", "mem-test"):
            (tmp_path / f"{name}-{mode}").mkdir()
        (tmp_path / f"speed-test-{mode}" / "speed_test.txt").write_text(
            f"env=anymal | dtype={mode} | runtime_fps=100 | compile_time=10 | realtime_factor=1\n"
        )
        (tmp_path / f"mem-test-{mode}" / "mem_test.txt").write_text(f"env=anymal | dtype={mode} | max_mem_mb=100\n")
    with patch.object(benchmarks, "pages", return_value=iter(jobs)), patch.object(benchmarks, "artifact_files"):
        result = benchmarks.collect_results(tmp_path, dict(manifest, bootstrap=True))
        assert result["baseline_sha"] == "tested"
        assert result["genesis_sha"] == manifest["genesis_sha"]


@pytest.mark.parametrize("synthetic_merge", [True, False])
def test_prepare_anchors_cutoff_to_tested_wheel_base(manifest, tmp_path, synthetic_merge):
    import hashlib

    commit = "merge" if synthetic_merge else "head"
    run = dict(
        id=20,
        run_attempt=1,
        event="pull_request",
        head_sha="head",
        conclusion="success",
        path=".github/workflows/manylinux_wheel.yml",
    )
    pull = dict(number=12, state="open", head=dict(sha="head"), base=dict(repo=dict(full_name=benchmarks.REPOSITORY)))
    metadata = dict(
        commit=commit, tree="tree", run_id=20, attempt=1, wheel="wheel.whl", sha256=hashlib.sha256(b"wheel").hexdigest()
    )
    active = dict(manifest)

    def download(run_id, name, directory):
        directory.mkdir(parents=True)
        (directory / "wheel.whl").write_bytes(b"wheel")
        (directory / "wheel.json").write_text(json.dumps(metadata))
        return 30

    def request(endpoint):
        if endpoint.endswith("/commits/main"):
            return dict(sha="new-main")
        if endpoint.endswith(f"/git/commits/{commit}"):
            parents = ["tested-base", "head"] if synthetic_merge else ["pr-parent"]
            return dict(tree=dict(sha="tree"), parents=[dict(sha=sha) for sha in parents])
        assert endpoint.endswith("/compare/new-main...head")
        return dict(merge_base_commit=dict(sha="tested-base"))

    with (
        patch.object(benchmarks, "pages", side_effect=[iter([]), iter([pull])]),
        patch.object(benchmarks, "active_state", return_value=(active, "blob")),
        patch.object(benchmarks, "artifact_files", side_effect=download),
        patch.object(benchmarks, "github", side_effect=request),
        patch.dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "outputs")),
    ):
        benchmarks.prepare(tmp_path, dict(workflow_run=run), 100, 1)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["main_cutoff"] == "tested-base"
    assert saved["wheel"]["commit"] == commit


@pytest.mark.parametrize("selected_sha", ["main", "older-main"])
def test_bootstrap_resolves_genesis_and_uses_latest_successful_main_wheel(tmp_path, selected_sha):
    wheel_run = dict(head_branch="main", event="push", conclusion="success", head_sha=selected_sha)
    wheel = dict(commit=selected_sha, artifact_id=30, run_id=20, base_commit=None)
    runs = [
        dict(wheel_run, conclusion=None, head_sha="main"),
        dict(wheel_run, conclusion="failure", head_sha="main"),
        dict(wheel_run, head_branch="feature"),
        dict(wheel_run, event="pull_request"),
        wheel_run,
        dict(wheel_run, head_sha="even-older-main"),
    ]

    def request(endpoint):
        if endpoint.endswith("/commits/main"):
            return dict(sha="main")
        if endpoint.endswith("/commits/release"):
            return dict(sha="selected-genesis")
        assert endpoint.endswith(f"/compare/{selected_sha}...main")
        return dict(status="identical" if selected_sha == "main" else "ahead")

    with (
        patch.object(benchmarks, "pages", side_effect=[iter([]), iter(runs)]) as pages,
        patch.object(benchmarks, "github", side_effect=request),
        patch.object(benchmarks, "validate_wheel", return_value=wheel) as validate,
        patch.dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "outputs")),
    ):
        benchmarks.prepare(tmp_path, dict(inputs=dict(genesis_ref="release")), 100, 1)
        assert pages.call_args.args == (
            f"repos/{benchmarks.REPOSITORY}/actions/workflows/manylinux_wheel.yml/runs?branch=main&status=success",
            "workflow_runs",
        )
        validate.assert_called_once_with(wheel_run, tmp_path / "wheel")
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["genesis_sha"] == "selected-genesis"
    assert saved["main_cutoff"] == selected_sha
    assert saved["head_sha"] == selected_sha
    assert saved["wheel"]["commit"] == selected_sha


@pytest.mark.parametrize("status", ["behind", "diverged"])
def test_bootstrap_rejects_wheel_outside_main_history(tmp_path, status):
    run = dict(head_branch="main", event="push", conclusion="success", head_sha="other")
    with (
        patch.object(benchmarks, "pages", side_effect=[iter([]), iter([run])]),
        patch.object(benchmarks, "github", side_effect=[dict(sha="main"), dict(sha="genesis"), dict(status=status)]),
        patch.object(benchmarks, "validate_wheel") as validate,
    ):
        with pytest.raises(ValueError, match="must belong to Quadrants main history"):
            benchmarks.prepare(tmp_path, dict(inputs=dict(genesis_ref="release")), 100, 1)
        validate.assert_not_called()


def test_bootstrap_requires_a_successful_main_wheel(tmp_path):
    with (
        patch.object(benchmarks, "pages", side_effect=[iter([]), iter([])]),
        patch.object(benchmarks, "github", side_effect=[dict(sha="main"), dict(sha="genesis")]),
    ):
        with pytest.raises(ValueError, match="Build a successful Manylinux wheel"):
            benchmarks.prepare(tmp_path, dict(inputs=dict(genesis_ref="release")), 100, 1)
