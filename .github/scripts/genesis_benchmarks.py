"""Manage Genesis benchmark selection, publication, promotion, and comparison."""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import wandb

REPOSITORY = os.environ["BENCHMARK_REPOSITORY"]
GENESIS = os.environ["GENESIS_REPOSITORY"]
PROJECT = os.environ["BENCHMARK_PROJECT"]
STAGING_PROJECT = os.environ["BENCHMARK_STAGING_PROJECT"]
STATE_BRANCH = os.environ["BENCHMARK_STATE_BRANCH"]
STATE_PATH = f"repos/{REPOSITORY}/contents/active.json"
WHEEL_ARTIFACT = os.environ["BENCHMARK_WHEEL_ARTIFACT"]


def github(endpoint, payload=None, method=None):
    command = ["gh", "api", endpoint]
    if payload is not None:
        command += ["--input", "-"]
    if method:
        command += ["--method", method]
    result = subprocess.run(
        command, input=json.dumps(payload) if payload is not None else None, text=True, capture_output=True
    )
    if result.returncode:
        raise RuntimeError(f"GitHub request failed: {endpoint}\n{result.stderr}")
    return json.loads(result.stdout) if result.stdout.strip() else None


def pages(endpoint, key=None):
    separator = "&" if "?" in endpoint else "?"
    for page in range(1, 1001):
        result = github(f"{endpoint}{separator}per_page=100&page={page}")
        values = result[key] if key else result
        yield from values
        if len(values) < 100:
            return
    raise RuntimeError(f"Pagination limit reached: {endpoint}")


def artifact_files(run_id, name, directory, started_after=None):
    artifacts = list(pages(f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts", "artifacts"))
    matches = [artifact for artifact in artifacts if artifact["name"] == name and not artifact["expired"]]
    if len(matches) != 1:
        raise ValueError(f"Run {run_id} needs exactly one available {name} artifact.")
    if started_after and matches[0]["created_at"] < started_after:
        raise ValueError(f"Artifact {name} belongs to an earlier benchmark attempt.")
    archive = subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}/actions/artifacts/{matches[0]['id']}/zip"])
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for member in bundle.infolist():
            path = Path(member.filename)
            if member.is_dir() or path.name != member.filename or path.name in ("", ".", ".."):
                raise ValueError(f"Unexpected artifact path: {member.filename}")
            (directory / path.name).write_bytes(bundle.read(member))
    return matches[0]["id"]


def active_state():
    # Listing the branch distinguishes an absent selection from authentication or service failures.
    branches = pages(f"repos/{REPOSITORY}/branches")
    if not any(branch["name"] == STATE_BRANCH for branch in branches):
        return None, None
    files = github(f"repos/{REPOSITORY}/contents?ref={STATE_BRANCH}")
    if not any(file["name"] == "active.json" for file in files):
        return None, None
    file = github(f"{STATE_PATH}?ref={STATE_BRANCH}")
    return json.loads(base64.b64decode(file["content"])), file["sha"]


def validate_wheel(run, directory):
    if run["path"] != ".github/workflows/manylinux_wheel.yml" or run["conclusion"] != "success":
        raise ValueError("A successful Manylinux wheel build is required.")
    artifact_id = artifact_files(run["id"], WHEEL_ARTIFACT, directory)
    manifests = list(directory.glob("*.json"))
    if len(manifests) != 1:
        raise ValueError("Build a wheel with source provenance before running Genesis benchmarks.")
    metadata = json.loads(manifests[0].read_text())
    if metadata["run_id"] != run["id"] or metadata["attempt"] != run["run_attempt"]:
        raise ValueError("Wheel metadata belongs to another run or attempt.")
    if Path(metadata["wheel"]).name != metadata["wheel"]:
        raise ValueError("Invalid wheel filename.")
    digest = hashlib.sha256((directory / metadata["wheel"]).read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("Wheel digest differs from its provenance.")
    commit = github(f"repos/{REPOSITORY}/git/commits/{metadata['commit']}")
    if commit["tree"]["sha"] != metadata["tree"]:
        raise ValueError("Wheel tree differs from its source commit.")
    base_commit = None
    if metadata["commit"] != run["head_sha"]:
        parents = [parent["sha"] for parent in commit["parents"]]
        if run["event"] != "pull_request" or len(parents) != 2 or parents[1] != run["head_sha"]:
            raise ValueError("Wheel source does not match the triggering commit or its PR merge checkout.")
        base_commit = parents[0]
    return {**metadata, "artifact_id": artifact_id, "base_commit": base_commit}


def prepare(directory, event, run_id, attempt):
    artifacts = list(pages(f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts", "artifacts"))
    if any(artifact["name"] == "benchmark-input" for artifact in artifacts):
        artifact_files(run_id, "benchmark-input", directory)
        manifest = json.loads((directory / "manifest.json").read_text())
    else:
        main_sha = github(f"repos/{REPOSITORY}/commits/main")["sha"]
        if "workflow_run" in event:
            active, _ = active_state()
            if active is None:
                raise ValueError("Baseline unavailable. Run Generate Genesis benchmark baseline first.")
            wheel_run = event["workflow_run"]
            pulls = list(pages(f"repos/{REPOSITORY}/commits/{wheel_run['head_sha']}/pulls"))
            pulls = [
                pull
                for pull in pulls
                if pull["head"]["sha"] == wheel_run["head_sha"] and pull["base"]["repo"]["full_name"] == REPOSITORY
            ]
            pulls.sort(key=lambda pull: pull["state"] != "open")
            if not pulls or (pulls[0]["state"] != "open" and not pulls[0]["merged_at"]):
                raise ValueError("The wheel must belong to an open or merged PR in Quadrants.")
            manifest = {**active, "pr": pulls[0]["number"], "main_cutoff": main_sha, "bootstrap": False}
        else:
            requested = event["inputs"]["genesis_ref"]
            genesis_sha = github(f"repos/{GENESIS}/commits/{quote(requested, safe='')}")["sha"]
            image = github(f"repos/{GENESIS}/contents/.github/genesis_image_ver?ref={genesis_sha}")
            runs = pages(
                f"repos/{REPOSITORY}/actions/workflows/manylinux_wheel.yml/runs?head_sha={main_sha}", "workflow_runs"
            )
            wheel_run = next(
                (
                    run
                    for run in runs
                    if run["head_branch"] == "main"
                    and run["event"] != "pull_request"
                    and run["conclusion"] == "success"
                ),
                None,
            )
            if wheel_run is None:
                raise ValueError(f"Build the Manylinux wheel on main ({main_sha}) before bootstrapping.")
            manifest = dict(
                genesis_ref=requested,
                genesis_sha=genesis_sha,
                main_cutoff=main_sha,
                pr=None,
                bootstrap=True,
                image_version=base64.b64decode(image["content"]).decode().strip(),
                activation=str(run_id),
                constraints="",
            )
        manifest["wheel"] = validate_wheel(wheel_run, directory / "wheel")
        manifest["head_sha"] = wheel_run["head_sha"]
        if not manifest["bootstrap"]:
            # A synthetic merge wheel includes its first parent; a head-only wheel includes only the shared history.
            manifest["main_cutoff"] = (
                manifest["wheel"]["base_commit"]
                or github(f"repos/{REPOSITORY}/compare/{main_sha}...{manifest['wheel']['commit']}")[
                    "merge_base_commit"
                ]["sha"]
            )
        if manifest["bootstrap"] and manifest["wheel"]["commit"] != main_sha:
            raise ValueError("Bootstrap wheel must represent the selected main commit.")
    manifest["run_id"] = run_id
    manifest["attempt"] = attempt
    manifest["record_id"] = f"{run_id}-{attempt}"
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    outputs = dict(
        genesis_sha=manifest["genesis_sha"],
        image_version=manifest["image_version"],
        wheel_run=manifest["wheel"]["run_id"],
        wheel_artifact=manifest["wheel"]["artifact_id"],
        constraints=manifest["constraints"],
        snapshot_exists=str(bool(artifacts and any(a["name"] == "benchmark-input" for a in artifacts))).lower(),
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in outputs.items():
            delimiter = hashlib.sha256(str(value).encode()).hexdigest()
            stream.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


def collect_results(directory, manifest):
    run_id = manifest["run_id"]
    jobs = list(pages(f"repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{manifest['attempt']}/jobs", "jobs"))
    if not any("production-benchmarks-" in job["name"] for job in jobs):
        return None
    for mode, label in (("0", "field"), ("1", "ndarray")):
        matches = [job for job in jobs if job["name"].endswith(f"production-benchmarks-{label}")]
        if len(matches) != 1 or matches[0]["conclusion"] != "success":
            raise ValueError(f"Both benchmark modes must succeed in attempt {manifest['attempt']}.")
        for name in (f"speed-test-{mode}", f"mem-test-{mode}", f"benchmark-environment-{mode}"):
            artifact_files(run_id, name, directory / name, started_after=matches[0]["started_at"])
    environments = [
        json.loads(next((directory / f"benchmark-environment-{mode}").glob("*.json")).read_text())
        for mode in ("0", "1")
    ]
    if environments[0] != environments[1]:
        raise ValueError("Field and ndarray environments differ.")
    environment = environments[0]
    key = hashlib.sha256(json.dumps([manifest["genesis_sha"], environment], sort_keys=True).encode()).hexdigest()
    if not manifest["bootstrap"] and key != manifest["compatibility_key"]:
        raise ValueError("Benchmark environment changed. Run manual bootstrap before activating it.")
    for kind, metrics in (("speed", ("runtime_fps", "compile_time")), ("mem", ("max_mem_mb",))):
        for mode in ("0", "1"):
            files = list((directory / f"{kind}-test-{mode}").glob(f"{kind}_test*.txt"))
            if len(files) != 1:
                raise ValueError(f"Missing {kind} measurements for mode {mode}.")
            rows = [
                dict(part.strip().split("=", 1) for part in line.split("|") if "=" in part)
                for line in files[0].read_text().splitlines()
                if line.strip()
            ]
            if not rows:
                raise ValueError("Empty benchmark result.")
            for row in rows:
                values = [float(row[metric].strip()) for metric in metrics]
                if any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError("Benchmark measurements must be positive and finite.")
    manifest.update(
        compatibility_key=key,
        environment=environment,
        constraints="\n".join(environment["packages"]),
        baseline_sha=None,
    )
    if manifest["bootstrap"]:
        manifest["baseline_sha"] = manifest["wheel"]["commit"]
    return manifest


def promote(manifest):
    if manifest["bootstrap"]:
        return manifest["baseline_sha"]
    pull = github(f"repos/{REPOSITORY}/pulls/{manifest['pr']}")
    if not pull["merged"] or pull["base"]["ref"] != "main" or pull["head"]["sha"] != manifest["head_sha"]:
        return None
    landed = github(f"repos/{REPOSITORY}/git/commits/{pull['merge_commit_sha']}")
    if landed["tree"]["sha"] != manifest["wheel"]["tree"]:
        return None
    return pull["merge_commit_sha"]


def store_measurements(directory, manifest, project):
    # Each publication has its own identity, so simultaneous promotion handlers cannot mix partial uploads.
    publication = uuid.uuid4().hex
    config = {**manifest, "publication": publication}
    entity, project_name = project.split("/")
    if project == PROJECT:
        config["revision"] = f"{manifest['baseline_sha']}@{REPOSITORY}/main"
        for kind, metrics in (("speed", ("runtime_fps", "compile_time", "realtime_factor")), ("mem", ("max_mem_mb",))):
            summary = {}
            for path in directory.glob(f"{kind}-test-*/*.txt"):
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = dict(part.strip().split("=", 1) for part in line.split("|") if "=" in part)
                    values = {metric: float(row.pop(metric)) for metric in metrics}
                    suffix = "-".join(f"{key}={value}" for key, value in sorted(row.items()))
                    summary.update({f"{metric}-{suffix}": value for metric, value in values.items()})
            if not summary:
                raise ValueError(f"Missing {kind} measurements for baseline publication.")
            with wandb.init(entity=entity, project=project_name, name=f"{kind}-{publication}", config=config) as run:
                run.summary.update(summary)
    # Publish the record last: selection must never see a partially published speed/memory baseline.
    with wandb.init(entity=entity, project=project_name, config={**config, "kind": "record"}) as run:
        artifact = wandb.Artifact(f"measurements-{publication}", type="genesis-benchmarks")
        artifact.add_dir(str(directory))
        logged = run.log_artifact(artifact).wait()
        config["artifact"] = f"{project}/{logged.name}"
        run.config.update({"artifact": config["artifact"]}, allow_val_change=True)
    return config


def publish_baseline(directory, manifest):
    existing = wandb.Api().runs(
        PROJECT,
        filters={"config.record_id": manifest["record_id"], "config.kind": "record", "state": "finished"},
    )
    if next(iter(existing), None) is None:
        # Upload local bytes into the baseline project; never reference an expiring staging artifact.
        store_measurements(directory, manifest, PROJECT)


def publish(directory, manifest):
    project = PROJECT if manifest["bootstrap"] else STAGING_PROJECT
    collected = collect_results(directory, manifest)
    if collected is None:
        # A comparison-only retry reuses one complete published attempt instead of combining old job artifacts.
        records = [
            dict(run.config)
            for run in wandb.Api().runs(
                project, filters={"config.run_id": manifest["run_id"], "config.kind": "record", "state": "finished"}
            )
            if run.config["genesis_sha"] == manifest["genesis_sha"] and run.config["wheel"] == manifest["wheel"]
        ]
        if not records:
            raise ValueError("No retained complete attempt. Rerun both benchmark modes before publication.")
        manifest = max(records, key=lambda record: record["attempt"])
        wandb.Api().artifact(manifest["artifact"]).download(root=str(directory))
    else:
        manifest = collected
        (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
        manifest = store_measurements(directory, manifest, project)
    # Check after publication: the merge handler covers earlier uploads, this covers merges during the upload.
    landed = promote(manifest)
    if landed and not manifest["bootstrap"]:
        manifest["baseline_sha"] = landed
        publish_baseline(directory, manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return manifest


def activate(manifest):
    current, file_sha = active_state()
    if current and int(current["activation"]) > manifest["run_id"]:
        raise ValueError("A newer bootstrap is active. This request has been superseded.")
    if current and current["record_id"] == manifest["record_id"]:
        return
    if current is None and not any(branch["name"] == STATE_BRANCH for branch in pages(f"repos/{REPOSITORY}/branches")):
        # The branch starts at a known main commit and is advanced only by normal contents commits.
        github(f"repos/{REPOSITORY}/git/refs", {"ref": f"refs/heads/{STATE_BRANCH}", "sha": manifest["main_cutoff"]})
    payload = dict(
        message=f"Activate Genesis {manifest['genesis_sha']} for Quadrants benchmarks.",
        branch=STATE_BRANCH,
        content=base64.b64encode(json.dumps(manifest, sort_keys=True).encode()).decode(),
    )
    if file_sha:
        payload["sha"] = file_sha
    github(STATE_PATH, payload, method="PUT")
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(f"Activated Genesis `{manifest['genesis_sha']}` with baseline `{manifest['baseline_sha']}`.\n")


def reconcile(pr):
    runs = wandb.Api().runs(STAGING_PROJECT, filters={"config.pr": pr, "config.kind": "record", "state": "finished"})
    for run in runs:
        manifest = dict(run.config)
        landed = promote(manifest)
        if landed:
            manifest["baseline_sha"] = landed
            with tempfile.TemporaryDirectory() as directory:
                wandb.Api().artifact(manifest["artifact"]).download(root=directory)
                publish_baseline(Path(directory), manifest)
            print(f"Promoted {run.id} to main baseline {landed}")


def cleanup():
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    # Materialize before deleting so pagination cannot skip runs as the result set shrinks.
    runs = list(wandb.Api().runs(STAGING_PROJECT, filters={"createdAt": {"$lt": cutoff.isoformat()}}))
    for run in runs:
        if datetime.fromisoformat(run.created_at.replace("Z", "+00:00")) < cutoff:
            run.delete(delete_artifacts=True)
            print(f"Deleted expired staging run {run.id}")


def select_baseline(manifest, records, history, landed=None):
    if landed in history:
        history = history[history.index(landed) + 1 :]
    positions = {sha: index for index, sha in enumerate(history)}
    candidates = [
        record
        for record in records
        if record.get("compatibility_key") == manifest["compatibility_key"]
        and record.get("baseline_sha") in positions
        and record["record_id"] != manifest["record_id"]
    ]
    return min(
        candidates,
        key=lambda record: (positions[record["baseline_sha"]], -int(record["run_id"]), -int(record["attempt"])),
        default=None,
    )


def prepare_comparison(manifest):
    subprocess.run(["git", "fetch", "origin", manifest["main_cutoff"]], check=True)
    history = subprocess.check_output(
        ["git", "rev-list", "--first-parent", manifest["main_cutoff"]], text=True
    ).splitlines()
    pull = github(f"repos/{REPOSITORY}/pulls/{manifest['pr']}")
    landed = pull["merge_commit_sha"] if pull["merged"] else None
    records = [
        dict(run.config)
        for run in wandb.Api().runs(
            PROJECT,
            filters={
                "config.compatibility_key": manifest["compatibility_key"],
                "config.kind": "record",
                "state": "finished",
            },
        )
    ]
    baseline = select_baseline(manifest, records, history, landed)
    if baseline is None:
        raise ValueError("Baseline unavailable. Run manual bootstrap or retry with an eligible baseline.")
    context = (
        f"Genesis: `{manifest['genesis_ref']}` (`{manifest['genesis_sha']}`)\n\n"
        f"Quadrants candidate: `{manifest['wheel']['commit']}`\n\n"
        f"Environment: `{manifest['compatibility_key']}`\n\n"
        f"Quadrants baseline: `{baseline['baseline_sha']}`\n\n"
        f"Main cutoff: `{manifest['main_cutoff']}`\n\n"
        f"[Baseline run](https://github.com/{REPOSITORY}/actions/runs/{baseline['run_id']})"
    )
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write(context + "\n")
    filters = json.dumps(
        {"config.publication": baseline["publication"], "config.compatibility_key": manifest["compatibility_key"]}
    )
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in dict(wandb_project=PROJECT.split("/")[1], wandb_filters=filters).items():
            delimiter = hashlib.sha256(value.encode()).hexdigest()
            stream.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "publish", "activate", "select", "reconcile", "cleanup"))
    parser.add_argument("--directory", type=Path, default=Path("benchmark-results"))
    args = parser.parse_args()
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args.directory, event, int(os.environ["GITHUB_RUN_ID"]), int(os.environ["GITHUB_RUN_ATTEMPT"]))
    elif args.command == "cleanup":
        cleanup()
    elif args.command == "reconcile":
        reconcile(event.get("number") or int(event["inputs"]["pr"]))
    else:
        manifest = json.loads((args.directory / "manifest.json").read_text())
        if args.command == "publish":
            publish(args.directory, manifest)
        elif args.command == "activate":
            activate(manifest)
        else:
            prepare_comparison(manifest)
