"""Upload Quadrants interop benchmark results (JSON) to Weights & Biases.

Usage::

    python tests/python/upload_bench_to_wandb.py \
        --in-file results.json \
        --project quadrants-interop-benchmarks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8").strip()
    except Exception:
        return f"{uuid.uuid4().hex}@UNKNOWN"


def upload(in_file: str, project: str, run_prefix: str | None = None) -> None:
    import wandb

    revision = _git_revision()
    with open(in_file) as f:
        results = json.load(f)

    name = revision[:12]
    if run_prefix:
        name = f"{run_prefix}-{name}"

    run = wandb.init(
        project=project,
        name=name,
        config={"revision": revision},
        settings=wandb.Settings(x_disable_stats=True, console="off"),
    )

    for entry in results:
        bench = entry["benchmark"]
        arch = entry.get("arch", "unknown")
        n_envs = entry.get("n_envs", 0)
        tag = f"{bench}-arch={arch}-n_envs={n_envs}"

        run.log(
            {
                f"steps_per_sec-{tag}": entry["steps_per_sec"],
                f"us_per_step-{tag}": entry["us_per_step"],
            }
        )
        print(f"Uploaded {tag}: {entry['steps_per_sec']} steps/s, {entry['us_per_step']} us/step")

    run.finish()
    print(f"\nDone. {len(results)} benchmark results uploaded to {project}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Quadrants bench results to W&B")
    parser.add_argument("--in-file", required=True, help="Path to JSON results file")
    parser.add_argument("--project", default="quadrants-interop-benchmarks", help="W&B project name")
    parser.add_argument("--run-prefix", default=None, help="Prefix for W&B run name")
    args = parser.parse_args()
    upload(args.in_file, args.project, args.run_prefix)
