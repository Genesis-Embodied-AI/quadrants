"""Upload Quadrants interop benchmark results to Weights & Biases.

Reads the pipe-delimited text file produced by test_interop_benchmarks.py and
uploads each row as a W&B metric, matching the Genesis upload convention.

Usage::

    python tests/python/upload_bench_to_wandb.py \
        --in-file bench_interop.txt \
        --project quadrants-interop-benchmarks \
        --metrics steps_per_sec us_per_step
"""

from __future__ import annotations

import argparse
import subprocess
import uuid


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8").strip()
    except Exception:
        return f"{uuid.uuid4().hex}@UNKNOWN"


def _pprint_oneline(params: dict, delimiter: str = "-") -> str:
    return delimiter.join(f"{k}={v}" for k, v in sorted(params.items()))


def upload(in_file: str, project: str, metric_names: list[str] | None, run_prefix: str | None = None) -> None:
    import wandb

    revision = _git_revision()
    print(f"Uploading results to W&B project '{project}' for revision: {revision}")

    name = revision[:12]
    if run_prefix:
        name = f"{run_prefix}-{name}"

    run = wandb.init(
        project=project,
        name=name,
        config={"revision": revision},
        settings=wandb.Settings(x_disable_stats=True, console="off"),
    )

    uploaded = 0
    with open(in_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            params = {}
            for part in line.split(" \t| "):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k.strip()] = v.strip()

            if not params:
                continue

            if metric_names:
                metrics = {k: float(params.pop(k)) for k in metric_names if k in params}
            else:
                metrics = {}
                for k in list(params.keys()):
                    try:
                        metrics[k] = float(params[k])
                    except ValueError:
                        continue
                    del params[k]

            if not metrics:
                continue

            benchmark_id_suffix = _pprint_oneline(params)
            for metric_name, metric_value in metrics.items():
                benchmark_id = f"{metric_name}-{benchmark_id_suffix}"
                print(f"  {benchmark_id}: {metric_value}")
                run.log({benchmark_id: metric_value})

            uploaded += 1

    run.finish()
    print(f"\nDone. {uploaded} rows uploaded to {project}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Quadrants bench results to W&B")
    parser.add_argument("--in-file", required=True, help="Path to pipe-delimited results file")
    parser.add_argument("--project", default="quadrants-interop-benchmarks", help="W&B project name")
    parser.add_argument("--run-prefix", default=None, help="Prefix for W&B run name")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metric field names to upload")
    args = parser.parse_args()
    upload(args.in_file, args.project, args.metrics, args.run_prefix)
