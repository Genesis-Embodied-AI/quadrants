"""Export per-mode timings and changes from the matching fast-math baseline."""

import argparse
import csv
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("inputs", type=Path, nargs="+")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows = []
for path in args.inputs:
    data = json.loads(path.read_text())
    baselines = {
        (record["arch"], record["precision"]): statistics.median(record["step_seconds"])
        for record in data["results"]
        if record["mode"] == "fast"
    }
    for record in data["results"]:
        samples = record["step_seconds"]
        median = statistics.median(samples)
        rows.append(
            {
                "arch": record["arch"],
                "precision": record["precision"],
                "mode": record["mode"],
                "n_envs": record["n_envs"],
                "steps_per_sample": record["timing_steps"],
                "median_ms": median * 1000,
                "min_ms": min(samples) * 1000,
                "max_ms": max(samples) * 1000,
                "change_percent": (median / baselines[record["arch"], record["precision"]] - 1) * 100,
            }
        )
with args.output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(
        f'{row["arch"]} fp{row["precision"]} {row["mode"]}: {row["median_ms"]:.6f} ms '
        f'[{row["min_ms"]:.6f}, {row["max_ms"]:.6f}], {row["change_percent"]:+.2f}%'
    )
