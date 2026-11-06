"""Print paired timings and export a CSV from run.py's results.json."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    assert all(row["exit_code"] == 0 for row in rows)
    pairs = sorted({row["pair"] for row in rows})
    indexed = {(row["pair"], row["variant"]): row for row in rows}
    columns = [
        "pair",
        "variant",
        "process_seconds",
        "suite_seconds",
        "ast_seconds",
        "outer_transforms",
        "all_transforms",
    ]
    with args.results.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for metric in ("process_seconds", "suite_seconds", "ast_seconds"):
        baseline = [indexed[(pair, "main")][metric] for pair in pairs]
        candidate = [indexed[(pair, "pr1076")][metric] for pair in pairs]
        savings = [a - b for a, b in zip(baseline, candidate)]
        percentages = [100 * (a - b) / a for a, b in zip(baseline, candidate)]
        print(metric)
        print(f"  main: {baseline}")
        print(f"  PR:   {candidate}")
        print(f"  medians: {statistics.median(baseline):.6f} -> {statistics.median(candidate):.6f} seconds")
        print(
            f"  median paired saving: {statistics.median(savings):.6f} seconds ({statistics.median(percentages):.3f}%)"
        )
        print(f"  paired saving range: {min(savings):.6f} to {max(savings):.6f} seconds")
        print(f"  PR faster in {sum(value > 0 for value in savings)}/{len(pairs)} pairs")


if __name__ == "__main__":
    main()
