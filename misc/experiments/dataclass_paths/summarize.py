"""Summarize recorded microbenchmark samples without treating historical timings as controls."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--entry-only", action="store_true")
    args = parser.parse_args()
    name = "micro-entry" if args.entry_only else "micro"
    folder = args.results / name
    trials = 6 if args.entry_only else 4
    variants = ("tuple-entry", "list-entry") if args.entry_only else ("tuple", "list", "elided")
    rows = []
    for base in ("parent", "guard"):
        controls = [json.loads((folder / f"{base}-control-{trial}.json").read_text()) for trial in range(trials)]
        print(f"\n{base}: microseconds per call; percentage change is the median of {trials} paired process ratios")
        for case in controls[0]["cases"]:
            for metric in ("schema", "dispatch"):
                baseline = [c["cases"][case][metric]["median_us"] for c in controls]
                row = {"snapshot": base, "case": case, "metric": metric, "baseline_medians_us": baseline}
                text = [f"{case}/{metric}: control {statistics.median(baseline):.3f}"]
                for variant in variants:
                    values = [
                        json.loads((folder / f"{base}-{variant}-{trial}.json").read_text())["cases"][case][metric][
                            "median_us"
                        ]
                        for trial in range(trials)
                    ]
                    deltas = [(value / original - 1) * 100 for value, original in zip(values, baseline)]
                    row[variant] = {
                        "medians_us": values,
                        "paired_changes_percent": deltas,
                        "median_change_percent": statistics.median(deltas),
                    }
                    text.append(f"{variant} {statistics.median(values):.3f} ({statistics.median(deltas):+.2f}%)")
                print("; ".join(text))
                rows.append(row)
    (args.results / f"{name}-summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
