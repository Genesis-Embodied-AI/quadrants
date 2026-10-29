"""Summarize downloaded artifacts from profile_amd_ci.yml without importing Quadrants."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


def startup_summary(path):
    measurements = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        item = json.loads(line)
        if "seconds" not in item:
            continue
        phase = item["phase"]
        phase = re.sub(r"cycle_[1-9][0-9]*\.", "warm_cycle.", phase)
        phase = re.sub(r"process_[0-9]+\.", "process.", phase)
        measurements[phase].append(item["seconds"])
    return {
        phase: {
            "count": len(values),
            "median_seconds": round(statistics.median(values), 6),
            "min_seconds": round(min(values), 6),
            "max_seconds": round(max(values), 6),
        }
        for phase, values in measurements.items()
    }


def pytest_summary(path):
    phases = defaultdict(list)
    cases = defaultdict(dict)
    outcome = None
    for line in path.read_text().splitlines():
        match = re.match(r"([0-9.]+)s\s+(setup|call|teardown)\s+(.+)", line)
        if match:
            duration, phase, nodeid = match.groups()
            phases[phase].append(float(duration))
            cases[nodeid][phase] = float(duration)
        if re.search(r"\bpassed\b.*\bin [0-9.]+s", line):
            outcome = line.strip("= ")
    return {
        "outcome": outcome,
        "phases": {
            phase: {
                "count": len(values),
                "sum_worker_seconds": round(sum(values), 3),
                "median_seconds": round(statistics.median(values), 3),
            }
            for phase, values in phases.items()
        },
        "slowest_cases": sorted(
            ({"nodeid": nodeid, **values} for nodeid, values in cases.items()),
            key=lambda case: sum(case.get(phase, 0) for phase in ("setup", "call", "teardown")),
            reverse=True,
        )[:10],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directories", nargs="+", type=Path)
    args = parser.parse_args()
    results = {}
    for directory in args.artifact_directories:
        result = {}
        for label in ("no-torch", "with-torch", "torch-disabled", "trace"):
            path = directory / f"{label}.log"
            if path.exists():
                result[label] = startup_summary(path)
        for workers in (1, 4):
            path = directory / f"pytest-{workers}.log"
            if path.exists():
                result[f"pytest-{workers}"] = pytest_summary(path)
        results[str(directory)] = result
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
