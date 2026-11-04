"""Extract complete FEM timings and paired changes from the preserved driver records and pytest logs."""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    folder = args.results / "fem"
    runs = json.loads((folder / "runs.json").read_text())
    for run in runs:
        log = (folder / f"{run['label']}-{run['trial']}.log").read_text()
        log = re.sub(r"\x1b\[[0-9;]*m", "", log)
        total = re.search(r"1 passed in ([\d.]+)s", log)
        run["pytest_seconds"] = float(total[1]) if total else None
        for phase in ("call", "setup", "teardown"):
            match = re.search(rf"([\d.]+)s {phase}\s+tests/deformable", log)
            run[phase + "_seconds"] = float(match[1]) if match else None
        print(run["label"], run["trial"], run["exit"], run["wall_seconds"], run["pytest_seconds"], run["call_seconds"])
    pairs = []
    for trial in sorted({run["trial"] for run in runs}):
        control = next(run for run in runs if run["trial"] == trial and run["label"] == "guard-control")
        candidate = next(run for run in runs if run["trial"] == trial and run["label"] != "guard-control")
        pair = {"trial": trial}
        for metric in ("wall_seconds", "pytest_seconds", "call_seconds"):
            pair[metric + "_change_percent"] = (candidate[metric] / control[metric] - 1) * 100
        pairs.append(pair)
        print(pair)
    (args.results / "fem-summary.json").write_text(json.dumps({"runs": runs, "pairs": pairs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
