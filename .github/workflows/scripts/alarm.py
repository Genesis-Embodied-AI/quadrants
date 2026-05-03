"""Benchmark regression alarm for Quadrants interop benchmarks.

Compares current benchmark results against historical W&B data and produces
a Markdown report highlighting regressions and anomalies.

Invoked by the ``benchmark_alarm.yml`` workflow after benchmark artifacts
have been downloaded.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import wandb
from frozendict import frozendict
from wandb.apis.public import Run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kv_str_to_fdict(kv_str: str) -> frozendict[str, str]:
    kv: dict[str, str] = {}
    if kv_str:
        for token in kv_str.split("-"):
            token = token.strip()
            if token and "=" in token:
                k, v = token.split("=", 1)
                kv[k.strip()] = v.strip()
    return frozendict(kv)


def _merge_key_order(tuples: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    merged: list[str] = list(tuples[-1])
    seen = set(merged)
    for t in tuples[:-1]:
        for key in t:
            if key not in seen:
                merged.append(key)
                seen.add(key)
    return tuple(merged)


def _fmt(v: float, is_int: bool) -> str:
    if v != v:
        return "NaN"
    return f"{int(v):,}" if is_int else f"{v:.2f}"


class _SortKey:
    def __init__(self, param_names: Iterable[str]) -> None:
        self._names = list(param_names)

    def __call__(self, d: frozendict[str, Any]) -> list[tuple[bool, Any]]:
        return [(d.get(c) is None, d.get(c)) for c in self._names]


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------


def parse_results(path: Path, metric_keys: tuple[str, ...]) -> dict[frozendict[str, str], dict[str, float]]:
    results: dict[frozendict[str, str], dict[str, float]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        kv: dict[str, str] = {}
        for part in line.replace(" \t| ", "|").split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip()] = v.strip()
        metrics: dict[str, float] = {}
        for k in metric_keys:
            try:
                metrics[k] = float(kv.pop(k))
            except (ValueError, TypeError, KeyError):
                pass
        results[frozendict(kv)] = metrics
    return results


# ---------------------------------------------------------------------------
# Alarm
# ---------------------------------------------------------------------------


class Alarm:
    def __init__(self, args: argparse.Namespace) -> None:
        self.max_valid = args.max_valid_revisions
        self.max_fetch = args.max_fetch_revisions
        self.wandb_entity = os.environ["WANDB_ENTITY"]
        self.wandb_project = args.wandb_project
        self.metric_keys: tuple[str, ...] = tuple(args.metrics)
        self.tolerances = dict(zip(args.metrics, args.tolerances))
        self.signs = dict(zip(args.metrics, args.signs))
        self.artifacts_dir = Path(args.artifacts_dir).expanduser().resolve()
        self.check_body_path = Path(args.check_body_path).expanduser()
        self.csv_dir = Path(args.csv_dir).expanduser().resolve()
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.exit_regression = args.exit_code_regression
        self.exit_alert = args.exit_code_alert
        self.run_prefix = args.run_prefix

    def load_current(self) -> "CurrentResults":
        results: dict[frozendict[str, str], dict[str, float]] = {}
        paths = list(self.artifacts_dir.rglob("*.txt"))
        assert paths, f"No .txt files found in {self.artifacts_dir}"
        for p in paths:
            results |= parse_results(p, self.metric_keys)
        param_names = _merge_key_order(tuple(tuple(kv.keys()) for kv in results.keys()))
        return CurrentResults(results=results, param_names=param_names, all_configs=frozenset(results.keys()))

    def fetch_wandb(self, current: "CurrentResults") -> dict[str, dict[frozendict[str, str], dict[str, float]]]:
        api = wandb.Api()
        runs_iter: Iterable[Run] = api.runs(f"{self.wandb_entity}/{self.wandb_project}", order="-created_at")

        hashes: set[str] = set()
        records: dict[str, dict[frozendict[str, str], dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

        for run in runs_iter:
            if self.run_prefix and not run.name.startswith(self.run_prefix):
                continue
            if len(hashes) >= self.max_fetch:
                break
            complete = sum(1 for r in records.values() if current.all_configs.issubset(r.keys()))
            if complete >= self.max_valid:
                break
            if run.state != "finished":
                continue
            try:
                config = run.config
                summary = run.summary
            except Exception:
                continue
            try:
                commit = config["revision"]
                hashes.add(commit)
            except (KeyError, TypeError):
                continue
            if len(records) >= self.max_valid and commit not in records:
                continue
            for k, v in summary.items():
                if k.startswith("_"):
                    continue
                metric_name, _, kv_str = k.partition("-")
                fdict = _kv_str_to_fdict(kv_str)
                records[commit][fdict][metric_name] = v
        return dict(records)

    def build_table(
        self,
        current: "CurrentResults",
        records: dict[str, dict[frozendict[str, str], dict[str, float]]],
        metric: str,
        alias: str,
    ) -> tuple[list[str], bool, bool]:
        sign = self.signs[metric]
        tol = self.tolerances[metric]
        reg_found = alert_found = False
        rows_md: list[str] = []
        rows_csv: list[dict[str, Any]] = []

        header = (
            "| status | "
            + " | ".join(current.param_names)
            + f" | current {alias} | baseline {alias} [last (mean +/- std)] | delta |"
        )
        align = "|:---:|" + "|".join(":---" for _ in current.param_names) + "|---:|---:|---:|"

        for cfg in sorted(current.results.keys(), key=_SortKey(current.param_names)):
            val = current.results[cfg].get(metric)
            if val is None:
                continue
            is_int = isinstance(val, int) or (isinstance(val, float) and val.is_integer())
            val_s = _fmt(val, is_int)
            params_s = [cfg.get(k, "-") for k in current.param_names]

            row_data: dict[str, Any] = {**dict(zip(current.param_names, params_s)), "current": val}

            prev = [r[cfg][metric] for r in records.values() if cfg in r and metric in r[cfg]]
            if prev:
                last = prev[0]
                mean = statistics.fmean(prev)
                delta = (val - last) / last * 100.0
                base_s = _fmt(last, is_int)
                delta_s = f"{delta:+.1f}%"

                if len(prev) >= self.max_valid:
                    ci95 = statistics.stdev(prev) / math.sqrt(len(prev)) * 1.96 if len(prev) > 1 else math.nan
                    base_s += f" ({_fmt(mean, is_int)} +/- {_fmt(ci95, is_int)})"
                    if sign * delta < -tol:
                        picto, delta_s = "🔴", f"**{delta_s}**"
                        reg_found = True
                        row_data["status"] = "regression"
                    elif sign * delta > tol:
                        picto, delta_s = "⚠️", f"**{delta_s}**"
                        alert_found = True
                        row_data["status"] = "alert"
                    else:
                        picto = "✅"
                        row_data["status"] = "ok"
                else:
                    picto = "ℹ️"
                    row_data["status"] = "n/a"
            else:
                picto = "ℹ️"
                base_s = delta_s = "---"
                row_data["status"] = "new"

            rows_md.append("| " + " | ".join((picto, *params_s, val_s, base_s, delta_s)) + " |")
            rows_csv.append(row_data)

        baselines = [f"- Commit {i}: {sha[:12]}" for i, sha in enumerate(records.keys(), 1)]
        footer = [f"**Baselines:** {len(records)} commits"] + baselines

        csv_path = self.csv_dir / f"{metric}.csv"
        if rows_csv:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows_csv[0].keys())
                w.writeheader()
                w.writerows(rows_csv)

        return [header, align, *rows_md, "", *footer], reg_found, alert_found


class CurrentResults:
    def __init__(
        self,
        results: dict[frozendict[str, str], dict[str, float]],
        param_names: tuple[str, ...],
        all_configs: frozenset[frozendict[str, str]],
    ) -> None:
        self.results = results
        self.param_names = param_names
        self.all_configs = all_configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadrants benchmark regression alarm")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing benchmark .txt files")
    parser.add_argument("--wandb-project", default="quadrants-interop-benchmarks")
    parser.add_argument("--run-prefix", default=None, help="Only compare W&B runs whose name starts with this")
    parser.add_argument("--metrics", nargs="+", required=True, help="Metric names to check")
    parser.add_argument("--tolerances", nargs="+", type=float, required=True, help="Tolerance pct per metric")
    parser.add_argument("--signs", nargs="+", type=int, required=True, help="1 if higher=better, -1 if lower=better")
    parser.add_argument("--max-valid-revisions", type=int, default=5)
    parser.add_argument("--max-fetch-revisions", type=int, default=40)
    parser.add_argument("--check-body-path", required=True)
    parser.add_argument("--csv-dir", default="/tmp/bench_csv")
    parser.add_argument("--exit-code-regression", type=int, default=42)
    parser.add_argument("--exit-code-alert", type=int, default=43)
    args = parser.parse_args()

    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY must be set"
    assert "WANDB_ENTITY" in os.environ, "WANDB_ENTITY must be set"
    assert (
        len(args.metrics) == len(args.tolerances) == len(args.signs)
    ), "--metrics, --tolerances, and --signs must have the same length"

    alarm = Alarm(args)
    current = alarm.load_current()
    records = alarm.fetch_wandb(current)

    reg_any = alert_any = False
    sections: list[str] = []
    thr_parts = [f"{m} +/- {alarm.tolerances[m]:.0f}%" for m in args.metrics]
    sections.append(f"Thresholds: {', '.join(thr_parts)}")
    sections.append("")

    for metric in args.metrics:
        alias = metric.replace("_", " ")
        table, reg, alert = alarm.build_table(current, records, metric, alias)
        reg_any |= reg
        alert_any |= alert
        sections.append(f"### {alias}")
        sections.extend(table)
        sections.append("")

    alarm.check_body_path.write_text("\n".join(sections) + "\n", encoding="utf-8")

    if reg_any:
        sys.exit(alarm.exit_regression)
    if alert_any:
        sys.exit(alarm.exit_alert)
    sys.exit(0)
