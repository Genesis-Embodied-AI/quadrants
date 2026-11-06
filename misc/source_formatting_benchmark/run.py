"""Compare pinned main and PR #1076 Python code using the same main CI wheel.

Run with the wheel's Python 3.12 environment and the repository's test dependencies installed.
Every measured subprocess starts fresh and runs through tests/run_tests.py, with disk compilation caches disabled.
"""

import argparse
import hashlib
import importlib
import json
import os
import runpy
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MAIN = "96f31f1083c7d45c2730b708198afb354ef33e16"
PR = "2a5e06e69d9efefc1bb31e0434fc918eba717fc8"
REPO = Path(__file__).resolve().parents[2]
FILES = ["classfunc", "class_method", "matrix", "loops", "fem99_headless"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO)


def prepare(wheel, output):
    changed = git("diff", "--name-only", MAIN, PR).decode().splitlines()
    assert all(path.startswith("python/quadrants/") and path.endswith(".py") for path in changed), changed
    for variant, commit in (("main", MAIN), ("pr1076", PR)):
        target = output / variant
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
        for path in changed:
            relative = Path(path).relative_to("python")
            # Verify the wheel's original Python code matches the pinned baseline, before applying PR source.
            assert (target / relative).read_bytes() == git("show", f"{MAIN}:{path}"), path
            (target / relative).write_bytes(git("show", f"{commit}:{path}"))
    return changed


def child(output, files):
    import quadrants as qd

    # Force both source-to-LLVM and offline caches off, including after the test fixture sets its cache env vars.
    original_init = qd.init

    def init(*args, **kwargs):
        os.environ["QD_OFFLINE_CACHE"] = "0"
        kwargs.update(offline_cache=False, src_ll_cache=False, cpu_max_num_threads=8)
        return original_init(*args, **kwargs)

    qd.init = init
    transform = importlib.import_module("quadrants.lang.ast.transform")
    original_transform = transform.transform_tree
    depth = 0
    metrics = {"ast_seconds": 0.0, "outer_transforms": 0, "all_transforms": 0}

    def timed_transform(*args, **kwargs):
        nonlocal depth
        outer = depth == 0
        if outer:
            start = time.perf_counter()
        depth += 1
        metrics["all_transforms"] += 1
        try:
            return original_transform(*args, **kwargs)
        finally:
            depth -= 1
            if outer:
                metrics["ast_seconds"] += time.perf_counter() - start
                metrics["outer_transforms"] += 1

    for name in ("quadrants.lang.ast", "quadrants.lang.ast.transform", "quadrants.lang.kernel", "quadrants.lang.func"):
        module = importlib.import_module(name)
        assert module.transform_tree is original_transform
        module.transform_tree = timed_transform

    sys.argv = [str(REPO / "tests/run_tests.py"), *files, "--arch", "cpu", "-t", "1", "--timeout", "0"]
    start = time.perf_counter()
    exit_code = 0
    try:
        runpy.run_path(str(REPO / "tests/run_tests.py"), run_name="__main__")
    except SystemExit as error:
        exit_code = error.code or 0
    finally:
        metrics.update(suite_seconds=time.perf_counter() - start, exit_code=exit_code, package=qd.__file__)
        output.write_text(json.dumps(metrics, indent=2))
        print("BENCH_METRICS", json.dumps(metrics), flush=True)
    raise SystemExit(exit_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--files", nargs="+", default=FILES)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child(args.output, args.files)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    changed = prepare(args.wheel, args.output)
    metadata = {
        "main": MAIN,
        "pr1076": PR,
        "harness": git("rev-parse", "HEAD").decode().strip(),
        "wheel": args.wheel.name,
        "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
        "python": sys.version,
        "files": args.files,
        "changed_python_files": changed,
        "cpu": subprocess.check_output(["lscpu"], text=True),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    records = []
    # One unmeasured warm-up pair, then alternating AB/BA measured pairs.
    for pair in range(-1, args.pairs):
        order = ("main", "pr1076") if pair % 2 == 0 else ("pr1076", "main")
        for variant in order:
            name = f"{'warmup' if pair < 0 else f'pair{pair}'}-{variant}"
            result_file = args.output / f"{name}.json"
            env = os.environ.copy()
            env.update(
                PYTHONPATH=f"{args.output / variant}:{REPO}",
                PYTHONHASHSEED="0",
                OMP_NUM_THREADS="8",
                OPENBLAS_NUM_THREADS="1",
                QD_WANTED_ARCHS="cpu",
                TMPDIR=str(args.output),
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--output",
                str(result_file),
                "--files",
                *args.files,
            ]
            print("RUN", name, flush=True)
            start = time.perf_counter()
            with (args.output / f"{name}.log").open("w") as log:
                process = subprocess.Popen(
                    command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
                )
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
            wall = time.perf_counter() - start
            if code:
                raise RuntimeError(f"{name} failed with exit {code}; inspect its full log")
            result = json.loads(result_file.read_text())
            result.update(pair=pair, variant=variant, process_seconds=wall)
            if pair >= 0:
                records.append(result)
                (args.output / "results.json").write_text(json.dumps(records, indent=2))
    for metric in ("process_seconds", "suite_seconds", "ast_seconds"):
        a = [r[metric] for r in records if r["variant"] == "main"]
        b = [r[metric] for r in records if r["variant"] == "pr1076"]
        print(
            f"{metric}: main median={statistics.median(a):.6f}, PR median={statistics.median(b):.6f}, "
            f"reduction={100 * (1 - statistics.median(b) / statistics.median(a)):.3f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
