"""Run isolated path variants sequentially, recording source identities and complete child output.

Run from the experiment checkout in its cluster container. Fetch the experiment branches before invoking this script.
Examples:
    python misc/experiments/dataclass_paths/run.py ROOT tests
    python misc/experiments/dataclass_paths/run.py ROOT micro
    python misc/experiments/dataclass_paths/run.py ROOT fem --candidate tuple
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REFS = {
    "parent-control": "8102aca75",
    "parent-tuple": "bc5627b1a3975b29550ac7febdb467b74a67eb4c",
    "parent-list": "5de461c6a161c9517294d9b3cc16a42dcf8be2d7",
    "parent-elided": "f65e25189d0f8aa309e4561c42eb93801c3a8534",
    "guard-control": "8d6de6254",
    "guard-tuple": "d1f210c37b37e3671c65c22e1c2ad31a2c11c13d",
    "guard-list": "68e98eb29c78c487f0f4d3e8a63b8f085b9d90ee",
    "guard-elided": "3719f411499d28b4f827632dab5067e0ba7ec587",
}
REFS.update(
    {
        "parent-tuple-entry": "96c6416ac9c2d08c062460118fb82930905452d2",
        "parent-list-entry": "4212ada716a873039727576babf8f0ad8f098073",
        "guard-tuple-entry": "b734597279cf3ebee02bd6ce4e3d2552033b46a5",
        "guard-list-entry": "8d6959860a03c1ffd8a8304435da29cfce6396e9",
    }
)


def revision(repo, ref):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", ref], text=True).strip()


def checkout(root, label, refs):
    target = root / "tmp" / ("experiment-" + label)
    commit = refs[label]
    if not target.exists():
        subprocess.run(["git", "-C", str(root / "quadrants"), "worktree", "add", str(target), commit], check=True)
    assert revision(target, "HEAD") == commit, (target, commit)
    return target


def install(root, tree, environment):
    site = environment / "lib/python3.12/site-packages"
    for package in ("quadrants", "quadrants_pytest"):
        shutil.copytree(tree / "python" / package, site / package, dirs_exist_ok=True)
        for cache in (site / package).rglob("__pycache__"):
            shutil.rmtree(cache)
    modules = ["_runtime_dataclass.py", "_runtime_dataclass_guards.py", "_runtime_dataclass_final.py"]
    hashes = {}
    for name in modules:
        source = tree / "python/quadrants/lang" / name
        if source.exists():
            installed = site / "quadrants/lang" / name
            assert source.read_bytes() == installed.read_bytes()
            hashes[name] = hashlib.sha256(installed.read_bytes()).hexdigest()
    return {"commit": revision(tree, "HEAD"), "installed_path": str(site / "quadrants"), "hashes": hashes}


def run(command, cwd, env, logfile):
    start = time.perf_counter()
    print("Running", command, "log:", logfile, flush=True)
    with logfile.open("w") as output:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    record = {"command": command, "cwd": str(cwd), "exit": result.returncode, "wall_seconds": elapsed}
    print(json.dumps(record), flush=True)
    print("".join(logfile.read_text(errors="replace").splitlines(keepends=True)[-5:]), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=("tests", "micro", "fem", "types"))
    parser.add_argument("--candidate", choices=("tuple", "list", "tuple-entry", "list-entry"), default="tuple")
    parser.add_argument("--entry-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / "tmp/path-results" / (args.phase + ("-entry" if args.entry_only else ""))
    folder.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ, GS_ENABLE_NDARRAY="1", MADRONA_DISABLE_CUDA_HEAP_SIZE="1", QD_OFFLINE_CACHE_CLEANING_POLICY="never"
    )
    environment = root / ("fem-venv" if args.phase == "fem" else ".venv")
    python = str(environment / "bin/python")
    env["VIRTUAL_ENV"] = str(environment)
    env["PATH"] = str(environment / "bin") + os.pathsep + env["PATH"]
    refs = {label: revision(root / "quadrants", ref) for label, ref in REFS.items()}
    (folder / "refs.json").write_text(json.dumps(refs, indent=2) + "\n")
    records = []
    benchmark = Path(__file__).with_name("benchmark.py").resolve()
    if args.phase == "micro":
        orders = (
            ("control", "tuple", "list", "elided"),
            ("elided", "list", "tuple", "control"),
            ("tuple", "control", "elided", "list"),
            ("list", "elided", "control", "tuple"),
        )
        if args.entry_only:
            orders = (
                ("control", "tuple-entry", "list-entry"),
                ("list-entry", "tuple-entry", "control"),
                ("tuple-entry", "control", "list-entry"),
                ("list-entry", "control", "tuple-entry"),
                ("tuple-entry", "list-entry", "control"),
                ("control", "list-entry", "tuple-entry"),
            )
        schedule = [
            (f"{base}-{kind}", trial)
            for base in ("parent", "guard")
            for trial, order in enumerate(orders)
            for kind in order
        ]
    elif args.phase == "fem":
        schedule = [
            (f"guard-{kind}", trial)
            for trial, order in enumerate((("control", args.candidate), (args.candidate, "control")))
            for kind in order
        ]
    else:
        kinds = ("tuple-entry", "list-entry") if args.entry_only else ("control", "tuple", "list")
        schedule = [(f"{base}-{kind}", 0) for base in ("parent", "guard") for kind in kinds]
    try:
        for label, trial in schedule:
            tree = checkout(root, label, refs)
            identity = install(root, tree, environment)
            stem = f"{label}-{trial}"
            if args.phase == "micro":
                command = [
                    python,
                    str(benchmark),
                    "--label",
                    identity["commit"],
                    "--iterations",
                    "1000",
                    "--repeats",
                    "7",
                    "--output",
                    str(folder / f"{stem}.json"),
                ]
                cwd = root / "tmp"
            elif args.phase == "tests":
                command = [
                    python,
                    "tests/run_tests.py",
                    "test_runtime_typed_dataclass",
                    "test_runtime_dataclass_paths",
                    "test_py_dataclass",
                ]
                if label.startswith("parent"):
                    command.append("test_unused_runtime_dataclass")
                command += ["--arch", "cpu,cuda", "-t", "4", "-v"]
                cwd = tree
            elif args.phase == "types":
                command = [str(environment / "bin/pyright")]
                cwd = tree
            else:
                cache = folder / f"cache-{stem}"
                assert not cache.exists(), f"FEM run requires a fresh cache: {cache}"
                env["QD_OFFLINE_CACHE_FILE_PATH"] = str(cache)
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=30s",
                    "1300s",
                    python,
                    "-m",
                    "pytest",
                    "-v",
                    "-ra",
                    "--backend",
                    "gpu",
                    "--dev",
                    "--forked",
                    "-n",
                    "0",
                    "--durations=0",
                    "--durations-min=0",
                    "tests/deformable/test_fem.py::test_hard_constraint[64-False]",
                ]
                cwd = root / "genesis"
            record = run(command, cwd, env, folder / f"{stem}.log")
            records.append(dict(record, label=label, trial=trial, identity=identity))
            (folder / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
    finally:
        install(root, checkout(root, "parent-control", refs), environment)


if __name__ == "__main__":
    main()
