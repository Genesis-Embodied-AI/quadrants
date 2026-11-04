"""Run the repository's format/lint hooks on changed implementation files and the common experiment harness."""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--entry-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    folder = root / "tmp/path-results" / ("lint-entry" if args.entry_only else "lint")
    folder.mkdir(parents=True, exist_ok=True)
    hook = str(root / ".venv/bin/pre-commit")
    for base in ("parent", "guard"):
        for kind in ("tuple", "list"):
            label = f"{base}-{kind}" + ("-entry" if args.entry_only else "")
            tree = root / "tmp" / f"experiment-{label}"
            files = ["python/quadrants/lang/_runtime_dataclass.py"]
            if base == "guard":
                files.append("python/quadrants/lang/_runtime_dataclass_guards.py")
            with (folder / f"{label}.log").open("w") as output:
                result = subprocess.run(
                    [hook, "run", "--files", *files], cwd=tree, stdout=output, stderr=subprocess.STDOUT
                )
            print(label, result.returncode, flush=True)
    tree = root / "quadrants"
    files = [str(p.relative_to(tree)) for p in (tree / "misc/experiments/dataclass_paths").glob("*.py")]
    files.append("tests/python/test_runtime_dataclass_paths.py")
    with (folder / "harness.log").open("w") as output:
        result = subprocess.run([hook, "run", "--files", *files], cwd=tree, stdout=output, stderr=subprocess.STDOUT)
    print("harness", result.returncode, flush=True)


if __name__ == "__main__":
    main()
