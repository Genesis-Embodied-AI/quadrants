#!/usr/bin/env python3
"""Generate and post-process the pybind11 type stub for ``quadrants_python``.

Runs ``pybind11-stubgen`` against the compiled ``quadrants._lib.core.quadrants_python`` extension (imported from
``--package-dir``, where ``quadrants`` may be a namespace package containing only the compiled artifacts), applies the
repo's YAML stub replacements, and writes ``quadrants_python.pyi`` + ``py.typed`` into ``--output-dir``.

This is invoked by the scikit-build-core CMake install step. The logic was lifted out of ``setup.py``
(``generate_pybind11_stubs`` / ``postprocess_stubs``) so a single implementation is shared by every build path.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

MODULE = "quadrants._lib.core.quadrants_python"
STUB_REL = pathlib.Path("quadrants/_lib/core/quadrants_python.pyi")


def postprocess_stubs(stub_path: pathlib.Path, repl_funcs_path: pathlib.Path, repl_global_path: pathlib.Path) -> None:
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(repl_funcs_path) as f:
        replacements_funcs = yaml.load(f)
    with open(repl_global_path) as f:
        replacements_global = yaml.load(f)

    new_lines = []
    for line in stub_path.read_text().split("\n"):
        func_name = line.lstrip().partition("(")[0]
        if func_name in replacements_funcs:
            line = replacements_funcs[func_name]
        for src, dst in replacements_global.items():
            line = line.replace(src, dst)
        new_lines.append(line)
    stub_path.write_text("\n".join(new_lines))


def generate(
    package_dir: pathlib.Path,
    output_dir: pathlib.Path,
    repo_root: pathlib.Path,
    stub_build_dir: pathlib.Path | None = None,
) -> None:
    package_dir = pathlib.Path(package_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    repo_root = pathlib.Path(repo_root).resolve()

    # Scratch dir for pybind11-stubgen's raw output. It MUST live outside the install tree: output_dir is inside the
    # wheel staging dir, and scikit-build-core sweeps everything under it into the wheel, so leaving scratch files there
    # would ship them.
    owns_scratch = stub_build_dir is None
    if owns_scratch:
        stub_build_dir = pathlib.Path(tempfile.mkdtemp(prefix="qd_stubgen_"))
    else:
        stub_build_dir = pathlib.Path(stub_build_dir).resolve()
        stub_build_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(package_dir) + os.pathsep + env.get("PYTHONPATH", "")

        # Invoke via `-m` on the current interpreter rather than the `pybind11-stubgen` console script, so it does not
        # depend on the venv's bin dir being on PATH.
        cmd = [sys.executable, "-m", "pybind11_stubgen", MODULE, "--ignore-all-errors", "-o", str(stub_build_dir)]
        print(" ".join(cmd), flush=True)
        subprocess.check_call(cmd, env=env)

        stub_file = stub_build_dir / STUB_REL
        postprocess_stubs(
            stub_file,
            repo_root / "stub_replacements_funcs.yaml",
            repo_root / "stub_replacements_global.yaml",
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(stub_file, output_dir / "quadrants_python.pyi")
        (output_dir / "py.typed").touch()
        print(f"wrote {output_dir / 'quadrants_python.pyi'} and py.typed", flush=True)
    finally:
        if owns_scratch:
            shutil.rmtree(stub_build_dir, ignore_errors=True)


def main() -> None:
    default_repo_root = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--package-dir", required=True, help="dir to put on PYTHONPATH so the compiled extension is importable"
    )
    p.add_argument("--output-dir", required=True, help="dir to write quadrants_python.pyi and py.typed into")
    p.add_argument("--repo-root", default=str(default_repo_root), help="repo root holding stub_replacements_*.yaml")
    p.add_argument("--stub-build-dir", default=None, help="scratch dir for pybind11-stubgen output")
    a = p.parse_args()
    generate(
        pathlib.Path(a.package_dir),
        pathlib.Path(a.output_dir),
        pathlib.Path(a.repo_root),
        pathlib.Path(a.stub_build_dir) if a.stub_build_dir else None,
    )


if __name__ == "__main__":
    main()
