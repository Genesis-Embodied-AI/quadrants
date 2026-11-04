"""Prepare the pinned Genesis CI dependencies in an experiment-owned Python 3.12 environment."""

import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    environment = root / "fem-venv"
    override = root / "tmp/path-fem-overrides.txt"
    override.write_text(f"quadrants @ {args.wheel.resolve().as_uri()}\nnumpy==2.5.4\nscipy==1.18.1\npytest==9.1.1\n")
    source = (root / "genesis/.github/workflows/scripts/production_build.sh").read_text()
    source = source.replace("curl -LsSf https://astral.sh/uv/install.sh | sh\n", "")
    source = source.replace("/venv", str(environment))
    source = source.replace("uv pip install torch --", "uv pip install torch==2.13.0 --")
    generated = root / "tmp/path-fem-production-build.sh"
    generated.write_text(source)
    env = dict(os.environ, UV_OVERRIDE=str(override), UV_CACHE_DIR=str(root / "tmp/uv-cache"))
    subprocess.run(["bash", str(generated)], cwd=root / "genesis", env=env, check=True)
    with (root / "tmp/path-fem-packages.txt").open("w") as output:
        subprocess.run(["uv", "pip", "freeze", "--python", str(environment / "bin/python")], stdout=output, check=True)


if __name__ == "__main__":
    main()
