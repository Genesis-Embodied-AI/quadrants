#!/usr/bin/env python3
"""Builds the pre-compiled fatbins for the graph_do_while condition kernel.

The output is a C header (graph_do_while_cond_fatbin.h) containing one fatbin per build toolkit as byte arrays, which is
compiled into the quadrants binary. At runtime this lets us load the condition kernel without needing libcudadevrt.a on
the user's system.

Each SM architecture is built with the oldest toolkit that supports it (sm_110 needs CUDA 13.0; the rest use CUDA 12.8),
so all required toolkits must be installed -- otherwise the script raises `MissingToolkitError`. See _fatbin_common.py
for the rationale.

Usage:
    python scripts/build_condition_kernel_fatbin.py

See docs/source/user_guide/building_cudagraph_conditional_fatbin.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fatbin_common import build_fatbin_header  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "quadrants" / "runtime" / "cuda" / "graph_do_while_cond.cu"
OUT_HEADER = REPO_ROOT / "quadrants" / "runtime" / "cuda" / "graph_do_while_cond_fatbin.h"

SM_VERSIONS = [90, 100, 110, 120]


def main() -> None:
    build_fatbin_header(
        script_name=Path(__file__).name,
        src=SRC,
        out_header=OUT_HEADER,
        sm_versions=SM_VERSIONS,
        base_name="kConditionKernelFatbin",
    )


if __name__ == "__main__":
    main()
