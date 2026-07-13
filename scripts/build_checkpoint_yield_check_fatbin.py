#!/usr/bin/env python3
"""Builds the pre-compiled fatbins for the qd.checkpoint() yield-check kernel.

The output is a C header (checkpoint_yield_check_fatbin.h) containing one fatbin per build toolkit as byte arrays, which
is compiled into the quadrants binary. Same pattern as the IF-gate fatbin and the graph_do_while condition fatbin -- the
user does not need libcudadevrt.a at runtime.

Each SM architecture is built with the oldest toolkit that supports it (sm_110 needs CUDA 13.0; the rest use CUDA 12.8),
so all required toolkits must be installed -- otherwise the script raises `MissingToolkitError`. See _fatbin_common.py
for the rationale.

Usage:
    python scripts/build_checkpoint_yield_check_fatbin.py

See docs/source/user_guide/building_cudagraph_conditional_fatbin.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fatbin_common import build_fatbin_header  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "quadrants" / "runtime" / "cuda" / "checkpoint_yield_check.cu"
OUT_HEADER = REPO_ROOT / "quadrants" / "runtime" / "cuda" / "checkpoint_yield_check_fatbin.h"

# Targets cover both pre-Hopper (Turing / Ampere / Ada Lovelace) and Hopper+ (Blackwell / Thor). The yield-check kernel
# itself only uses `atomicCAS` and direct pointer writes (no device runtime calls), so it builds fine on every SM target
# here. Pre-Hopper coverage is required because the qd.checkpoint pre-Hopper CUDA path (codegen prologue + flat graph)
# still wires the yield-check kernel as a regular kernel node, just inline after each yielding checkpoint's last body
# kernel instead of inside an IF conditional body. Hopper+ coverage is unchanged from before.
SM_VERSIONS = [75, 80, 86, 89, 90, 100, 110, 120]


def main() -> None:
    build_fatbin_header(
        script_name=Path(__file__).name,
        src=SRC,
        out_header=OUT_HEADER,
        sm_versions=SM_VERSIONS,
        base_name="kCheckpointYieldCheckKernelFatbin",
    )


if __name__ == "__main__":
    main()
