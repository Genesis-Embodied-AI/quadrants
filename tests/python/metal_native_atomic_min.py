"""Minimal Metal native-float-atomic probe (no pytest).

Usage:
  QD_METAL_NATIVE_FLOAT_ATOMICS=1 QD_DUMP_MSL=1 python tests/python/metal_native_atomic_min.py

Prints whether a single f32 atomic_add into a scalar completes, and (with QD_DUMP_MSL=1)
dumps the SPIRV-Cross MSL to stderr before pipeline creation.
"""

from __future__ import annotations

import os
import sys
import traceback


def main() -> int:
    os.environ.setdefault("QD_DUMP_MSL", "1")
    print("env QD_METAL_NATIVE_FLOAT_ATOMICS=", os.environ.get("QD_METAL_NATIVE_FLOAT_ATOMICS"))
    print("env QD_DUMP_MSL=", os.environ.get("QD_DUMP_MSL"))

    import quadrants as qd

    qd.init(arch=qd.metal, offline_cache=False)
    print("init_ok")

    x = qd.field(dtype=qd.f32, shape=64)
    acc = qd.field(dtype=qd.f32, shape=())

    @qd.kernel
    def reduce():
        for i in range(64):
            qd.atomic_add(acc[None], x[i])

    for i in range(64):
        x[i] = 1.0
    acc[None] = 0.0

    print("launching reduce() ...", flush=True)
    try:
        reduce()
        qd.sync()
        print("WORKLOAD_OK acc=", float(acc[None]), flush=True)
        return 0
    except Exception as e:
        print("WORKLOAD_ERR:", type(e).__name__, e, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
