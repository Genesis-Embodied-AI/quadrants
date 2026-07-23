"""Env-gated profiler for the fastcache pruning phase (temporary, for compile-time investigation).

Enabled only when ``QD_PRUNE_PROFILE`` is set to a path prefix. Each process appends its accumulated
stats to ``{QD_PRUNE_PROFILE}_{pid}.json`` at interpreter exit, so an xdist run produces one file per
worker. Zero overhead (a few ``perf_counter`` calls per kernel) when the env var is unset, and it never
touches compilation results - it only reads ``pruning.used_vars_by_func_id`` / ``edges_by_call_site``.
"""

import atexit
import json
import os
import time

_PATH = os.environ.get("QD_PRUNE_PROFILE")
enabled = bool(_PATH)

# Accumulated over every cold kernel materialization in this process.
stats = {
    "n_kernels": 0,  # cold materializations (discovery + enforce passes both run)
    "discovery_s": 0.0,  # wall time of the pass-0 AST walk (records call edges / marks used)
    "fixpoint_s": 0.0,  # wall time of propagate_fixpoint() (0 on main - it has no such call)
    "enforce_s": 0.0,  # wall time of the pass-1 AST walk (prunes args)
    "backend_s": 0.0,  # wall time of prog.create_kernel() (LLVM/PTX/SASS codegen), both passes
    "params_kept": 0,  # sum over kernels of total flat params kept after prefix-collapse
    "n_edges": 0,  # sum over kernels of recorded call-site edges
    "n_pairs": 0,  # sum over kernels of (caller_arg, callee_param) forwarding pairs
}


class timer:
    """Context manager that adds elapsed wall time to ``stats[key]`` (only when enabled)."""

    __slots__ = ("key", "_t0")

    def __init__(self, key: str) -> None:
        self.key = key

    def __enter__(self):
        if enabled:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if enabled:
            stats[self.key] += time.perf_counter() - self._t0


def add(key: str, val) -> None:
    if enabled:
        stats[key] += val


def _dump() -> None:
    try:
        with open(f"{_PATH}_{os.getpid()}.json", "w") as f:
            json.dump(stats, f)
    except OSError:
        pass


if enabled:
    atexit.register(_dump)
