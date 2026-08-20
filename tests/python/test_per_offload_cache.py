"""Per-construct FRONTEND split (no reuse tier).

`compile_to_offloads` splits the frontend (simplify / merge_global_ptrs / offload) to run per top-level construct
instead of once over the whole kernel, isolating each construct by its backward slice, and falls back to the
whole-kernel path for kernels that are not recompute-safe (autodiff, mesh-for, or a local produced inside one
construct and consumed by another). This PR ships the split WITHOUT a reuse tier, so it recompiles every construct on
every compile; the reuse (a disk manifest keyed by `get_hashed_per_construct_cache_key`) is added by the cross-process
cache PR.

These tests assert the split's STRUCTURE and CORRECTNESS, not reuse:
 - the split fires for recompute-safe kernels and enumerates the expected constructs, with
   `frontend_constructs_recompiled == frontend_constructs_total` and `frontend_constructs_cache_hit == 0`;
 - kernels that are not recompute-safe fall back to the whole-kernel path (`frontend_constructs_total == -1`);
 - results stay numerically correct through the split-and-reassemble path.

Counts are exposed as `kernel._primal.per_offload_cache_observations`. `offline_cache=False` so the on-disk
whole-kernel cache never short-circuits codegen and the split always runs on the (cold) compile.
"""

import numpy as np

import quadrants as qd

from tests import test_utils

_N = 8
_C = (61001.0, 61002.0, 61003.0, 61004.0)


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_enumerates_all_constructs() -> None:
    # Four independent top-level loops => four task-emitting constructs. The kernel is recompute-safe, so the split
    # fires.
    @qd.kernel
    def kernel_a(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]
        for i in range(_N):
            x[i] += _C[2]
        for i in range(_N):
            x[i] += _C[3]

    arr = qd.ndarray(qd.f32, shape=(_N,))
    kernel_a(arr)

    obs = kernel_a._primal.per_offload_cache_observations
    # The split ran and enumerated all four constructs. With no reuse tier, every construct is recompiled and none is
    # a cache hit -- this pins down that the split is reuse-free by construction (the flip to hit > 0 is what the
    # cross-process cache PR asserts once it adds the disk manifest).
    assert obs.frontend_constructs_total == 4, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs

    # Numerical correctness: the reassembled per-construct tasks compute the same sum as the whole-kernel path would.
    assert np.allclose(arr.to_numpy(), sum(_C), atol=1.0), arr.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_correct_with_shared_serial_def() -> None:
    # A serial prologue defines `base`, consumed by two later loop constructs. The backward slice must recompute the
    # defining stores into each consuming construct (an operand-only slice would drop them and read zeros). Asserts the
    # split fires and stays numerically correct.
    @qd.kernel
    def kernel_shared(x: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        base = _C[0] + _C[1]
        for i in range(_N):
            x[i] += base
        for i in range(_N):
            y[i] += base + _C[2]

    x = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    kernel_shared(x, y)

    obs = kernel_shared._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs

    assert np.allclose(x.to_numpy(), _C[0] + _C[1], atol=1.0), x.to_numpy()
    assert np.allclose(y.to_numpy(), _C[0] + _C[1] + _C[2], atol=1.0), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_not_recompute_safe() -> None:
    # `s` is written INSIDE the first loop (one construct) and read by the second loop (another construct), so the
    # kernel is not recompute-safe: the split must fall back to the whole-kernel path and NOT run, leaving the
    # sentinel -1 on the observability surface.
    @qd.kernel
    def kernel_unsafe(x: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        s = 0.0
        for i in range(_N):
            s += x[i]
        for i in range(_N):
            y[i] = s

    x = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    x.from_numpy(np.arange(_N, dtype=np.float32))
    kernel_unsafe(x, y)

    obs = kernel_unsafe._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # Correctness of the whole-kernel fallback path.
    assert np.allclose(y.to_numpy(), float(np.arange(_N).sum()), atol=1e-2), y.to_numpy()
