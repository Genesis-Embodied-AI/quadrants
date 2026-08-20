"""Per-construct FRONTEND split (no reuse tier).

`compile_to_offloads` splits the frontend (simplify / merge_global_ptrs / offload) to run per top-level construct
instead of once over the whole kernel, isolating each construct by its backward slice (in
`transforms/split_frontend_per_construct.cpp`), and falls back to the whole-kernel path for kernels that are not
recompute-safe: autodiff, mesh-for, a loop-produced local shared across constructs, an effectful producer (e.g. a
local capturing a global atomic) a later construct would clone, or a serial field load that a later construct would
recompute after an intervening construct mutated global memory. This PR ships the split WITHOUT a
reuse tier, so it recompiles every construct on every compile; the reuse (a disk manifest keyed by
`get_hashed_per_construct_cache_key`) is added by the cross-process cache PR.

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


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_field_load_shadowed() -> None:
    # `base` snapshots x[0] BEFORE the first loop overwrites x[0]; a later construct consumes `base`. The backward
    # slice would recompute the `x[0]` load into that later construct, where it would observe the mutated value instead
    # of the source-order snapshot -- so the kernel is NOT recompute-safe and must fall back to the whole-kernel path.
    @qd.kernel
    def kernel_shadow(x: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        base = x[0]
        for i in range(_N):
            x[i] = 2.0
        for i in range(_N):
            y[i] = base

    x = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    x.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_shadow(x, y)

    obs = kernel_shadow._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # `base` must be the ORIGINAL x[0] (7.0), not the value the first loop wrote (2.0). A split that recomputed the
    # load after the store would wrongly yield 2.0.
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_carried_rmw_local() -> None:
    # Two constructs each read-modify-write the same local `s`, and the second also stores it. The second construct
    # depends on the value the first produced, so it is not recomputable per-construct (its slice would drop the first
    # loop and restart `s` from the serial init). Checking readers against the *union* of writer constructs would wrongly
    # accept this (both readers are also writers), so the gate must reject it and fall back.
    @qd.kernel
    def kernel_carry(out: qd.types.ndarray()) -> None:
        s = 1
        for i in range(1):
            s += 1
        for i in range(1):
            s += 1
            out[0] = s

    out = qd.ndarray(qd.i32, shape=(1,))
    kernel_carry(out)

    obs = kernel_carry._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # s: 1 -> 2 (loop A) -> 3 (loop B). A split that dropped loop A from B's slice would wrongly produce 2.
    assert int(out.to_numpy()[0]) == 3, out.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_effectful_producer() -> None:
    # `old` captures the return value of a global atomic, which is an EFFECT that emits its own task. A later construct
    # reads `old`; cloning its producer into that construct's backward slice would run the atomic a second time. The
    # gate must reject recomputing an effectful producer and fall back.
    @qd.kernel
    def kernel_atomic(counter: qd.types.ndarray(), out: qd.types.ndarray()) -> None:
        old = qd.atomic_add(counter[0], 1)
        for i in range(_N):
            out[i] = old

    counter = qd.ndarray(qd.i32, shape=(1,))
    out = qd.ndarray(qd.i32, shape=(_N,))
    kernel_atomic(counter, out)

    obs = kernel_atomic._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # The atomic must run exactly ONCE: counter == 1, and every out[i] is the pre-increment return value 0. A split that
    # cloned the atomic into the loop construct would increment counter twice and store 1.
    assert int(counter.to_numpy()[0]) == 1, counter.to_numpy()
    assert np.all(out.to_numpy() == 0), out.to_numpy()
