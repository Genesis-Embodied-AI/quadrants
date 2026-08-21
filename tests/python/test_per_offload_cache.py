"""Per-construct FRONTEND split (no reuse tier).

`compile_to_offloads` splits the frontend (simplify / merge_global_ptrs / offload) to run per top-level construct
instead of once over the whole kernel, isolating each construct by its backward slice (in
`transforms/split_frontend_per_construct.cpp`), and falls back to the whole-kernel path for kernels that are not
recompute-safe: autodiff, mesh-for, a concurrently-executed region (`qd.stream_parallel()` / `qd.graph.parallel()`,
whose constructs share one global-temp buffer), a loop-produced local shared across constructs, a `qd.append` index a
later construct consumes, a non-recomputable producer a later construct would clone (an effectful one such as a global
atomic, a non-deterministic `qd.random()`, or a `qd.volatile_load()`), or a serial field load that a later construct
would recompute after an intervening effect (a store, atomic, or sparse activate/deactivate) mutated global state. It
also declines the split (as a diagnostic) when `QD_DUMP_CFG` asks for a whole-kernel CFG dump. This PR ships the split
WITHOUT a reuse tier, so it recompiles every construct on every compile; the reuse (a disk manifest keyed by a stable
per-construct cache key) is added by the cross-process cache PR.

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
def test_per_construct_frontend_split_observations_reset_on_cached_relaunch() -> None:
    # `per_offload_cache_observations` is a single per-Kernel attribute describing the MOST RECENT compile. A relaunch
    # served from a cached artifact runs no frontend split, so the counts must reset to the no-split sentinel (-1)
    # instead of leaking the previous split's counts. This exercises the in-process compiled-kernel reuse path; the
    # fastcache-restore path Codex flagged is the same code -- both hand `launch_kernel` a ready artifact and skip
    # `prog.compile_kernel`, so the reset covers both.
    @qd.kernel
    def kernel_relaunch(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]

    arr = qd.ndarray(qd.f32, shape=(_N,))
    kernel_relaunch(arr)

    obs = kernel_relaunch._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == 2, obs
    assert obs.frontend_constructs_cache_hit == 0, obs

    # Second launch reuses the compiled artifact (no split runs), so the observation reports the no-split sentinel
    # rather than the 2 constructs the first compile recorded.
    kernel_relaunch(arr)
    obs = kernel_relaunch._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    assert np.allclose(arr.to_numpy(), 2.0 * (_C[0] + _C[1]), atol=1.0), arr.to_numpy()


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


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_random_producer() -> None:
    # `r` is a single PRNG sample consumed by two later constructs. `RandStmt` has no global side effect, so it is not
    # a task effect, but it is NON-deterministic: recomputing it into each construct resamples and advances the PRNG
    # twice, so the gate must reject it and fall back.
    @qd.kernel
    def kernel_rand(x: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        r = qd.random()
        for i in range(_N):
            x[i] = r
        for i in range(_N):
            y[i] = r

    x = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    kernel_rand(x, y)

    obs = kernel_rand._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # Both loops must observe the SAME sample. If the split recomputed qd.random() per construct the two loops would get
    # different draws.
    assert np.allclose(x.to_numpy(), y.to_numpy()), (x.to_numpy(), y.to_numpy())


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_volatile_load() -> None:
    # `v` snapshots a VOLATILE load, consumed by two later constructs. A volatile load must be observed exactly once, in
    # place; recomputing it into each construct (as the backward slice would) turns one read of a concurrently-updated
    # cell into several and reorders it. The gate must reject recomputing a volatile load and fall back.
    @qd.kernel
    def kernel_volatile(flags: qd.types.ndarray(), x: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        v = qd.volatile_load(flags[0])
        for i in range(_N):
            x[i] = v
        for i in range(_N):
            y[i] = v

    flags = qd.ndarray(qd.f32, shape=(1,))
    x = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    flags.from_numpy(np.array([5.0], dtype=np.float32))
    kernel_volatile(flags, x, y)

    obs = kernel_volatile._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # Both loops observe the single volatile read (5.0).
    assert np.allclose(x.to_numpy(), 5.0), x.to_numpy()
    assert np.allclose(y.to_numpy(), 5.0), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_stream_parallel() -> None:
    # Loops inside separate `qd.stream_parallel()` blocks run concurrently on their own streams while sharing one
    # kernel context / global-temp buffer. Per-construct offload restarts global-temp offset allocation at 0 for each
    # construct, so concurrent constructs would alias the same offset and race. The two loops are otherwise
    # split-eligible (independent, recompute-safe), so the ONLY reason to fall back here is the concurrent region.
    @qd.kernel
    def kernel_streams(x: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        with qd.stream_parallel():
            for i in range(_N):
                x[i] += _C[0]
        with qd.stream_parallel():
            for j in range(_N):
                y[j] += _C[1]

    x = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    kernel_streams(x, y)

    obs = kernel_streams._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    assert np.allclose(x.to_numpy(), _C[0], atol=1.0), x.to_numpy()
    assert np.allclose(y.to_numpy(), _C[1], atol=1.0), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False, require=qd.extension.sparse)
def test_per_construct_frontend_split_fallback_sparse_deactivate_shadow() -> None:
    # `snap` snapshots x[0] before a construct deactivates x's sparse cells; a later construct consumes the snapshot.
    # A sparse deactivate mutates global structure (SNodeOpStmt), so recomputing the load into the later construct
    # would read the default 0 instead of the source-order snapshot. The gate must treat the deactivate as an
    # intervening effect and fall back.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    ptr = qd.root.pointer(qd.i, _N)
    ptr.dense(qd.i, 1).place(x)
    qd.root.dense(qd.i, _N).place(y)

    @qd.kernel
    def setup() -> None:
        x[0] = 7.0

    @qd.kernel
    def kernel_sparse() -> None:
        snap = x[0]
        for i in range(_N):
            qd.deactivate(ptr, i)
        for i in range(_N):
            y[i] = snap

    setup()
    kernel_sparse()

    obs = kernel_sparse._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # snap captured the original x[0] == 7.0 before the deactivation, so every y[i] must be 7.0, not the 0 a
    # recomputed-after-deactivate load would read.
    assert np.allclose(y.to_numpy(), 7.0), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False, require=qd.extension.sparse)
def test_per_construct_frontend_split_fallback_append_result_shared() -> None:
    # `idx` captures the index returned by a top-level `qd.append` -- an `SNodeOpStmt::allocate` that writes its result
    # through a local alloca WITHOUT a `LocalStoreStmt`. A later construct reads `idx`. If the append is not tracked as
    # a local writer, the consumer's backward slice pulls a zero-initialized alloca (reads 0) and the effectful-producer
    # gate never sees the append; the split must treat the append as a writer, notice the effect crossing constructs,
    # and fall back.
    a = qd.field(qd.i32)
    qd.root.dynamic(qd.i, 256).place(a)

    @qd.kernel
    def prefill() -> None:
        for _ in range(3):
            qd.append(a.parent(), [], 7)

    @qd.kernel
    def kernel_append(out: qd.types.ndarray()) -> None:
        idx = qd.append(a.parent(), [], 9)
        for i in range(_N):
            out[i] = idx

    out = qd.ndarray(qd.i32, shape=(_N,))
    prefill()
    kernel_append(out)

    obs = kernel_append._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # The append returns the pre-append length (3 elements were prefilled), so every out[i] must be 3 -- not the 0 a
    # zero-init-alloca slice would read.
    assert np.all(out.to_numpy() == 3), out.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_dump_cfg(monkeypatch) -> None:
    # `QD_DUMP_CFG=1` is documented to dump the WHOLE-kernel CFG. The split would build/dump a CFG per construct into
    # the same filename (later constructs overwriting earlier ones), so it must decline the split in this diagnostic
    # mode even for an otherwise split-eligible kernel.
    monkeypatch.setenv("QD_DUMP_CFG", "1")

    @qd.kernel
    def kernel_cfg(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]

    arr = qd.ndarray(qd.f32, shape=(_N,))
    kernel_cfg(arr)

    obs = kernel_cfg._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    assert np.allclose(arr.to_numpy(), _C[0] + _C[1], atol=1.0), arr.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_struct_member_recomputed() -> None:
    # A local `qd.Struct` member is written at top level -- lowered to `LocalStoreStmt(GetElementStmt(alloca), ...)` --
    # and read by two later loop constructs. `resolve_local_alloca` must chase `GetElementStmt::src` to the base alloca;
    # otherwise the member store is not recognized as a writer, each consumer's backward slice keeps only the
    # zero-initialized struct alloca, and both loops read the stale 0 instead of the assigned value. With the member
    # writer visible, the store is pure/recomputable, so the split still fires (2 constructs) and stays correct.
    S = qd.types.struct(a=qd.f32, b=qd.f32)

    @qd.kernel
    def kernel_struct(out1: qd.types.ndarray(), out2: qd.types.ndarray()) -> None:
        s = S(a=0.0, b=0.0)
        s.a = _C[0]
        for i in range(_N):
            out1[i] = s.a
        for i in range(_N):
            out2[i] = s.a

    out1 = qd.ndarray(qd.f32, shape=(_N,))
    out2 = qd.ndarray(qd.f32, shape=(_N,))
    kernel_struct(out1, out2)

    obs = kernel_struct._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == 2, obs

    # Both loops must observe the member store (`s.a = _C[0]`) recomputed into their construct, not the 0 a slice that
    # dropped the GetElementStmt writer would read.
    assert np.allclose(out1.to_numpy(), _C[0], atol=1.0), out1.to_numpy()
    assert np.allclose(out2.to_numpy(), _C[0], atol=1.0), out2.to_numpy()
