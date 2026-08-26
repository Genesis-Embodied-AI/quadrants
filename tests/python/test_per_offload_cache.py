"""Per-construct FRONTEND split (no reuse tier).

`compile_to_offloads` splits the frontend (simplify / merge_global_ptrs / offload) to run per top-level construct
instead of once over the whole kernel, isolating each construct by its backward slice (in
`transforms/split_frontend_per_construct.cpp`), and falls back to the whole-kernel path for kernels that are not
recompute-safe: autodiff, mesh-for, a concurrently-executed region (`qd.stream_parallel()` / `qd.graph.parallel()`,
whose constructs share one global-temp buffer), a loop-produced local shared across constructs, a `qd.append` index a
later construct consumes, a local a later construct reads that a `@qd.real_func` (via a `qd.ref` argument) or an
external / bitcode call wrote to, a loop-carried local a later construct reads read-only through a `qd.ref` argument, a
non-recomputable producer a later construct would clone (an effectful one such as a global atomic, a non-deterministic
`qd.random()`, or a `qd.volatile_load()`), or a serial field load that a later construct would recompute after an
intervening effect (a store, atomic, or sparse activate/deactivate) mutated global state. It also declines the split
when `QD_DUMP_CFG` asks for a whole-kernel CFG dump (cfg_optimization, a pre-existing pass, forces the whole-kernel path
under it); `QD_DUMP_IR` / `QD_DUMP_SIMPLIFY` / `print_ir` are observation-only and keep the split active, emitting their
output per construct. This PR ships the split WITHOUT a reuse tier, so it recompiles every construct on every compile;
the reuse (a disk manifest keyed by a stable per-construct cache key) is added by the cross-process cache PR.

These tests assert the split's STRUCTURE and CORRECTNESS, not reuse:
 - the split fires for recompute-safe kernels and enumerates the expected constructs, with
   `frontend_constructs_recompiled == frontend_constructs_total` and `frontend_constructs_cache_hit == 0`;
 - kernels that are not recompute-safe fall back to the whole-kernel path (`frontend_constructs_total == -1`);
 - results stay numerically correct through the split-and-reassemble path.

Counts are exposed as `kernel._primal.per_offload_cache_observations`. `offline_cache=False` so the on-disk
whole-kernel cache never short-circuits codegen and the split always runs on the (cold) compile.
"""

import glob
import os
import shutil
import tempfile

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.util import has_clangpp

from tests import test_utils

# Dedicated IR-dump directories for the observation-only diagnostic tests below, so their assertions don't race the
# default /tmp/ir shared with other runs. Each test clears its own directory at the start of every arch run.
_IR_DUMP_DIR = os.path.join(tempfile.gettempdir(), "qd_test_per_construct_dump_ir")
_SIMPLIFY_DUMP_DIR = os.path.join(tempfile.gettempdir(), "qd_test_per_construct_dump_simplify")

# Kernel coverage (QD_KERNEL_COVERAGE=1) rewrites every kernel with per-line probe stores to a global coverage field,
# which adds top-level global-write constructs (changing the split's construct partition) and makes the split fall back
# to the whole-kernel path. Both defeat these observation assertions, so CI runs this file in a separate no-coverage
# phase (see .github/workflows/scripts_new/linux/4_test.sh), mirroring test_offline_cache.py.
pytestmark = pytest.mark.skipif(
    os.environ.get("QD_KERNEL_COVERAGE") == "1",
    reason="Kernel coverage instrumentation disables the per-construct split and changes construct counts",
)

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
def test_per_construct_frontend_split_scalar_reassigned_between_constructs() -> None:
    # A top-level scalar is REASSIGNED between two constructs, so each loop must observe the store that precedes IT in
    # source order (loop0 -> 1, loop1 -> 2). The backward slice may pull a local's writers only from EARLIER segments;
    # cloning the later `a = 2` into loop0's slice would make the isolated loop0 read 2 instead of 1. Regression test
    # for the source-order restriction on local writers in the slice.
    @qd.kernel
    def kernel_reassign(a_out: qd.types.ndarray(), b_out: qd.types.ndarray()) -> None:
        a = 1
        for i in range(_N):
            a_out[i] = a
        a = 2
        for i in range(_N):
            b_out[i] = a

    a_arr = qd.ndarray(qd.i32, shape=(_N,))
    b_arr = qd.ndarray(qd.i32, shape=(_N,))
    kernel_reassign(a_arr, b_arr)

    obs = kernel_reassign._primal.per_offload_cache_observations
    # The scalar is written at top level (not loop-carried), so the kernel is recompute-safe and the split fires.
    assert obs.frontend_constructs_total == 2, obs
    # Each loop reads the value assigned immediately before it, NOT the later reassignment.
    assert (a_arr.to_numpy() == 1).all(), a_arr.to_numpy()
    assert (b_arr.to_numpy() == 2).all(), b_arr.to_numpy()


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


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False, debug_dump_path=_IR_DUMP_DIR)
def test_per_construct_frontend_split_dump_ir_observation_only(monkeypatch) -> None:
    # `QD_DUMP_IR=1` is observation-only under the split: rather than declining, the split still runs and dumps each
    # construct's IR to `<kernel>_construct<i>_<stage>.ll`, so the snapshots reflect what was actually compiled.
    shutil.rmtree(_IR_DUMP_DIR, ignore_errors=True)
    os.makedirs(_IR_DUMP_DIR, exist_ok=True)
    monkeypatch.setenv("QD_DUMP_IR", "1")

    @qd.kernel
    def kernel_ir(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]

    arr = qd.ndarray(qd.f32, shape=(_N,))
    kernel_ir(arr)

    obs = kernel_ir._primal.per_offload_cache_observations
    # The diagnostic did NOT disable the split; it ran and enumerated both constructs.
    assert obs.frontend_constructs_total == 2, obs
    # One per-construct IR snapshot per construct was written, so the dump reflects the split rather than a synthetic
    # whole-kernel view.
    per_construct_dumps = glob.glob(os.path.join(_IR_DUMP_DIR, "*_construct*_after_simplify_I.ll"))
    listing = sorted(os.listdir(_IR_DUMP_DIR)) if os.path.isdir(_IR_DUMP_DIR) else "<no dump dir>"
    assert len(per_construct_dumps) >= 2, listing

    assert np.allclose(arr.to_numpy(), _C[0] + _C[1], atol=1.0), arr.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False, debug_dump_path=_SIMPLIFY_DUMP_DIR)
def test_per_construct_frontend_split_dump_simplify_observation_only(monkeypatch) -> None:
    # `QD_DUMP_SIMPLIFY=1` is observation-only under the split too: simplify.cpp keys its dump filenames off a global
    # call counter, so each construct's simplify passes already land in distinct files. The split keeps running.
    # simplify.cpp's QD_DUMP_SIMPLIFY writer does not create the dump directory itself, so make sure it exists (in
    # normal use /tmp/ir already does); create it after clearing so the assertion sees only this run's files.
    shutil.rmtree(_SIMPLIFY_DUMP_DIR, ignore_errors=True)
    os.makedirs(_SIMPLIFY_DUMP_DIR, exist_ok=True)
    monkeypatch.setenv("QD_DUMP_SIMPLIFY", "1")

    @qd.kernel
    def kernel_simplify(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]

    arr = qd.ndarray(qd.f32, shape=(_N,))
    kernel_simplify(arr)

    obs = kernel_simplify._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == 2, obs
    simplify_dumps = glob.glob(os.path.join(_SIMPLIFY_DUMP_DIR, "*.ir"))
    listing = sorted(os.listdir(_SIMPLIFY_DUMP_DIR)) if os.path.isdir(_SIMPLIFY_DUMP_DIR) else "<no dump dir>"
    assert len(simplify_dumps) >= 2, listing

    assert np.allclose(arr.to_numpy(), _C[0] + _C[1], atol=1.0), arr.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False, print_ir=True)
def test_per_construct_frontend_split_print_ir_observation_only() -> None:
    # `qd.init(print_ir=True)` is observation-only under the split: each construct's passes print (under a per-construct
    # banner) instead of the split declining. Assert the split still ran (the console output is per construct).
    @qd.kernel
    def kernel_print(x: qd.types.ndarray()) -> None:
        for i in range(_N):
            x[i] += _C[0]
        for i in range(_N):
            x[i] += _C[1]

    arr = qd.ndarray(qd.f32, shape=(_N,))
    kernel_print(arr)

    obs = kernel_print._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == 2, obs

    assert np.allclose(arr.to_numpy(), _C[0] + _C[1], atol=1.0), arr.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_realfunc_ref_write() -> None:
    # A `@qd.real_func` mutates a local through a `qd.ref` argument -- a `FuncCallStmt` writer, reported via the shared
    # `get_store_destination` trait, not a `LocalStoreStmt`. A later loop reads that local. If the call is not tracked
    # as a writer, the consumer's backward slice keeps only the earlier initialization and reads the stale value; since
    # a real_func call is effectful, the split must fall back rather than clone the call into the consumer.
    @qd.real_func
    def assign_c0(a: qd.ref(qd.f32)):
        a = _C[0]

    @qd.kernel
    def kernel_ref(out: qd.types.ndarray()) -> None:
        a = 5.0
        assign_c0(a)
        for i in range(_N):
            out[i] = a

    out = qd.ndarray(qd.f32, shape=(_N,))
    kernel_ref(out)

    obs = kernel_ref._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # The real_func wrote _C[0] into `a` before the loop, so every out[i] must be _C[0] -- not the initial 5.0 a slice
    # that dropped the FuncCallStmt writer would recompute.
    assert np.allclose(out.to_numpy(), _C[0], atol=1.0), out.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_realfunc_ref_read() -> None:
    # A local ACCUMULATED inside one top-level loop (loop-carried) is then read -- READ-ONLY -- by a `@qd.real_func`
    # through a `qd.ref` argument in a LATER loop. That read reaches the callee as a `ReferenceStmt` (the shared `Load`
    # trait) with no store destination, so a store-only scan of the consumer misses it. The loop-carried-local gate must
    # still see the read via `get_load_pointers`; otherwise it accepts the split, isolates the second loop, drops the
    # producing loop (whose write is inside a container, so it is not a top-level writer the slice recomputes), and the
    # callee reads the 0.0 initialization instead of the accumulated value.
    @qd.real_func
    def read_a(a: qd.ref(qd.f32)) -> qd.f32:
        return a

    @qd.kernel
    def kernel_ref_read(out: qd.types.ndarray()) -> None:
        acc = 0.0
        for i in range(_N):
            acc += 1.0
        for i in range(_N):
            out[i] = read_a(acc)

    out = qd.ndarray(qd.f32, shape=(_N,))
    kernel_ref_read(out)

    obs = kernel_ref_read._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # `acc` accumulates to _N over the first loop, so every out[i] must be _N -- not the 0.0 initialization a slice that
    # dropped the producing loop would recompute.
    assert np.allclose(out.to_numpy(), float(_N)), out.to_numpy()


@pytest.mark.skipif(not has_clangpp(), reason="Clang not installed.")
@test_utils.test(arch=[qd.x64, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_external_func_write() -> None:
    # An external (SourceBuilder/bitcode) call mutates a local through a pointer output arg -- an `ExternalFuncCallStmt`
    # whose get_store_destination() reports the output alloca, surfaced by the generic writer analysis. A later loop
    # reads that local; the call must be counted as a writer, otherwise the consumer's slice keeps the stale init. Since
    # the external call is effectful, the split falls back.
    sb = qd.lang.source_builder.SourceBuilder.from_source(
        """
    extern "C" {
        void plus_one(float *a, float *out) { *out = (*a) + 1.0f; }
    }
    """
    )

    @qd.kernel
    def kernel_ext(out: qd.types.ndarray()) -> None:
        v = 5.0
        r = 0.0
        sb.plus_one(v, r)
        for i in range(_N):
            out[i] = r

    arr = qd.ndarray(qd.f32, shape=(_N,))
    kernel_ext(arr)

    obs = kernel_ext._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    # The external call wrote 6.0 into `r` before the loop, so every out[i] must be 6.0, not the initial 0.0 a slice
    # that dropped the ExternalFuncCallStmt writer would recompute.
    assert np.allclose(arr.to_numpy(), 6.0, atol=1e-3), arr.to_numpy()


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_per_construct_frontend_split_graph_do_while() -> None:
    # A `qd.graph.do_while` kernel is driven from the HOST: `offload` flattens the loop body into a CONTIGUOUS run of
    # tasks all tagged with the loop's `graph_do_while_level_id`, and the host graph driver relaunches that run each
    # iteration until the on-device break flag fires. The per-construct split offloads each construct in isolation and
    # reassembles, which re-runs the offloader's serial-bucket / region-tag assignment out of whole-kernel context: a
    # serial task (e.g. a pass-made global-temp materialization, whose region tag defaults to level -1) can flush at the
    # wrong level and get wedged out of the body's contiguous run, so the host loop counter is stranded outside the body,
    # never decrements, and the kernel spins forever (this is what hung Genesis's decomposed rigid constraint solver on
    # coupled contact). The split therefore REATTACHES each construct's do_while level onto its reassembled tasks (see
    # `construct_gdw_level` in split_frontend_per_construct.cpp) so the body stays one contiguous same-level run. Assert
    # the split FIRES (this is the giant-solver kernel we most want per-construct-cacheable, so it must NOT be gated out)
    # and the reassembled loop is still correct: the do_while runs 3 times and increments x each time, so x must end at 3.
    @qd.kernel(graph=True)
    def run(x: qd.types.ndarray(qd.i32, ndim=1), counter: qd.types.ndarray(qd.i32, ndim=0)) -> None:
        while qd.graph.do_while(counter):
            for i in range(_N):
                if x[i] < 100:
                    x[i] = x[i] + 1
            counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(_N,))
    x.from_numpy(np.zeros(_N, dtype=np.int32))
    counter = qd.ndarray(qd.i32, shape=())
    counter.from_numpy(np.array(3, dtype=np.int32))
    run(x, counter)

    obs = run._primal.per_offload_cache_observations
    # Split fired (do_while kernels are no longer gated out), with no reuse tier (every construct recompiled).
    assert obs.frontend_constructs_total >= 1, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs

    # Correct + terminating: a mis-leveled reassembled task would strand the counter outside the loop body -> the host
    # do_while never terminates (hang) or the increment runs the wrong number of times.
    assert np.all(x.to_numpy() == 3), x.to_numpy()


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_per_construct_frontend_split_nested_graph_do_while() -> None:
    # NESTED `qd.graph.do_while`: an outer loop resets and drives an inner loop. The graph_do_while transformer FLATTENS
    # the nest before the frontend split runs, so the top-level block is a flat run of single-level constructs -- the
    # inner-body range-for and the inner counter decrement carry `graph_do_while_level_id == 1`, while the inner reset
    # and outer decrement carry level 0. This guards `construct_gdw_level`'s container branch (which re-stamps a whole
    # construct to a for-loop's own level): if it recovered the WRONG level for a deeper-nested construct it would
    # collapse the two levels together, and the host driver -- which rebuilds the loop nest purely from the flat tasks'
    # level ids -- would either run the inner loop the wrong number of times or strand a counter and hang. Assert the
    # split FIRES on the nested kernel and the reassembled two-level nest is still correct: x is incremented once per
    # (outer, inner) iteration, so it must end at _OUTER * _INNER, and the outer counter must drain to 0.
    _OUTER, _INNER = 3, 4

    @qd.kernel(graph=True)
    def run(
        x: qd.types.ndarray(qd.i32, ndim=1),
        outer: qd.types.ndarray(qd.i32, ndim=0),
        inner: qd.types.ndarray(qd.i32, ndim=0),
    ) -> None:
        while qd.graph.do_while(outer):
            for _ in range(1):
                inner[()] = _INNER
            while qd.graph.do_while(inner):
                for i in range(_N):
                    x[i] = x[i] + 1
                for _ in range(1):
                    inner[()] = inner[()] - 1
            for _ in range(1):
                outer[()] = outer[()] - 1

    x = qd.ndarray(qd.i32, shape=(_N,))
    x.from_numpy(np.zeros(_N, dtype=np.int32))
    outer = qd.ndarray(qd.i32, shape=())
    outer.from_numpy(np.array(_OUTER, dtype=np.int32))
    inner = qd.ndarray(qd.i32, shape=())
    inner.from_numpy(np.array(0, dtype=np.int32))
    run(x, outer, inner)

    obs = run._primal.per_offload_cache_observations
    # Split fired on the nested kernel (no per-level gating), every construct recompiled (no reuse tier).
    assert obs.frontend_constructs_total >= 1, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs

    # Correct + terminating: collapsing the inner level onto the outer (or vice versa) would miscount the nested work or
    # strand a counter -> wrong total or a hang.
    assert np.all(x.to_numpy() == _OUTER * _INNER), x.to_numpy()
    assert outer.to_numpy() == 0, outer.to_numpy()


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
