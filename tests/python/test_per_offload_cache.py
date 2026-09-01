"""Per-construct FRONTEND split.

`compile_to_offloads` runs the frontend (simplify / merge_global_ptrs / offload) per top-level construct instead of
once over the whole kernel, isolating each construct by its backward slice (see
`transforms/split_frontend_per_construct.cpp` for the split and the recompute-safety fallback conditions).

These tests assert the split's structure and correctness (there is no reuse tier yet):
 - the split fires for recompute-safe kernels, with `frontend_constructs_recompiled == frontend_constructs_total` and
   `frontend_constructs_cache_hit == 0`;
 - non-recompute-safe kernels fall back to the whole-kernel path (`frontend_constructs_total == -1`);
 - results stay numerically correct through the split-and-reassemble path.

Counts are exposed as `kernel._primal.per_offload_cache_observations`. `offline_cache=False` so the on-disk cache never
short-circuits codegen and the split always runs on the cold compile.
"""

import dataclasses
import glob
import os
import shutil
import tempfile

import numpy as np
import pytest

import quadrants as qd
from quadrants._test_tools import qd_init_same_arch
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
    # Four independent loops enumerate as four constructs; with no reuse tier every one is recompiled, none a hit.
    assert obs.frontend_constructs_total == 4, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs
    assert np.allclose(arr.to_numpy(), sum(_C), atol=1.0), arr.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_observations_reset_on_cached_relaunch() -> None:
    # `per_offload_cache_observations` describes only the MOST RECENT compile. A relaunch served from a cached artifact
    # runs no split, so the counts must reset to the no-split sentinel (-1) rather than leak the previous split's counts.
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

    # Second launch reuses the compiled artifact, so the observation resets to the sentinel.
    kernel_relaunch(arr)
    obs = kernel_relaunch._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs

    assert np.allclose(arr.to_numpy(), 2.0 * (_C[0] + _C[1]), atol=1.0), arr.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_correct_with_shared_serial_def() -> None:
    # A serial prologue defines `base`, consumed by two later loop constructs; the backward slice must recompute the
    # defining stores into each (an operand-only slice would drop them and read zeros).
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
    # A top-level scalar is reassigned between two constructs, so each loop must read the store that precedes it
    # (loop0 -> 1, loop1 -> 2). Guards the slice's source-order restriction: only writers from EARLIER segments are
    # pulled in, else loop0 would clone the later `a = 2` and read 2.
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
    assert obs.frontend_constructs_total == 2, obs
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
    # `base` must stay the ORIGINAL x[0] (7.0), not the 2.0 the first loop wrote.
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_recomputed_read_disjoint_write_splits() -> None:
    # Mirror of `..._fallback_field_load_shadowed`, but the intervening loop writes a DIFFERENT ndarray (`b`) than the
    # snapshotted read (`a[0]`). Recomputing the `a[0]` load into the consumer cannot observe a write to `b`, so the
    # SNode/ndarray-aware recompute-safety check must allow the split -- the old address-agnostic check wrongly forbade
    # it. This is the case that unblocks multi-array kernels (e.g. the qipc solver).
    @qd.kernel
    def kernel_disjoint(a: qd.types.ndarray(), b: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        base = a[0]
        for i in range(_N):
            b[i] = 2.0
        for i in range(_N):
            y[i] = base

    a = qd.ndarray(qd.f32, shape=(_N,))
    b = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_disjoint(a, b, y)

    obs = kernel_disjoint._primal.per_offload_cache_observations
    # The disjoint write does not alias the recomputed read, so the split fires (it is NOT the -1 fallback sentinel).
    assert obs.frontend_constructs_total >= 2, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs
    # `base` is a[0] == 7.0 (a is never written), so every y[i] must be 7.0; b holds the unrelated 2.0.
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()
    assert np.allclose(b.to_numpy(), 2.0, atol=1e-2), b.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_recomputed_read_disjoint_atomic_splits() -> None:
    # Same as above but the intervening effect is an ATOMIC to a different ndarray, exercising the atomic `dest` branch
    # of the write-pointer mapping: an atomic on `b` cannot alias a recomputed read of `a`, so the split still fires.
    @qd.kernel
    def kernel_disjoint_atomic(a: qd.types.ndarray(), b: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        base = a[0]
        for i in range(_N):
            qd.atomic_add(b[i], 1.0)
        for i in range(_N):
            y[i] = base

    a = qd.ndarray(qd.f32, shape=(_N,))
    b = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    b.from_numpy(np.zeros(_N, dtype=np.float32))
    kernel_disjoint_atomic(a, b, y)

    obs = kernel_disjoint_atomic._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()
    assert np.allclose(b.to_numpy(), 1.0, atol=1e-2), b.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_alias_guard_falls_back() -> None:
    # The split clears condition (3) between the recomputed read `a[0]` and the intervening write `b[i]` because they
    # are ndarray accesses on DIFFERENT args -- disjointness the caller can defeat by binding one buffer to both `a` and
    # `b`. The launch-time no-alias guard must catch that and run the whole-kernel variant (which snapshots `base`
    # before the write), so the aliased call gets the correct answer instead of the split's recompute-after-write value.
    @qd.kernel
    def kernel_guarded(a: qd.types.ndarray(), b: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        base = a[0]
        for i in range(_N):
            b[i] = 2.0
        for i in range(_N):
            y[i] = base

    # Distinct buffers: the split fires (assuming a and b disjoint) and is correct. This also arms the guard.
    a = qd.ndarray(qd.f32, shape=(_N,))
    b = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_guarded(a, b, y)

    obs = kernel_guarded._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs
    assert kernel_guarded._primal._split_alias_guard_by_key, "split did not record the cross-arg ndarray assumption"
    assert not kernel_guarded._primal._compiled_no_split_by_key, "no fallback should be built for a disjoint call"
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()

    # Aliased call: `a` and `b` are the SAME buffer, so the split would recompute `base = a[0]` in the y-loop AFTER
    # `b[i]=2.0` overwrote it and read 2.0. The guard detects the shared allocation and runs the whole-kernel variant,
    # which captured `base` (7.0) before the write. Correctness -- not just the reset sentinel -- is the real check.
    shared = qd.ndarray(qd.f32, shape=(_N,))
    y2 = qd.ndarray(qd.f32, shape=(_N,))
    shared.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_guarded(shared, shared, y2)

    assert kernel_guarded._primal._compiled_no_split_by_key, "guard did not build the whole-kernel fallback variant"
    assert np.allclose(y2.to_numpy(), 7.0, atol=1e-2), y2.to_numpy()  # 7.0 (whole-kernel), never 2.0 (split miscompile)
    assert np.allclose(shared.to_numpy(), 2.0, atol=1e-2), shared.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_alias_guard_cleared_on_reset() -> None:
    # The whole-kernel fallback cache holds CompiledKernelData owned by the current Program, so reset() must drop it
    # (and the guard state) alongside the other compiled-data caches. Otherwise a re-launch after `qd.reset()` hands the
    # new Program a fallback the destroyed one built -> invalid backend state. Reusing the SAME kernel across the reset
    # is the point of the test: a freshly created kernel could never carry the stale entry.
    @qd.kernel
    def kernel_guarded(a: qd.types.ndarray(), b: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        base = a[0]
        for i in range(_N):
            b[i] = 2.0
        for i in range(_N):
            y[i] = base

    shared = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    shared.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_guarded(shared, shared, y)  # aliased a/b arms the guard and builds the fallback
    assert kernel_guarded._primal._compiled_no_split_by_key, "aliased launch did not build the whole-kernel fallback"
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()

    # Re-init the same arch: destroys the Program and, via impl.reset(), resets every registered kernel in place.
    qd_init_same_arch(offline_cache=False)
    assert not kernel_guarded._primal._compiled_no_split_by_key, "reset left a fallback owned by the destroyed Program"
    assert not kernel_guarded._primal._split_alias_guard_by_key, "reset left stale guard state"

    # Reuse the same kernel against the new Program: the fallback rebuilds fresh (no stale-Program launch), and the
    # aliased call is still correct (7.0 whole-kernel, never 2.0 split miscompile).
    shared2 = qd.ndarray(qd.f32, shape=(_N,))
    y2 = qd.ndarray(qd.f32, shape=(_N,))
    shared2.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_guarded(shared2, shared2, y2)
    assert kernel_guarded._primal._compiled_no_split_by_key, "fallback not rebuilt after reset"
    assert np.allclose(y2.to_numpy(), 7.0, atol=1e-2), y2.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_alias_guard_benign_alias_still_splits() -> None:
    # The guard falls back only on the SPECIFIC arg pairs the split relied on being disjoint, not on any two args that
    # happen to share a buffer. Here `d` aliases `a`, but the split never recomputes a read of `d` across a write (it
    # is read and consumed in one construct), so `d` is in no recorded pair. The only recorded pair is (a, b) -- from
    # recomputing `a[0]` into the y-loop past the write to `b` -- and a/b are distinct buffers. So the benign a/d alias
    # must NOT force a whole-kernel fallback: the split stays. This is qipc `_step_kernel`'s `triplet_overflow` self-
    # alias in miniature (bound to two params, but disjointness was never assumed about it). A one-bit guard would
    # wrongly fall back here.
    @qd.kernel
    def kernel_benign(
        a: qd.types.ndarray(), b: qd.types.ndarray(), y: qd.types.ndarray(), d: qd.types.ndarray()
    ) -> None:
        base = a[0]
        for i in range(_N):
            b[i] = d[i]
        for i in range(_N):
            y[i] = base

    a = qd.ndarray(qd.f32, shape=(_N,))
    b = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_benign(a, b, y, a)  # d bound to a's buffer: a benign alias outside any relied-on pair

    obs = kernel_benign._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs  # the split fired despite the a/d alias
    assert kernel_benign._primal._split_alias_guard_by_key, "split did not record the (a, b) assumption"
    assert (
        not kernel_benign._primal._compiled_no_split_by_key
    ), "benign a/d alias must not build a whole-kernel fallback"
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()
    assert np.allclose(b.to_numpy(), 7.0, atol=1e-2), b.to_numpy()  # b[i] = d[i] = a[i] = 7


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_clamped_boundary_recompute_unsafe() -> None:
    # alias_analysis proves `a[_N]` and `a[_N - 1]` disjoint by their raw indices (they differ by 1), but with
    # `boundary="clamp"` the out-of-range `a[_N]` maps onto the last element `a[_N - 1]`, so they are the SAME address.
    # Boundary lowering runs after the split, so condition (3) would otherwise clear the recomputed read `a[_N]` across
    # the intervening write `a[_N - 1]=2` and the split would then read 2 instead of the original last element. Clamped
    # same-buffer accesses must force the whole-kernel path. (Negative literals like `a[-1]` are rejected by the
    # frontend, so the out-of-bounds-high index is how a compile-time-constant clamp is expressed here.)
    @qd.kernel
    def kernel_clamp(a: qd.types.ndarray(boundary="clamp"), y: qd.types.ndarray()) -> None:
        base = a[_N]  # out of range high -> clamps to a[_N - 1]
        for i in range(_N):
            a[_N - 1] = 2.0
        for i in range(_N):
            y[i] = base

    a = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_clamp(a, y)

    obs = kernel_clamp._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs  # fell back: clamp defeats the raw-index disjointness proof
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()  # 7.0 (base snapshot), never 2.0 (split recompute)
    assert np.allclose(a.to_numpy()[_N - 1], 2.0, atol=1e-2), a.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_cross_arg_grad_pair_unsafe() -> None:
    # Recompute a read of `x.grad` past a write to `y.grad` on two DISTINCT args. alias_analysis calls them different by
    # arg id, so condition (3) clears -- but both accesses are GRADIENT buffers, and the launch guard resolves a slot
    # only to its arg's PRIMAL alloc_id, so it cannot see two args' gradient companions sharing one allocation.
    # Recording that pair would be unguardable, so the split must refuse it and run whole-kernel (constructs == -1).
    # (Distinct grad buffers make this call numerically fine either way; the point is the split DECISION, not a value.)
    @qd.kernel
    def kernel_grad(x: qd.types.ndarray(), y: qd.types.ndarray(), out: qd.types.ndarray()) -> None:
        base = x.grad[0]
        for i in range(_N):
            y.grad[i] = 2.0
        for i in range(_N):
            out[i] = base

    x = qd.ndarray(qd.f32, shape=(_N,), needs_grad=True)
    y = qd.ndarray(qd.f32, shape=(_N,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(_N,))
    x.grad.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_grad(x, y, out)

    obs = kernel_grad._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == -1, obs  # refused: a gradient-buffer pair is not launch-guardable
    assert not kernel_grad._primal._split_alias_guard_by_key, "an unguardable gradient pair must not be recorded"
    assert np.allclose(out.to_numpy(), 7.0, atol=1e-2), out.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_alias_guard_dataclass_field_slots() -> None:
    # The split can rest on two ndarray FIELDS of a typed dataclass being disjoint. Those fields flatten to their own
    # arg slots, so the guard must resolve each slot back to its live ndarray via the root dataclass arg + attribute
    # chain. Without that, the disjoint launch can't be verified and always falls back to whole-kernel, defeating the
    # split. The `not _compiled_no_split_by_key` assertion on the disjoint launch is what the field-slot recording buys.
    @dataclasses.dataclass
    class Pair:
        fa: qd.types.NDArray[qd.f32, 1]
        fb: qd.types.NDArray[qd.f32, 1]

    @qd.kernel
    def kernel_pair(p: Pair, y: qd.types.ndarray()) -> None:
        base = p.fa[0]
        for i in range(_N):
            p.fb[i] = 2.0
        for i in range(_N):
            y[i] = base

    def _nd(v):
        a = qd.ndarray(qd.f32, shape=(_N,))
        a.from_numpy(np.full(_N, v, dtype=np.float32))
        return a

    # Disjoint fields: the guard resolves both field slots, sees distinct buffers, and keeps the split.
    fa, fb, y = _nd(7.0), _nd(0.0), qd.ndarray(qd.f32, shape=(_N,))
    kernel_pair(Pair(fa, fb), y)
    obs = kernel_pair._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs  # the split fired at compile time
    assert kernel_pair._primal._split_alias_guard_by_key, "split did not record the (fa, fb) field assumption"
    assert not kernel_pair._primal._compiled_no_split_by_key, "disjoint fields must not build a whole-kernel fallback"
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()

    # Aliased fields: one buffer bound to both fa and fb. The guard resolves both to a single alloc_id and falls back to
    # the whole-kernel variant, which snapshots base before the write (7.0), not the split's post-write recompute (2.0).
    shared, y2 = _nd(7.0), qd.ndarray(qd.f32, shape=(_N,))
    kernel_pair(Pair(shared, shared), y2)
    assert kernel_pair._primal._compiled_no_split_by_key, "aliased fields did not trigger the whole-kernel fallback"
    assert np.allclose(y2.to_numpy(), 7.0, atol=1e-2), y2.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_alias_guard_tensor_arg_slots() -> None:
    # A top-level param annotated as the public `qd.Tensor` (ndarray-backed) flattens to an external-tensor slot just
    # like `qd.types.ndarray()`. Recording keys off the generated object's slot, not the annotation type, so the guard
    # resolves `qd.Tensor` args too; without that a disjoint launch could not verify and always fell back.
    @qd.kernel
    def kernel_tensor(a: qd.Tensor, b: qd.Tensor, y: qd.types.ndarray()) -> None:
        base = a[0]
        for i in range(_N):
            b[i] = 2.0
        for i in range(_N):
            y[i] = base

    def _t(v):
        t = qd.tensor(qd.f32, shape=(_N,), backend=qd.Backend.NDARRAY)
        t.from_numpy(np.full(_N, v, dtype=np.float32))
        return t

    a, b, y = _t(7.0), _t(0.0), qd.ndarray(qd.f32, shape=(_N,))
    kernel_tensor(a, b, y)
    obs = kernel_tensor._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs  # split fired and stayed (guard resolved the Tensor slots)
    assert kernel_tensor._primal._split_alias_guard_by_key, "split did not record the (a, b) tensor assumption"
    assert not kernel_tensor._primal._compiled_no_split_by_key, "disjoint Tensor args must not build a fallback"
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_per_construct_frontend_split_barrier_between_constructs_splits() -> None:
    # `internal_func_is_memory_free` allowlist regression. A block barrier reports has_global_side_effect (to pin
    # ordering) but writes no addressable memory. Here it is the only statement between the recomputed read `a[0]` and
    # its consumer loop: a barrier NOT on the allowlist would be gathered as an unpinnable (may-alias-all) write and
    # force the whole-kernel fallback; allowlisted, the recompute is provably unaffected and the split fires. This is
    # the path that lets qipc's `_step_kernel` (dense with block-reduction barriers) split at all. CUDA-only: block
    # barriers are unsupported on the CPU backend.
    @qd.kernel
    def kernel_barrier(a: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
        base = a[0]
        qd.simt.block.sync()
        for i in range(_N):
            y[i] = base

    a = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    kernel_barrier(a, y)

    obs = kernel_barrier._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs  # not the -1 fallback: the barrier did not block the split
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_alias_guard_struct_member_path() -> None:
    # ndarrays reaching the kernel through `@qd.data_oriented` struct members. Exercises the guard's struct branch
    # (`_struct_ndarray_launch_info_by_key` + `_resolve_struct_ndarray`) on the disjoint launch, and pins where the
    # aliasing check happens for struct members: `_predeclare_struct_ndarrays` dedups members by `id()`, so binding one
    # ndarray to two members collapses them to a single predeclared arg under a distinct specialization key. The split
    # then sees one arg (no cross-member disjointness to assume) and stays correct WITHOUT a launch-time fallback --
    # aliasing is resolved at compile time here, unlike the top-level-param case.
    @qd.data_oriented
    class State:
        def __init__(self, a, b, y):
            self.a = a
            self.b = b
            self.y = y

    @qd.kernel
    def step(s: qd.template()) -> None:
        base = s.a[0]
        for i in range(_N):
            s.b[i] = 2.0
        for i in range(_N):
            s.y[i] = base

    # Distinct members: a, b, y are three predeclared args, so the split assumes them disjoint and arms the guard. The
    # launch runs the guard's struct branch (resolving each member), finds no collision, and stays on the split.
    a = qd.ndarray(qd.f32, shape=(_N,))
    b = qd.ndarray(qd.f32, shape=(_N,))
    y = qd.ndarray(qd.f32, shape=(_N,))
    a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    step(State(a, b, y))

    obs = step._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total >= 2, obs
    assert step._primal._split_alias_guard_by_key, "split did not record the cross-member ndarray assumption"
    assert step._primal._struct_ndarray_launch_info_by_key, "struct ndarray launch info was not recorded"
    assert not step._primal._compiled_no_split_by_key, "no fallback should be built for a disjoint call"
    assert np.allclose(y.to_numpy(), 7.0, atol=1e-2), y.to_numpy()

    # Aliased members: the same buffer bound to `a` and `b`. `_predeclare_struct_ndarrays` collapses them to one arg
    # under a fresh specialization key, so the split never assumes them disjoint and the result is correct with no
    # launch-time whole-kernel fallback.
    shared = qd.ndarray(qd.f32, shape=(_N,))
    y2 = qd.ndarray(qd.f32, shape=(_N,))
    shared.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    step(State(shared, shared, y2))

    assert not step._primal._compiled_no_split_by_key, "struct-member aliasing is handled statically, not by fallback"
    assert np.allclose(y2.to_numpy(), 7.0, atol=1e-2), y2.to_numpy()  # 7.0 via compile-time dedup, never 2.0


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_per_construct_frontend_split_alias_guard_armed_from_fastcache(tmp_path) -> None:
    # The assumed-disjoint pairs ride on the compiled artifact, so a kernel restored from the fastcache in a fresh
    # process (never freshly compiled here) must still arm the launch-time guard from the serialized
    # `split_assumed_disjoint_pairs` -- otherwise a cross-process cache hit would silently run the unguarded split under
    # aliasing. Emulates the second process with `qd_init_same_arch` + a re-created Kernel object (empty guard dict).
    def _make():
        @qd.kernel(fastcache=True)
        def kernel_guarded(a: qd.types.ndarray(), b: qd.types.ndarray(), y: qd.types.ndarray()) -> None:
            base = a[0]
            for i in range(_N):
                b[i] = 2.0
            for i in range(_N):
                y[i] = base

        return kernel_guarded

    def _bufs():
        a = qd.ndarray(qd.f32, shape=(_N,))
        b = qd.ndarray(qd.f32, shape=(_N,))
        y = qd.ndarray(qd.f32, shape=(_N,))
        a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
        return a, b, y

    # Process 1: fresh compile arms the guard and stores the artifact (with the flag) to the fastcache.
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    k1 = _make()
    k1(*_bufs())
    assert k1._primal._split_alias_guard_by_key, "fresh compile did not arm the guard"

    # Process 2: a brand-new Kernel object restores from the fastcache. Its guard dict starts empty and must be armed
    # purely from the restored artifact's flag.
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    k2 = _make()
    a2, b2, y2 = _bufs()
    k2(a2, b2, y2)
    assert k2._primal.src_ll_cache_observations.cache_loaded, "expected a fastcache restore, not a fresh compile"
    assert k2._primal._split_alias_guard_by_key, "guard not armed from the restored artifact"
    assert not k2._primal._compiled_no_split_by_key, "disjoint call should not build a fallback"
    assert np.allclose(y2.to_numpy(), 7.0, atol=1e-2), y2.to_numpy()

    # Aliased call on the restored kernel: the guard armed from the cache must still fire.
    shared = qd.ndarray(qd.f32, shape=(_N,))
    y3 = qd.ndarray(qd.f32, shape=(_N,))
    shared.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    k2(shared, shared, y3)
    assert k2._primal._compiled_no_split_by_key, "guard armed from fastcache did not fall back under aliasing"
    assert np.allclose(y3.to_numpy(), 7.0, atol=1e-2), y3.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_per_construct_frontend_split_alias_guard_fastcache_fallback_prunes_callee(tmp_path) -> None:
    # The whole-kernel fallback for a fastcache-restored key must rebuild the FULL pruning (discovery pass), not just
    # enforce off the cached kernel-root used set. Here the kernel forwards a dataclass arg into a `@qd.func` that
    # reads one of its ndarray fields; only the discovery pass records that the callee uses `bias.vals`. Without it the
    # callee's used set is empty on rebuild, `bias.vals` is pruned, and the fallback build fails ("Name ... not
    # defined"). The aliased launch is what forces that rebuild, so this exercises the corner the armed-from-fastcache
    # test (no callee) does not.
    @dataclasses.dataclass
    class Bias:
        vals: qd.types.NDArray[qd.f32, 1]

    @qd.func
    def add_bias(bias: Bias, base, i):
        return base + bias.vals[i]

    def _make():
        @qd.kernel(fastcache=True)
        def kernel_biased(a: qd.types.ndarray(), b: qd.types.ndarray(), y: qd.types.ndarray(), bias: Bias) -> None:
            base = a[0]
            for i in range(_N):
                b[i] = 2.0
            for i in range(_N):
                y[i] = add_bias(bias, base, i)

        return kernel_biased

    def _bufs():
        a = qd.ndarray(qd.f32, shape=(_N,))
        b = qd.ndarray(qd.f32, shape=(_N,))
        y = qd.ndarray(qd.f32, shape=(_N,))
        vals = qd.ndarray(qd.f32, shape=(_N,))
        a.from_numpy(np.full(_N, 7.0, dtype=np.float32))
        vals.from_numpy(np.full(_N, 1.0, dtype=np.float32))
        return a, b, y, Bias(vals)

    # Process 1: fresh compile arms the guard and persists the artifact.
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    k1 = _make()
    k1(*_bufs())

    # Process 2: restore from fastcache, then an aliased launch forces the whole-kernel fallback rebuild.
    qd_init_same_arch(offline_cache_file_path=str(tmp_path), offline_cache=True)
    k2 = _make()
    a2, b2, y2, bias2 = _bufs()
    k2(a2, b2, y2, bias2)
    assert k2._primal.src_ll_cache_observations.cache_loaded, "expected a fastcache restore, not a fresh compile"

    shared = qd.ndarray(qd.f32, shape=(_N,))
    y3 = qd.ndarray(qd.f32, shape=(_N,))
    vals = qd.ndarray(qd.f32, shape=(_N,))
    shared.from_numpy(np.full(_N, 7.0, dtype=np.float32))
    vals.from_numpy(np.full(_N, 1.0, dtype=np.float32))
    k2(shared, shared, y3, Bias(vals))  # aliased a/b -> fallback build must not prune add_bias's `bias.vals`
    assert k2._primal._compiled_no_split_by_key, "aliased launch did not build the whole-kernel fallback"
    # base is snapshot 7.0 (whole-kernel), so y = 7.0 + 1.0; the split miscompile would give 2.0 + 1.0.
    assert np.allclose(y3.to_numpy(), 8.0, atol=1e-2), y3.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_carried_rmw_local() -> None:
    # Two constructs each read-modify-write the same local `s`, and the second also stores it. The second construct
    # depends on the value the first produced, so it is not recomputable per-construct (its slice would drop the first
    # loop and restart `s` from the serial init). Checking readers against the *union* of writer constructs would
    # wrongly accept this (both readers are also writers), so the gate must reject it and fall back.
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

    # Atomic runs exactly once: counter == 1, every out[i] the pre-increment 0.
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

    # Both loops must observe the same sample (recomputing per construct would draw twice).
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

    # snap captured x[0] == 7.0 before the deactivation, so every y[i] must be 7.0, not 0.
    assert np.allclose(y.to_numpy(), 7.0), y.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False, require=qd.extension.sparse)
def test_per_construct_frontend_split_fallback_append_result_shared() -> None:
    # `idx` captures the index a top-level `qd.append` returns -- an `SNodeOpStmt::allocate` that writes its result
    # through a local alloca with no `LocalStoreStmt`. Unless the append is tracked as a local writer, a later reader's
    # slice pulls a zero-init alloca and the effectful-producer gate never sees it. Must fall back.
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

    # The append returns the pre-append length (3 prefilled), so every out[i] must be 3, not 0.
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
    # The split still ran (not disabled) and wrote a per-construct IR snapshot for each construct.
    assert obs.frontend_constructs_total == 2, obs
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

    # The real_func wrote _C[0] into `a` before the loop, so every out[i] must be _C[0], not the initial 5.0.
    assert np.allclose(out.to_numpy(), _C[0], atol=1.0), out.to_numpy()


@test_utils.test(arch=[qd.cpu, qd.cuda], offline_cache=False)
def test_per_construct_frontend_split_fallback_realfunc_ref_read() -> None:
    # A loop-carried local is read READ-ONLY by a `@qd.real_func` through a `qd.ref` arg in a later loop. That read is a
    # `ReferenceStmt` (a Load) with no store destination, so a store-only scan would miss it and the loop-carried-local
    # gate would wrongly accept the split. Detection must go through the shared Load trait.
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

    # `acc` accumulates to _N over the first loop, so every out[i] must be _N, not the 0.0 init.
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

    # The external call wrote 6.0 into `r` before the loop, so every out[i] must be 6.0, not the initial 0.0.
    assert np.allclose(arr.to_numpy(), 6.0, atol=1e-3), arr.to_numpy()


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_per_construct_frontend_split_graph_do_while() -> None:
    # A `qd.graph.do_while` body is a host-driven, contiguous same-level task run (see `construct_gdw_level`). The split
    # must fire on it (not fall back) and reattach each construct's level so the body stays contiguous; a mis-leveled
    # task strands the loop counter outside the body and the kernel spins forever. Assert the split fires and the loop
    # is still correct: it runs 3 times incrementing x, so x must end at 3.
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
    assert obs.frontend_constructs_total >= 1, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs
    assert np.all(x.to_numpy() == 3), x.to_numpy()


@test_utils.test(arch=qd.cuda, offline_cache=False)
def test_per_construct_frontend_split_nested_graph_do_while() -> None:
    # Nested `qd.graph.do_while`: the transformer flattens the nest into single-level constructs before the split, so
    # the inner-body work carries level 1 and the outer work level 0. This exercises `construct_gdw_level`'s container
    # branch at a non-zero level; collapsing the two levels would miscount the inner loop or strand a counter. Assert
    # the split fires and the nest is correct: x increments once per (outer, inner), ending at _OUTER * _INNER.
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
    assert obs.frontend_constructs_total >= 1, obs
    assert obs.frontend_constructs_recompiled == obs.frontend_constructs_total, obs
    assert obs.frontend_constructs_cache_hit == 0, obs
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


# --- Cross-process per-task artifact cache (CUDA + AMDGPU backend reuse tier) -----------------------------------------
#
# The per-task artifact cache stores each offloaded task's fully compiled code + launch metadata on disk, keyed by the
# task's own IR (name-free), so a later process reuses an unchanged task instead of recompiling it. CUDA and AMDGPU
# fill it, and it is gated on `offline_cache`. Reuse is reported on `PerOffloadCacheObservations.tasks_*` (-1 when the
# tier did not run).


@test_utils.test(arch=[qd.cuda, qd.amdgpu], offline_cache=False)
def test_per_task_artifact_cache_disabled_without_offline_cache() -> None:
    # `offline_cache` is the sole gate for the per-task disk tier; with it off the tier never runs, so the per-task
    # counts stay at the -1 sentinel. The FRONTEND split is independent of the flag and still fires -- asserting both
    # shows the flag gates only the disk tier, not the split.
    @qd.kernel
    def kernel_two_loops(x: qd.types.ndarray(qd.f32, ndim=1)) -> None:
        for i in x:
            x[i] = x[i] * 2.0 + 1.0
        for i in x:
            x[i] = x[i] - 3.0

    arr = qd.ndarray(qd.f32, shape=(_N,))
    arr.from_numpy(np.arange(_N, dtype=np.float32))
    kernel_two_loops(arr)

    obs = kernel_two_loops._primal.per_offload_cache_observations
    assert obs.frontend_constructs_total == 2, obs
    assert obs.tasks_total == -1, obs
    assert obs.tasks_cache_hit == -1, obs
    assert obs.tasks_recompiled == -1, obs
    assert np.allclose(arr.to_numpy(), np.arange(_N) * 2.0 + 1.0 - 3.0), arr.to_numpy()


@pytest.mark.parametrize("arch", [qd.cuda, qd.amdgpu])
def test_per_task_artifact_cache_reuses_shared_task_cross_process(arch) -> None:
    # Two DIFFERENT kernels sharing a byte-identical first loop. A fresh runtime (cold in-memory, warm disk) compiling
    # the second must load that shared task's artifact from disk -- written when the first ran in the prior "process" --
    # while recompiling only the loop that differs. The per-task key is name-free, so the identical task aliases to one
    # artifact across the two kernels. Uses a re-`init` with the same cache path to emulate a second process, matching
    # test_offline_cache.py; the artifacts are written at JIT time, so no teardown is needed to flush them.
    if arch not in test_utils.expected_archs():
        pytest.skip(f"per-task artifact cache backend {arch} not available")

    cache_dir = tempfile.mkdtemp()
    try:
        qd.init(arch=arch, offline_cache=True, offline_cache_file_path=cache_dir)

        @qd.kernel
        def k_first(x: qd.types.ndarray(qd.f32, ndim=1)) -> None:
            for i in x:
                x[i] = x[i] * 2.0 + 1.0
            for i in x:
                x[i] = x[i] - 3.0

        a = qd.ndarray(qd.f32, shape=(_N,))
        a.from_numpy(np.arange(_N, dtype=np.float32))
        k_first(a)
        obs1 = k_first._primal.per_offload_cache_observations
        assert obs1.tasks_total >= 2, obs1
        assert obs1.tasks_cache_hit == 0, obs1

        # Second "process": fresh runtime, same disk. `k_second` is a new kernel (whole-kernel entry misses, codegen
        # runs), but its first loop matches `k_first`'s, so that task is served from disk.
        qd.init(arch=arch, offline_cache=True, offline_cache_file_path=cache_dir)

        @qd.kernel
        def k_second(x: qd.types.ndarray(qd.f32, ndim=1)) -> None:
            for i in x:
                x[i] = x[i] * 2.0 + 1.0
            for i in x:
                x[i] = x[i] + 7.0

        b = qd.ndarray(qd.f32, shape=(_N,))
        b.from_numpy(np.arange(_N, dtype=np.float32))
        k_second(b)
        obs2 = k_second._primal.per_offload_cache_observations
        assert obs2.tasks_cache_hit > 0, obs2
        assert obs2.tasks_recompiled >= 1, obs2
        assert obs2.tasks_cache_hit + obs2.tasks_recompiled == obs2.tasks_total, obs2

        assert np.allclose(b.to_numpy(), np.arange(_N) * 2.0 + 1.0 + 7.0), b.to_numpy()
    finally:
        qd.reset()
        shutil.rmtree(cache_dir, ignore_errors=True)
