"""Slice 3: integration / end-to-end tests for the qd.checkpoint yield-resume loop.

Mirrors the scenarios from qipc's ``test_resume_offset.cu`` (Sequential A + WHILE B) ported to
Quadrants' ``qd.checkpoint`` / ``kernel.resume(from_checkpoint=...)`` API. Each scenario sets
up a kernel with three checkpoints, primes a yield flag, and drives a host loop that mirrors
the qipc ``YieldResume::This`` / ``YieldResume::Next`` mapping:

- ``YieldResume::This``  -> ``from_checkpoint=status.checkpoint``      (re-run the yielding cp)
- ``YieldResume::Next``  -> ``from_checkpoint=status.checkpoint + 1``  (skip the yielding cp)

The qipc test's nested-WHILE Scenario C is out of scope for now because Quadrants does not
support nesting `qd.graph_do_while` (or putting a graph_do_while inside a `qd.checkpoint`); we
revisit this once those primitives compose.

All tests gate on backends that implement the host-side yield/resume contract: CUDA
native (slice 1d, IF + yield-check device kernels for SM 9.0+) and CPU/x64 (slice 6,
host-branch gating + yield emulation in the CPU `KernelLauncher`). On other backends every
checkpoint body runs unconditionally, which would make the counter assertions fail
spuriously; those are gated out by `_supports_checkpoint_yield_resume()` below.

Important: every checkpoint body must contain at least one top-level for-loop to materialise
as a distinct offloaded task. A bare scalar assignment like `x[0] = x[0] + 1` collapses with
its surrounding work into a single ``..._serial`` task with cp_id=-1, which the GraphManager
treats as "no checkpoint" -- the yield mechanism then can't fire. The tests below use
``for i in range(N): x[i] = ...`` to side-step that and stay focused on the slice 2/3 host API.
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _supports_checkpoint_yield_resume():
    # CUDA-native IF + yield-check device kernels (slice 1d): SM 9.0+ only. On older CUDA the
    # conditional-node primitive isn't available so the gating falls back to "run everything".
    if impl.current_cfg().arch == qd.cuda:
        return qd.lang.impl.get_cuda_compute_capability() >= 90
    # CPU host-branch gating + yield emulation in `runtime/cpu/kernel_launcher.cpp` (slice 6). The
    # launcher is arch-agnostic; same code path covers Linux x86 (`qd.x64`) and Apple Silicon
    # (`qd.arm64`).
    if impl.current_cfg().arch in (qd.x64, qd.arm64):
        return True
    # AMDGPU host-orchestrated sub-graph gating in `GraphManager::launch_cached_checkpoint_graph`
    # (slice 4); also covers WHILE via the streaming launcher's port of the same gating.
    if impl.current_cfg().arch == qd.amdgpu:
        return True
    # Vulkan / Metal share the GFX runtime, which gates per-task at `launch_kernel` time and reads
    # the user's `yield_on=` flag through `readback_data` between dispatches (slice 4 cont.).
    if impl.current_cfg().arch in (qd.vulkan, qd.metal):
        return True
    return False


def _supports_checkpoint_yield_resume_in_while_loop():
    # `graph_do_while + qd.checkpoint(yield_on=...)` semantics. On AMDGPU these kernels fall through
    # to the streaming launcher (HIP has no conditional graph nodes / indirect dispatch as of ROCm
    # 7.2); the streaming launcher implements the same host-branch gating + per-iter resume_point
    # reset that the CPU launcher does (slice 4 / slice 6 share the contract). So as of slice 4 the
    # AMDGPU answer is the same as the wider predicate above.
    return _supports_checkpoint_yield_resume()


N = 8


def _make_buffers():
    counters_a = qd.ndarray(qd.i32, shape=(N,))
    counters_b = qd.ndarray(qd.i32, shape=(N,))
    counters_c = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    counters_a.from_numpy(np.zeros(N, dtype=np.int32))
    counters_b.from_numpy(np.zeros(N, dtype=np.int32))
    counters_c.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))
    return counters_a, counters_b, counters_c, flag


@test_utils.test()
def test_resume_offset_sequential_this():
    """Scenario A, YieldResume::This: yielding cp re-runs from itself on the resume launch.

    Three sequential checkpoints (no WHILE). cp 0 yields on the first launch; the host loop
    calls `resume(from_checkpoint=0)`, which re-runs cp 0 (this time without yielding because
    the yield-check kernel cleared the flag), cp 1, and cp 2.

    Expected: counter A hit twice (yield launch + resume launch), B and C hit once each
    (resume launch only). Host loop sees one yield.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip(
            "resume-offset semantics only validated on backends with checkpoint yield/resume support (CUDA SM 9.0+ or CPU)"
        )

    @qd.kernel(graph=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint(yield_on=flag):  # cp 0
            for i in range(a.shape[0]):
                a[i] = a[i] + 1
        with qd.checkpoint():  # cp 1
            for i in range(b.shape[0]):
                b[i] = b[i] + 1
        with qd.checkpoint():  # cp 2
            for i in range(c.shape[0]):
                c[i] = c[i] + 1

    a, b, c, flag = _make_buffers()
    status = step(a, b, c, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        # YieldResume::This: re-run from the yielding checkpoint itself.
        status = step.resume(a, b, c, flag, from_checkpoint=status.checkpoint)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 2, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 1, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 1, dtype=np.int32))
    assert host_callbacks == 1


@test_utils.test()
def test_resume_offset_sequential_next():
    """Scenario A, YieldResume::Next: yielding cp is skipped on the resume launch.

    Three sequential checkpoints (no WHILE). cp 1 yields on the first launch; the host loop
    calls `resume(from_checkpoint=2)`, which skips cp 0 and cp 1, and runs cp 2 only.

    Expected: A hit once (yield launch), B hit once (yield launch), C hit once (resume
    launch). Host loop sees one yield.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip(
            "resume-offset semantics only validated on backends with checkpoint yield/resume support (CUDA SM 9.0+ or CPU)"
        )

    @qd.kernel(graph=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint():  # cp 0
            for i in range(a.shape[0]):
                a[i] = a[i] + 1
        with qd.checkpoint(yield_on=flag):  # cp 1
            for i in range(b.shape[0]):
                b[i] = b[i] + 1
        with qd.checkpoint():  # cp 2
            for i in range(c.shape[0]):
                c[i] = c[i] + 1

    a, b, c, flag = _make_buffers()
    status = step(a, b, c, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        # YieldResume::Next: skip the yielding checkpoint as well.
        status = step.resume(a, b, c, flag, from_checkpoint=status.checkpoint + 1)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 1, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 1, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 1, dtype=np.int32))
    assert host_callbacks == 1


@test_utils.test()
def test_resume_offset_loop_this():
    """Scenario B, YieldResume::This: yield inside a `qd.graph_do_while` re-runs the yielding cp.

    Three checkpoints inside a WHILE loop of 3 iterations. cp 1 yields on the first iteration
    of the first launch; the WHILE exits early. Host calls `resume(from_checkpoint=1)`. On the
    resume's first iteration, cp 0 is skipped (resume_point=1) and cp 1, cp 2, counter-decrement
    run. The cond-with-yield kernel resets resume_point to 0 at end of that iteration, so iters
    2 and 3 run the full body.

    Expected: A hit 3x (yield iter + 2 full iters in resume), B hit 4x (yield iter + 3 full),
    C hit 3x (3 full iters in resume). Host loop sees one yield.
    """
    if not _supports_checkpoint_yield_resume_in_while_loop():
        pytest.skip(
            "WHILE+yield/resume semantics not yet covered on this backend (e.g. AMDGPU slice 4 streaming-launcher fallback)"
        )

    # Note: every "logical statement" inside a checkpoint body must live in its own top-level
    # `for` loop so the offloader emits it as a range_for task tagged with the surrounding
    # checkpoint_id. Bare scalar assignments fall into the offloader's pending serial bucket
    # which loses checkpoint_id (defaults to -1 = "no checkpoint" = "always runs"). See the
    # module docstring; using `for i in range(1):` is the established workaround (mirrors
    # existing slice 1d tests like `test_checkpoint_yield_exits_graph_do_while_early`).
    @qd.kernel(graph=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.checkpoint():  # cp 0
                for i in range(a.shape[0]):
                    a[i] = a[i] + 1
            with qd.checkpoint(yield_on=flag):  # cp 1
                for i in range(b.shape[0]):
                    b[i] = b[i] + 1
            with qd.checkpoint():  # cp 2: c+=1 and decrement-the-counter, in separate range_fors
                for i in range(c.shape[0]):
                    c[i] = c[i] + 1
                for _ in range(1):
                    counter[()] = counter[()] - 1

    a, b, c, flag = _make_buffers()
    counter = qd.ndarray(qd.i32, shape=())
    counter.from_numpy(np.array(3, dtype=np.int32))
    status = step(a, b, c, counter, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        status = step.resume(a, b, c, counter, flag, from_checkpoint=status.checkpoint)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 3, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 4, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 3, dtype=np.int32))
    assert host_callbacks == 1
    assert counter.to_numpy() == 0


@test_utils.test()
def test_resume_offset_loop_next():
    """Scenario B, YieldResume::Next: yield in `qd.graph_do_while` skips yielding cp on resume.

    Same as `_loop_this` but the yielding cp is cp 2 (which also holds the counter decrement),
    and the host calls `resume(from_checkpoint=cp + 1)`. Layout intentionally mirrors qipc's
    Scenario B Next variant.

    Iter 1 (yield launch): cp 0 +=1, cp 1 +=1, cp 2 body runs (+=1 and counter -=1), then
    yield-check fires (yield_signal=2, resume_point=INT_MAX). cond-with-yield sees yield, exits.
    Status: yielded=True, checkpoint=2.

    Resume(from_checkpoint=3): all three checkpoints skipped on resume's first iter. counter
    not decremented this iter. cond-with-yield resets resume_point=0. Iters 2 and 3 run full.

    Expected: A hit 1+0+1+1 = 3, B hit 3, C hit 3. counter goes 3 -> 2 (iter 1) -> 2 (resume
    iter 1, skipped) -> 1 (iter 2) -> 0 (iter 3). Host loop sees one yield.
    """
    if not _supports_checkpoint_yield_resume_in_while_loop():
        pytest.skip(
            "WHILE+yield/resume semantics not yet covered on this backend (e.g. AMDGPU slice 4 streaming-launcher fallback)"
        )

    # See `_loop_this` comment about why the counter decrement uses `for _ in range(1):`.
    @qd.kernel(graph=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.checkpoint():  # cp 0
                for i in range(a.shape[0]):
                    a[i] = a[i] + 1
            with qd.checkpoint():  # cp 1
                for i in range(b.shape[0]):
                    b[i] = b[i] + 1
            with qd.checkpoint(yield_on=flag):  # cp 2
                for i in range(c.shape[0]):
                    c[i] = c[i] + 1
                for _ in range(1):
                    counter[()] = counter[()] - 1

    a, b, c, flag = _make_buffers()
    counter = qd.ndarray(qd.i32, shape=())
    counter.from_numpy(np.array(3, dtype=np.int32))
    status = step(a, b, c, counter, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        # YieldResume::Next: skip the yielding cp on the resume's first iteration.
        status = step.resume(a, b, c, counter, flag, from_checkpoint=status.checkpoint + 1)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 3, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 3, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 3, dtype=np.int32))
    assert host_callbacks == 1
    assert counter.to_numpy() == 0
