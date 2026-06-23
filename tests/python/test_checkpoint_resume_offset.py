"""End-to-end tests for the ``qd.checkpoint`` yield-resume loop, mirroring the qipc ``test_resume_offset.cu`` scenarios.

Two cross-product axes:

  - **Scenario A**: three sequential checkpoints (no enclosing WHILE).
  - **Scenario B**: three checkpoints inside a ``qd.graph_do_while`` body.

  - **YieldResume::This**: ``kernel.resume(from_checkpoint=status.checkpoint)`` -- re-run the yielding cp.
  - **YieldResume::Next**: the user labels every checkpoint AND the "next" one they want to resume past the yielder
    into; on yield they resume from the latter. With the new auto-wrap API the user can't compute
    ``status.checkpoint + 1`` because labels are opaque and implicit checkpoints between yielders carry ``None`` --
    instead the canonical pattern is to label both the yielder and the immediately-following resume target, then pass
    the target label explicitly.

The qipc test's nested-WHILE Scenario C is out of scope for now because Quadrants does not yet support nesting of
``qd.graph_do_while``.

All tests gate on backends that implement the host-side yield/resume contract; backends without it (e.g. older CUDA)
skip rather than asserting.
"""

from enum import IntEnum

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _supports_checkpoint_yield_resume():
    if impl.current_cfg().arch == qd.cuda:
        return qd.lang.impl.get_cuda_compute_capability() >= 90
    if impl.current_cfg().arch in (qd.x64, qd.arm64):
        return True
    if impl.current_cfg().arch == qd.amdgpu:
        return True
    if impl.current_cfg().arch in (qd.vulkan, qd.metal):
        return True
    return False


def _supports_checkpoint_yield_resume_in_while_loop():
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


class _CP(IntEnum):
    """Source-order checkpoint labels for the scenarios below. The two yielders are LOAD / SIM; AFTER_LOAD / AFTER_SIM
    are explicit resume targets the host loop uses for the ``YieldResume::Next`` variant (skip past the yielder)."""

    LOAD = 0
    AFTER_LOAD = 1
    SIM = 2
    AFTER_SIM = 3


@test_utils.test()
def test_resume_offset_sequential_this():
    """Scenario A, YieldResume::This: yielding cp re-runs from itself on the resume launch.

    Layout: one explicit yielder on the first counter; the second and third counters auto-wrap into implicit
    checkpoints. cp LOAD yields once on the first launch, the host clears the flag and calls
    ``resume(from_checkpoint=LOAD)``, which re-runs LOAD (this time without yielding because the host cleared the flag)
    and the trailing implicit checkpoints.

    Expected: A hit 2x (yield + resume), B and C hit 1x each (resume launch only). One host yield callback.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip(
            "resume-offset semantics only validated on backends with checkpoint yield/resume support (CUDA SM 9.0+ or "
            "CPU)"
        )

    @qd.kernel(graph=True, checkpoints=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint(_CP.LOAD, yield_on=flag):
            for i in range(a.shape[0]):
                a[i] = a[i] + 1
        for i in range(b.shape[0]):  # implicit, follows LOAD
            b[i] = b[i] + 1
        for i in range(c.shape[0]):  # implicit, follows implicit
            c[i] = c[i] + 1

    a, b, c, flag = _make_buffers()
    status = step(a, b, c, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        flag.from_numpy(np.array(0, dtype=np.int32))
        status = step.resume(a, b, c, flag, from_checkpoint=status.checkpoint)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 2, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 1, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 1, dtype=np.int32))
    assert host_callbacks == 1


@test_utils.test()
def test_resume_offset_sequential_next():
    """Scenario A, YieldResume::Next: yielding cp is skipped on the resume launch.

    Layout: an implicit (over A) then the SIM yielder (over B) then an explicit resume target AFTER_SIM (over C). On
    yield from SIM the host loops back with ``resume(from_checkpoint=AFTER_SIM)``, which skips A's implicit, SIM, and
    runs AFTER_SIM. To make AFTER_SIM addressable as a label, it has to be an explicit checkpoint -- a bare for-loop
    would auto-wrap as None-labelled and the user couldn't name it. This is the canonical Next-variant pattern under
    the new label-based API.

    Expected: A hit 1x (yield launch implicit ran), B hit 1x (yield launch SIM ran and then yielded -- body
    already incremented), C hit 1x (resume launch only). One host yield callback.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip(
            "resume-offset semantics only validated on backends with checkpoint yield/resume support (CUDA SM 9.0+ or "
            "CPU)"
        )

    @qd.kernel(graph=True, checkpoints=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        flag: qd.types.ndarray(qd.i32, ndim=0),
        zero: qd.types.ndarray(qd.i32, ndim=0),
    ):
        for i in range(a.shape[0]):  # implicit
            a[i] = a[i] + 1
        with qd.checkpoint(_CP.SIM, yield_on=flag):
            for i in range(b.shape[0]):
                b[i] = b[i] + 1
        with qd.checkpoint(_CP.AFTER_SIM, yield_on=zero):
            for i in range(c.shape[0]):
                c[i] = c[i] + 1

    a, b, c, flag = _make_buffers()
    zero = qd.ndarray(qd.i32, shape=())
    zero.from_numpy(np.array(0, dtype=np.int32))
    status = step(a, b, c, flag, zero)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        flag.from_numpy(np.array(0, dtype=np.int32))
        status = step.resume(a, b, c, flag, zero, from_checkpoint=_CP.AFTER_SIM)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 1, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 1, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 1, dtype=np.int32))
    assert host_callbacks == 1


@test_utils.test()
def test_resume_offset_loop_this():
    """Scenario B, YieldResume::This: yield inside a ``qd.graph_do_while`` re-runs the yielding cp.

    Three checkpoints inside a WHILE loop of 3 iterations: SIM yields on the first iteration of the first launch; the
    WHILE exits early. The host calls ``resume(from_checkpoint=SIM)``. On the resume's first iteration, the leading
    implicit is skipped and SIM, the trailing implicit (with counter decrement) all run. The cond-with-yield kernel
    resets resume_point to 0 at end of that iteration, so iters 2 and 3 run the full body.

    Expected: A hit 1+2=3, B hit 1+3=4, C hit 0+3=3, counter decremented 3x. One host yield callback.
    """
    if not _supports_checkpoint_yield_resume_in_while_loop():
        pytest.skip("WHILE+yield/resume semantics not yet covered on this backend")

    @qd.kernel(graph=True, checkpoints=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            for i in range(a.shape[0]):  # implicit
                a[i] = a[i] + 1
            with qd.checkpoint(_CP.SIM, yield_on=flag):
                for i in range(b.shape[0]):
                    b[i] = b[i] + 1
            for i in range(c.shape[0]):  # implicit (trailing)
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
        flag.from_numpy(np.array(0, dtype=np.int32))
        status = step.resume(a, b, c, counter, flag, from_checkpoint=status.checkpoint)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 3, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 4, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 3, dtype=np.int32))
    assert host_callbacks == 1
    assert counter.to_numpy() == 0


@test_utils.test()
def test_resume_offset_loop_next():
    """Scenario B, YieldResume::Next: yield in ``qd.graph_do_while`` skips yielding cp on resume.

    Same shape as ``_loop_this`` but with an explicit AFTER_SIM checkpoint after the yielder, so the host can resume
    past SIM. The labels make the resume target stable across renumbering of any neighbouring implicit checkpoints.

    Iter 1 (yield launch): implicit (A+=1), SIM (B+=1, yields), exit WHILE.
    Resume(from=AFTER_SIM): iter 1 of resume skips implicit + SIM, runs AFTER_SIM (C+=1) + counter --;     resume_point
    reset to 0; iters 2-3 run full body.
    Expected: A: 1 + 2 = 3, B: 1 + 2 = 3, C: 1 + 2 = 3, counter -> 0, one host yield.
    """
    if not _supports_checkpoint_yield_resume_in_while_loop():
        pytest.skip("WHILE+yield/resume semantics not yet covered on this backend")

    @qd.kernel(graph=True, checkpoints=True)
    def step(
        a: qd.types.ndarray(qd.i32, ndim=1),
        b: qd.types.ndarray(qd.i32, ndim=1),
        c: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
        zero: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            for i in range(a.shape[0]):  # implicit
                a[i] = a[i] + 1
            with qd.checkpoint(_CP.SIM, yield_on=flag):
                for i in range(b.shape[0]):
                    b[i] = b[i] + 1
            with qd.checkpoint(_CP.AFTER_SIM, yield_on=zero):
                for i in range(c.shape[0]):
                    c[i] = c[i] + 1
                for _ in range(1):
                    counter[()] = counter[()] - 1

    a, b, c, flag = _make_buffers()
    counter = qd.ndarray(qd.i32, shape=())
    counter.from_numpy(np.array(3, dtype=np.int32))
    zero = qd.ndarray(qd.i32, shape=())
    zero.from_numpy(np.array(0, dtype=np.int32))
    status = step(a, b, c, counter, flag, zero)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        flag.from_numpy(np.array(0, dtype=np.int32))
        status = step.resume(a, b, c, counter, flag, zero, from_checkpoint=_CP.AFTER_SIM)
    np.testing.assert_array_equal(a.to_numpy(), np.full(N, 3, dtype=np.int32))
    np.testing.assert_array_equal(b.to_numpy(), np.full(N, 3, dtype=np.int32))
    np.testing.assert_array_equal(c.to_numpy(), np.full(N, 3, dtype=np.int32))
    assert host_callbacks == 1
    assert counter.to_numpy() == 0
