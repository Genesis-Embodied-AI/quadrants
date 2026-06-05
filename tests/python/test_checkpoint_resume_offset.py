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

All tests gate on the CUDA-native IF path (SM 9.0+) since slice 1d's yield mechanism is the
only place these semantics are enforced today. On other backends every checkpoint body runs
unconditionally, which would make the counter assertions fail spuriously.

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


def _is_checkpoint_if_path_native():
    return impl.current_cfg().arch == qd.cuda and qd.lang.impl.get_cuda_compute_capability() >= 90


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
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

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
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

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
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

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
            with qd.checkpoint():  # cp 2: decrement counter in same checkpoint so a yielded
                # iteration does NOT advance the loop counter (matches qipc's check_iter
                # sequencing where the iter++ happens after the yield-bearing work).
                for i in range(c.shape[0]):
                    c[i] = c[i] + 1
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
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

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
