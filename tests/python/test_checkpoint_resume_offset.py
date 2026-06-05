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
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _is_checkpoint_if_path_native():
    return impl.current_cfg().arch == qd.cuda and qd.lang.impl.get_cuda_compute_capability() >= 90


N_COUNTERS = 4


def _make_buffers(num_counters: int = N_COUNTERS):
    counters = qd.ndarray(qd.i32, shape=(num_counters,))
    flag = qd.ndarray(qd.i32, shape=())
    counters.from_numpy(np.zeros(num_counters, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))
    return counters, flag


@test_utils.test()
def test_resume_offset_sequential_this():
    """Scenario A, YieldResume::This: yielding cp re-runs from itself on the resume launch.

    Three sequential checkpoints (no WHILE). cp 0 yields on the first launch; the host loop
    calls `resume(from_checkpoint=0)`, which re-runs cp 0 (this time without yielding because
    the yield-check kernel cleared the flag), cp 1, and cp 2.

    Expected: counter[0] hit twice (yield launch + resume launch), counter[1] and counter[2]
    hit once each (resume launch only). Host loop sees one yield.
    """
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

    @qd.kernel(graph=True)
    def step(counters: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=flag):  # cp 0
            counters[0] = counters[0] + 1
        with qd.checkpoint():  # cp 1
            counters[1] = counters[1] + 1
        with qd.checkpoint():  # cp 2
            counters[2] = counters[2] + 1

    counters, flag = _make_buffers()
    status = step(counters, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        # YieldResume::This: re-run from the yielding checkpoint itself.
        status = step.resume(counters, flag, from_checkpoint=status.checkpoint)
    h = counters.to_numpy()
    assert h.tolist() == [2, 1, 1, 0], f"counters mismatch: {h.tolist()}"
    assert host_callbacks == 1


@test_utils.test()
def test_resume_offset_sequential_next():
    """Scenario A, YieldResume::Next: yielding cp is skipped on the resume launch.

    Three sequential checkpoints (no WHILE). cp 1 yields on the first launch; the host loop
    calls `resume(from_checkpoint=2)`, which skips cp 0 and cp 1, and runs cp 2 only.

    Expected: counter[0] hit once (yield launch), counter[1] hit once (yield launch),
    counter[2] hit once (resume launch). Host loop sees one yield.
    """
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

    @qd.kernel(graph=True)
    def step(counters: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint():  # cp 0
            counters[0] = counters[0] + 1
        with qd.checkpoint(yield_on=flag):  # cp 1
            counters[1] = counters[1] + 1
        with qd.checkpoint():  # cp 2
            counters[2] = counters[2] + 1

    counters, flag = _make_buffers()
    status = step(counters, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        # YieldResume::Next: skip the yielding checkpoint as well.
        status = step.resume(counters, flag, from_checkpoint=status.checkpoint + 1)
    h = counters.to_numpy()
    assert h.tolist() == [1, 1, 1, 0], f"counters mismatch: {h.tolist()}"
    assert host_callbacks == 1


@test_utils.test()
def test_resume_offset_loop_this():
    """Scenario B, YieldResume::This: yield inside a `qd.graph_do_while` re-runs the yielding cp.

    Three checkpoints inside a WHILE loop of 3 iterations. cp 1 yields on the first iteration
    of the first launch; the WHILE exits early, the host loop calls `resume(from_checkpoint=1)`,
    which on its first WHILE iteration skips cp 0, runs cp 1, cp 2, and decrements the counter
    (last checkpoint). On subsequent iterations the cond-with-yield kernel resets resume_point
    to 0, so cp 0 runs again normally. Total 3 successful WHILE iterations.

    Expected: cp 0 hit 3 times (iter 1 yield-launch + iter 2 + iter 3 in resume launch),
    cp 1 hit 4 times (1 yield + 3 successful), cp 2 hit 3 times.
    Host loop sees one yield.
    """
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

    @qd.kernel(graph=True)
    def step(
        counters: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.checkpoint():  # cp 0
                counters[0] = counters[0] + 1
            with qd.checkpoint(yield_on=flag):  # cp 1
                counters[1] = counters[1] + 1
            with qd.checkpoint():  # cp 2: also decrement WHILE counter so the loop terminates
                counters[2] = counters[2] + 1
                counter[()] = counter[()] - 1

    counters, flag = _make_buffers()
    counter = qd.ndarray(qd.i32, shape=())
    counter.from_numpy(np.array(3, dtype=np.int32))
    status = step(counters, counter, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        status = step.resume(counters, counter, flag, from_checkpoint=status.checkpoint)
    h = counters.to_numpy()
    assert h.tolist() == [3, 4, 3, 0], f"counters mismatch: {h.tolist()}"
    assert host_callbacks == 1
    assert counter.to_numpy() == 0


@test_utils.test()
def test_resume_offset_loop_next():
    """Scenario B, YieldResume::Next: yield inside `qd.graph_do_while` skips the yielding cp on resume.

    Same setup as `test_resume_offset_loop_this` but the host loop calls
    `resume(from_checkpoint=cp + 1)`. cp 2 (which decrements the WHILE counter) yields on the
    first iteration of the first launch; the host loop calls `resume(from_checkpoint=3)` which
    skips every checkpoint (cp 0, cp 1, cp 2) on its first WHILE iteration. The cond-with-yield
    kernel resets resume_point=0 at end of that iteration, so iterations 2 and 3 run the full
    body. Total WHILE iterations: 1 yield + 0 decremented + 2 full = counter decremented 2x
    after the resume call returns, so counter starts at 3, ends at 1. Host loop only sees one
    yield because the next iter doesn't re-yield (flag cleared by yield-check kernel).

    NB: counter ends at 1 not 0 here because the resume's first iteration skips cp 2 (which
    holds the decrement). To terminate, callers would either decrement on host between
    resumes, or move the decrement out of any checkpoint. This test stays faithful to the
    qipc port to demonstrate the semantics, not to be a complete real-world pattern.
    """
    if not _is_checkpoint_if_path_native():
        pytest.skip("resume-offset semantics only validated on the CUDA-native IF path")

    @qd.kernel(graph=True)
    def step(
        counters: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.checkpoint():  # cp 0
                counters[0] = counters[0] + 1
            with qd.checkpoint():  # cp 1
                counters[1] = counters[1] + 1
            with qd.checkpoint(yield_on=flag):  # cp 2
                counters[2] = counters[2] + 1
                counter[()] = counter[()] - 1

    counters, flag = _make_buffers()
    counter = qd.ndarray(qd.i32, shape=())
    counter.from_numpy(np.array(3, dtype=np.int32))
    status = step(counters, counter, flag)
    host_callbacks = 0
    while status.yielded:
        host_callbacks += 1
        # YieldResume::Next: skip the yielding cp 2 entirely on the first resume iteration.
        status = step.resume(counters, counter, flag, from_checkpoint=status.checkpoint + 1)
    h = counters.to_numpy()
    # iter 1 (yield launch): cp 0 +=1, cp 1 +=1, cp 2 +=1 (yields). counter not decremented?
    # Actually it IS decremented because cp 2 ran before yielding; the yield-check kernel runs
    # AFTER cp 2's body. So counter goes 3 -> 2.
    # Resume launch iter 1: from_checkpoint=3 -> all three checkpoints skipped. counter stays 2.
    # cond-with-yield resets resume_point=0.
    # Resume launch iter 2: full body. counter 2 -> 1. cp 0/1/2 each += 1.
    # Resume launch iter 3: full body. counter 1 -> 0. cp 0/1/2 each += 1.
    # Total: cp 0 hit 1+0+1+1 = 3, cp 1 hit 3, cp 2 hit 3.
    assert h.tolist() == [3, 3, 3, 0], f"counters mismatch: {h.tolist()}"
    assert host_callbacks == 1
    assert counter.to_numpy() == 0
