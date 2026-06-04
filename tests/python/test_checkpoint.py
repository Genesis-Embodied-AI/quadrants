"""Tests for qd.checkpoint -- yield/resume stage primitive for graph kernels.

These tests cover *slice 1a* of the qd.checkpoint implementation: Python API surface plus AST
recognition only. The runtime does not yet build per-checkpoint IF conditional nodes, insert
yield-check kernels, or expose a host-side yield/resume loop -- so the kernels here run
through every checkpoint body unconditionally, exactly like a non-checkpointed kernel. The
tests therefore verify:

  - The ``qd.checkpoint`` symbol is importable and usable as a context manager outside of
    kernels (no-op behaviour).
  - A graph kernel with ``with qd.checkpoint():`` blocks parses successfully and runs every
    body kernel.
  - The kernel object records one entry per ``with`` block in
    ``kernel.checkpoint_yield_on_args`` (the ``cp_id``-indexed yield_on metadata that later
    slices will plumb through to the IR / runtime).
  - Misuse at AST time raises ``QuadrantsSyntaxError`` with the documented messages.

Later slices add separate test files / test cases for the runtime mechanism, the yield-check
kernel, the host-side status-object loop, and per-backend fallbacks.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_checkpoint_is_no_op_outside_kernels():
    """At Python runtime (outside kernels) qd.checkpoint must be a usable no-op context manager.

    Lets downstream consumers import the symbol unconditionally and use it inside helpers that
    are sometimes called from Python and sometimes from kernels. Mirrors how qd.stream_parallel
    behaves outside of @qd.kernel.
    """
    sentinel = []
    with qd.checkpoint():
        sentinel.append("body ran")
    with qd.checkpoint(yield_on=None):
        sentinel.append("body ran")
    assert sentinel == ["body ran", "body ran"]


@test_utils.test()
def test_checkpoint_kernel_runs_all_bodies():
    """Slice 1a: a graph kernel with checkpoints runs every body kernel (no IF / yield yet).

    Three checkpoints, each increments x by 1. Without the runtime-side IF mechanism we expect
    every body to execute on every launch, so x ends at 3.
    """
    N = 16

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))

    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))


@test_utils.test()
def test_checkpoint_records_yield_on_metadata():
    """The kernel object records one cp_id entry per `with qd.checkpoint(...)` in source order.

    This is the metadata that slice 1b will read when assigning cp_id to each for-loop and
    that the runtime (slices 1c/1d) reads to wire up yield-check kernels per checkpoint.
    """
    N = 4

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        first_flag: qd.types.ndarray(qd.i32, ndim=0),
        second_flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = 1
        with qd.checkpoint(yield_on=first_flag):
            for i in range(x.shape[0]):
                x[i] = 2
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = 3
        with qd.checkpoint(yield_on=second_flag):
            for i in range(x.shape[0]):
                x[i] = 4

    x = qd.ndarray(qd.i32, shape=(N,))
    first_flag = qd.ndarray(qd.i32, shape=())
    second_flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    first_flag.from_numpy(np.array(0, dtype=np.int32))
    second_flag.from_numpy(np.array(0, dtype=np.int32))

    k(x, first_flag, second_flag)

    assert k._primal.checkpoint_yield_on_args == [None, "first_flag", None, "second_flag"]


@test_utils.test()
def test_checkpoint_inside_graph_do_while_runs():
    """A checkpoint inside a qd.graph_do_while body is the canonical qipc pattern.

    Slice 1a doesn't yet enforce IF / yield, so this is just a smoke test that parsing the
    combination succeeds and the body kernels run as expected for the configured iteration
    count. The runtime semantics (skipping checkpoints below resume_point, yielding on flag)
    arrive in slices 1c/1d.
    """
    N = 8

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.checkpoint():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1
            with qd.checkpoint(yield_on=flag):
                for i in range(1):
                    counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(5, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))

    k(x, counter, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 5, dtype=np.int32))
    assert counter.to_numpy() == 0
    assert k._primal.checkpoint_yield_on_args == [None, "flag"]


@test_utils.test()
def test_checkpoint_without_graph_true_raises():
    """qd.checkpoint() is only meaningful in graph kernels; outside graph mode it must error."""

    @qd.kernel
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"requires @qd.kernel\(graph=True\)"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_yield_on_nonexistent_arg_raises():
    """yield_on must name a kernel parameter; typos / scope mismatches must error early."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=missing_flag):  # noqa: F821 - intentional bad reference
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="does not match any parameter"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_nested_raises():
    """Per design doc 8.2: checkpoints inside other checkpoints are forbidden at compile time."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1)):
        with qd.checkpoint():
            with qd.checkpoint():
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    with pytest.raises(qd.QuadrantsSyntaxError, match="cannot be nested"):
        k(x)


@test_utils.test()
def test_checkpoint_positional_arg_raises():
    """qd.checkpoint() takes no positional args; clarify the message at the call site."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(flag):  # type: ignore[call-arg]
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="takes no positional arguments"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_unexpected_kwarg_raises():
    """Only yield_on= is accepted; other kwargs must error so typos surface immediately."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(when=flag):  # type: ignore[call-arg]
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="unexpected keyword argument"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_yield_on_must_be_bare_name():
    """yield_on= must be a bare parameter name -- expressions / attributes are not supported.

    Keeps the parser path symmetric with qd.graph_do_while(name) and avoids the cost of trying
    to resolve arbitrary AST expressions to kernel parameters at compile time.
    """

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=flag.something):  # type: ignore[union-attr]
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="bare name of a kernel parameter"):
        k(x, flag)
