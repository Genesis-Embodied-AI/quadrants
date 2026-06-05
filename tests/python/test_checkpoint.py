"""Tests for qd.checkpoint -- yield/resume stage primitive for graph kernels.

These tests cover slices 1a–1d of the qd.checkpoint implementation:

  - *Slice 1a*: Python API surface plus AST recognition. ``qd.checkpoint`` is importable, usable
    as a no-op context manager outside kernels, and parses successfully inside graph kernels.
    Compile-time misuse raises ``QuadrantsSyntaxError``.
  - *Slice 1b*: ``checkpoint_id`` plumbing through the IR (covered by the integration tests
    below, since the IR threading is invisible at the Python boundary).
  - *Slice 1c*: On CUDA SM 9.0+, the GraphManager wires one IF conditional node per checkpoint;
    introspection via ``prog.get_graph_num_checkpoints_on_last_call()`` confirms the IF path.
  - *Slice 1d*: ``yield_on=`` injects a yield-check kernel; on yield the framework atomically
    records the first yielding cp_id, disables every later checkpoint in the launch, clears the
    user's flag, and (inside ``qd.graph_do_while``) exits the WHILE early.

On non-CUDA / pre-SM-9.0 backends the construct is accepted and every body still runs
unconditionally; the introspection-based assertions guard themselves behind
``_is_checkpoint_if_path_native``. The host-side ``GraphStatus`` / ``step.resume(...)`` API
arrives in slice 2.
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _is_checkpoint_if_path_native():
    """The CUDA-native IF-conditional path requires SM 9.0+ / CUDA 12.4+ (slice 1c).

    On other devices/backends the kernel still runs through every checkpoint body, so the
    behavioural tests pass everywhere, but the GraphManager-introspection assertions only
    apply on the native path.
    """
    return _on_cuda() and qd.lang.impl.get_cuda_compute_capability() >= 90


def _supports_checkpoint_yield_resume():
    """Backends that implement the checkpoint yield/resume host contract.

    Wider than `_is_checkpoint_if_path_native()`: also includes the CPU/x64 path, where the
    `KernelLauncher` emulates `resume_point` / `yield_signal` in host-side gating (slice 6).
    Use this predicate for tests of the behavioural yield/resume + `kernel.resume(...)` API;
    use `_is_checkpoint_if_path_native()` only for graph-introspection counters that exist
    on CUDA alone.
    """
    if _is_checkpoint_if_path_native():
        return True
    if impl.current_cfg().arch == qd.x64:
        return True
    return False


def _num_checkpoints_on_last_call():
    return impl.get_runtime().prog.get_graph_num_checkpoints_on_last_call()


def _last_yield_cp_id_on_last_call():
    return impl.get_runtime().prog.get_graph_last_yield_cp_id_on_last_call()


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
def test_checkpoint_emits_if_nodes_on_cuda_native():
    """Slice 1c: on CUDA SM 9.0+, the GraphManager wires one IF conditional node per checkpoint.

    Builds a kernel with three checkpoints and asserts the introspection counter sees three
    IF nodes. On non-CUDA / pre-SM-9.0 backends the kernel still runs but reports 0 since
    the IF path isn't available; the behavioural correctness assertion (incremented N times)
    still holds because the body kernels run unconditionally on those backends.
    """
    N = 8

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
    if _is_checkpoint_if_path_native():
        assert _num_checkpoints_on_last_call() == 3, (
            f"expected 3 IF conditional nodes on the native path, "
            f"got {_num_checkpoints_on_last_call()}"
        )


@test_utils.test()
def test_checkpoint_emits_if_nodes_inside_graph_do_while():
    """Slice 1c: IF conditional nodes nest correctly inside a graph_do_while body.

    Two checkpoints per iteration, three iterations -- the IF nodes live inside the WHILE
    body subgraph and get rebuilt fresh per loop iteration (CUDA semantics). Counter check
    on the native path confirms the GraphManager doesn't accidentally hoist IFs to top level.
    """
    N = 4

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
    counter.from_numpy(np.array(3, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))

    k(x, counter, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))
    assert counter.to_numpy() == 0
    if _is_checkpoint_if_path_native():
        assert _num_checkpoints_on_last_call() == 2, (
            f"expected 2 IF conditional nodes inside the WHILE body, "
            f"got {_num_checkpoints_on_last_call()}"
        )


@test_utils.test()
def test_checkpoint_no_yield_when_flag_is_zero():
    """Slice 1d: with all yield_on flags == 0 the kernel completes normally and reports no yield.

    Sanity check that the yield-check kernel doesn't fire spurious yields and that
    `get_graph_last_yield_cp_id_on_last_call()` returns -1 on the native path when no
    checkpoint requested a yield.
    """
    N = 8

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))

    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.ones(N, dtype=np.int32))
    if _is_checkpoint_if_path_native():
        assert _last_yield_cp_id_on_last_call() == -1


@test_utils.test()
def test_checkpoint_yields_when_flag_is_set():
    """Slice 1d: a non-zero yield_on flag fires the yield-check kernel.

    Pre-set the flag before launch. The yield-check kernel inside the IF body atomically
    records cp_id into `yield_signal`, then bumps `resume_point` so every later checkpoint is
    skipped. The third checkpoint must therefore NOT run on the native path. (On non-native
    backends every body runs unconditionally, so we skip the resume-skip assertion there.)
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip("yield semantics require the CUDA-native IF path (slice 1d) or CPU host-branch gating (slice 6)")
    N = 8

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 100

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))

    k(x, flag)
    # cp 0 ran (+1), cp 1 ran (+10) and signalled a yield, cp 2 was skipped.
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 11, dtype=np.int32))
    assert _last_yield_cp_id_on_last_call() == 1
    # The yield-check kernel must reset the user's flag to 0 so a follow-up call doesn't
    # immediately yield again. Matches docs/source/user_guide/graph.md "yield mechanism".
    assert flag.to_numpy() == 0


@test_utils.test()
def test_checkpoint_yield_first_wins_subsequent_skipped():
    """Slice 1d: when an earlier checkpoint yields, every later checkpoint in the same launch is skipped.

    Three checkpoints: cp 0 and cp 1 both have `yield_on=` set to non-zero before launch. cp 0
    fires first, its yield-check kernel atomically writes cp_id=0 to `yield_signal` and bumps
    `resume_point` to INT_MAX. cp 1's gate kernel then reads INT_MAX, disables the IF (so cp 1's
    body never runs, its flag stays at 1, and its yield-check never fires). cp 2 is likewise
    skipped. This matches the slice 1d design (`perso_hugh/doc/qipc/reentrant.md` section 5.2):
    first yielder wins, everything past the yield point is shipped to the host as-not-run.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip("yield ordering requires the CUDA-native IF path (slice 1d) or CPU host-branch gating (slice 6)")
    N = 4

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        flag_a: qd.types.ndarray(qd.i32, ndim=0),
        flag_b: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint(yield_on=flag_a):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(yield_on=flag_b):
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 100

    x = qd.ndarray(qd.i32, shape=(N,))
    flag_a = qd.ndarray(qd.i32, shape=())
    flag_b = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag_a.from_numpy(np.array(1, dtype=np.int32))
    flag_b.from_numpy(np.array(1, dtype=np.int32))

    k(x, flag_a, flag_b)
    np.testing.assert_array_equal(x.to_numpy(), np.ones(N, dtype=np.int32))
    assert _last_yield_cp_id_on_last_call() == 0
    # cp 0's yield-check cleared its own flag; cp 1's yield-check never ran so its flag stays.
    assert flag_a.to_numpy() == 0
    assert flag_b.to_numpy() == 1


@test_utils.test()
def test_checkpoint_yield_resets_between_launches():
    """Slice 1d: a kernel that yielded once must run cleanly on the next launch when the flag is reset.

    Verifies the per-launch reset path: yield_signal goes back to -1, resume_point goes back
    to 0, the user's yield_on ndarray was cleared by the yield-check kernel during the first
    launch so the second launch doesn't immediately yield again. Same cached graph for both
    launches.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip("yield reset semantics require the CUDA-native IF path (slice 1d) or CPU host-branch gating (slice 6)")
    N = 4

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 10

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))
    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 1, dtype=np.int32))
    assert _last_yield_cp_id_on_last_call() == 0
    # Second launch: flag has been cleared by the yield-check kernel; both bodies should run.
    x.from_numpy(np.zeros(N, dtype=np.int32))
    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 11, dtype=np.int32))
    assert _last_yield_cp_id_on_last_call() == -1


@test_utils.test()
def test_checkpoint_yield_exits_graph_do_while_early():
    """Slice 1d: a yield inside a graph_do_while body terminates the WHILE loop immediately.

    Without the cond-with-yield kernel, the body would re-enter on the next iteration with
    `resume_point == INT_MAX`, skip every checkpoint, never decrement the counter, and spin
    forever. The cond-with-yield variant checks `yield_signal != -1` and exits the WHILE.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip("WHILE early-exit requires the CUDA-native IF path (slice 1d) or CPU host-branch gating (slice 6)")
    N = 4

    @qd.kernel(graph=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.checkpoint(yield_on=flag):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1
            with qd.checkpoint():
                for i in range(1):
                    counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(100, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))

    k(x, counter, flag)
    # x[i] incremented once (cp 0 ran with flag set, yielded), counter NOT decremented (cp 1
    # was skipped because resume_point bumped to INT_MAX), then the WHILE exited because
    # yield_signal != -1.
    np.testing.assert_array_equal(x.to_numpy(), np.ones(N, dtype=np.int32))
    assert counter.to_numpy() == 100
    assert _last_yield_cp_id_on_last_call() == 0


@test_utils.test()
def test_checkpoint_returns_graph_status():
    """Slice 2: kernels with `yield_on=` checkpoints return a `GraphStatus` from every launch.

    No yield this launch -> `status.yielded` is False, `status.checkpoint` is None.
    """
    N = 4

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))

    status = k(x, flag)
    assert isinstance(status, qd.GraphStatus)
    if _is_checkpoint_if_path_native():
        assert status.yielded is False
        assert status.checkpoint is None
    np.testing.assert_array_equal(x.to_numpy(), np.ones(N, dtype=np.int32))


@test_utils.test()
def test_checkpoint_graph_status_reports_yield():
    """Slice 2: a yielding launch returns `GraphStatus(yielded=True, checkpoint=cp_id)`."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("GraphStatus yield reporting requires the CUDA-native IF path (slice 1d) or CPU host-branch gating (slice 6)")
    N = 4

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 10

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))

    status = k(x, flag)
    assert isinstance(status, qd.GraphStatus)
    assert status.yielded is True
    assert status.checkpoint == 1


@test_utils.test()
def test_checkpoint_resume_runs_only_from_checkpoint():
    """Slice 2: `kernel.resume(..., from_checkpoint=cp)` skips every checkpoint with cp_id < cp.

    Three checkpoints, each adds a distinct increment to x. Calling `resume(from_checkpoint=1)`
    should skip cp 0 and run cp 1 + cp 2. With no `yield_on=` flags fired, the resume call
    returns `GraphStatus(yielded=False, checkpoint=None)`.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip("from_checkpoint= requires the CUDA-native IF path (slice 1d) or CPU host-branch gating (slice 6)")
    N = 4

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 100

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))

    # Prime the cached graph with a fresh call first so resume hits a built graph.
    status = k(x, flag)
    assert status.yielded is False
    # After the priming call: x == 111.
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 111, dtype=np.int32))

    x.from_numpy(np.zeros(N, dtype=np.int32))
    status2 = k.resume(x, flag, from_checkpoint=1)
    assert isinstance(status2, qd.GraphStatus)
    assert status2.yielded is False
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 110, dtype=np.int32))


@test_utils.test()
def test_checkpoint_canonical_yield_resume_loop():
    """Slice 2: the canonical qipc-style yield/resume loop from the design doc.

    Kernel runs three checkpoints; cp 1 always yields once. The host loop catches the yield,
    resumes from cp 1, and the second launch completes cleanly.
    """
    if not _supports_checkpoint_yield_resume():
        pytest.skip("yield/resume loop requires the CUDA-native IF path (slice 1d) or CPU host-branch gating (slice 6)")
    N = 4

    @qd.kernel(graph=True)
    def step(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 100

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))

    status = step(x, flag)
    yields_seen = 0
    while status.yielded:
        # In real qipc-style code the host would grow a buffer here; in this test we just
        # decline to re-yield on the resume launch.
        yields_seen += 1
        assert flag.to_numpy() == 0, "yield-check kernel should have cleared the flag"
        status = step.resume(x, flag, from_checkpoint=status.checkpoint)
    assert yields_seen == 1
    # First launch: cp 0 (+1), cp 1 (+10, yields). Resume: cp 1 (+10), cp 2 (+100). Total: 121.
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 121, dtype=np.int32))


@test_utils.test()
def test_checkpoint_resume_invalid_args_raise():
    """Slice 2: misuse of `kernel.resume(from_checkpoint=...)` should fail fast."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    @qd.kernel(graph=True)
    def k_no_yield(x: qd.types.ndarray(qd.i32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(RuntimeError, match="non-negative integer"):
        k.resume(x, flag, from_checkpoint=-1)
    with pytest.raises(RuntimeError, match="non-negative integer"):
        k.resume(x, flag, from_checkpoint="zero")  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="from_checkpoint.* is only valid for kernels"):
        k_no_yield.resume(x, from_checkpoint=0)


@test_utils.test()
def test_checkpoint_non_yield_kernel_returns_none():
    """Kernels with `qd.checkpoint()` but no `yield_on=` keep returning None (no GraphStatus).

    The host-side `GraphStatus` surface is opt-in via `yield_on=` so existing graph kernels
    that just want skippable stages don't change return type out from under their callers.
    """
    N = 4

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1)):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    x.from_numpy(np.zeros(N, dtype=np.int32))
    ret = k(x)
    assert ret is None
    np.testing.assert_array_equal(x.to_numpy(), np.ones(N, dtype=np.int32))


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
