"""Tests for ``qd.checkpoint`` -- yield/resume stage primitive for graph kernels.

These tests cover the auto-checkpoint surface:

  - The user-facing API is ``qd.checkpoint(cp_id, yield_on=flag)``. Both arguments are required; ``cp_id`` is a user
    label (``int`` or ``IntEnum`` value), and ``yield_on`` is a 0-d ``qd.i32`` ndarray kernel parameter.
  - The kernel must be decorated with ``@qd.kernel(graph=True, checkpoints=True)`` to use ``qd.checkpoint(...)``. The
    flag opts the kernel into the resume model and enables the auto-wrap pass.
  - Auto-wrap: every top-level for-loop in the kernel body (including inside ``while qd.graph_do_while(...):``) that
    is not inside a ``with qd.checkpoint(...)`` becomes an implicit no-yield checkpoint. Implicit checkpoints carry no
    user label and never appear in ``GraphStatus.checkpoint``, but they DO consume an internal cp_id slot so a resume
    launch can skip them along with the explicit checkpoints declared earlier in source order.
  - ``status.checkpoint`` round-trips the user-supplied label (so ``qd.checkpoint(Stage.SIM, ...)`` surfaces as
    ``Stage.SIM`` on yield). ``kernel.resume(from_checkpoint=Stage.SIM)`` skips every checkpoint (implicit + explicit)
    declared before ``Stage.SIM`` in source order.

The behavioural assertions (yield, resume, kernel completes normally on no-yield) run on every backend that implements
the host-side yield/resume contract -- see ``_supports_checkpoint_yield_resume`` below. The CUDA-native-only counters
(IF conditional node count) are guarded behind ``_is_checkpoint_if_path_native``.
"""

import os
import pathlib
import subprocess
import sys
from enum import IntEnum

import numpy as np
import pydantic
import pytest

import quadrants as qd
from quadrants.lang import impl

from tests import test_utils

TEST_RAN = "test ran"
RET_SUCCESS = 42


def _on_cuda():
    return impl.current_cfg().arch == qd.cuda


def _is_checkpoint_if_path_native():
    """The CUDA-native IF-conditional path requires SM 9.0+ / CUDA 12.4+ (slice 1c).

    On other devices and backends the kernel still runs through every checkpoint body, so the behavioural tests pass
    everywhere, but the GraphManager-introspection assertions only apply on the native path.
    """
    return _on_cuda() and qd.lang.impl.get_cuda_compute_capability() >= 90


def _supports_checkpoint_yield_resume():
    """Backends that implement the checkpoint yield/resume host contract.

    Wider than `_is_checkpoint_if_path_native()`: also includes the CPU/x64 path (slice 6) and AMDGPU host-orchestrated
    sub-graph path (slice 4). Use this predicate for tests of the behavioural yield/resume + `kernel.resume(...)` API;
    use `_is_checkpoint_if_path_native()` only for graph-introspection counters that exist on CUDA alone.
    """
    if _is_checkpoint_if_path_native():
        return True
    # CPU backend: same `runtime/cpu/kernel_launcher.cpp` host-branch gating runs on both x64 and arm64 (the launcher is
    # arch-agnostic; only the LLVM codegen target differs). Apple Silicon surfaces as `qd.arm64`; Linux x86 as `qd.x64`.
    # Both go through the slice 6 path.
    if impl.current_cfg().arch in (qd.x64, qd.arm64):
        return True
    if impl.current_cfg().arch == qd.amdgpu:
        return True
    # GFX backends (Vulkan, Metal): per-task host gating + readback yield-check in `GfxRuntime` (slice 4 cont.); see
    # `runtime/gfx/runtime.cpp`'s task loop.
    if impl.current_cfg().arch in (qd.vulkan, qd.metal):
        return True
    return False


def _supports_checkpoint_yield_resume_in_while_loop():
    """Strict subset of `_supports_checkpoint_yield_resume`: returns true on backends where yield/resume also works
    inside a `qd.graph_do_while` body. Same predicate today since slice 4 ported the CPU launcher's host-branch gating
    plus per-iter resume_point reset to the AMDGPU streaming path."""
    return _supports_checkpoint_yield_resume()


def _num_checkpoints_on_last_call():
    return impl.get_runtime().prog.get_graph_num_checkpoints_on_last_call()


def _last_yield_cp_id_on_last_call():
    return impl.get_runtime().prog.get_graph_last_yield_cp_id_on_last_call()


# ----------------------------------------------------------------------------------------------------------------------
# Python-runtime surface (works outside @qd.kernel).
# ----------------------------------------------------------------------------------------------------------------------


def test_checkpoint_is_no_op_outside_kernels():
    """At Python runtime (outside kernels) ``qd.checkpoint`` must be a usable no-op context manager.

    Lets downstream consumers import the symbol unconditionally and use it inside helpers that are sometimes called
    from Python and sometimes from kernels. The new API has two required args; both are accepted unchanged here (the
    Python runtime stub is just ``del cp_id, yield_on; yield``).
    """
    sentinel = []
    with qd.checkpoint(0, None):
        sentinel.append("body ran")
    with qd.checkpoint(7, None):
        sentinel.append("body ran")
    assert sentinel == ["body ran", "body ran"]


# ----------------------------------------------------------------------------------------------------------------------
# Decorator + opt-in flag.
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_checkpoint_in_non_checkpoints_kernel_raises():
    """Using ``qd.checkpoint(...)`` in a ``@qd.kernel(graph=True)`` without ``checkpoints=True`` must error at compile
    time, pointing at the fix-it (add ``checkpoints=True``)."""

    @qd.kernel(graph=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"checkpoints=True"):
        k(x, flag)


def test_kernel_checkpoints_requires_graph_true():
    """``@qd.kernel(checkpoints=True)`` without ``graph=True`` is rejected at decorator time -- the resume model is only
    meaningful for graph kernels (the gate / yield-check lowering, the resume_point slot, and the kernel.resume API all
    depend on the graph-capture path)."""
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"checkpoints=True\) requires graph=True"):

        @qd.kernel(checkpoints=True)
        def k(x: qd.types.ndarray(qd.i32, ndim=1)):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1


# ----------------------------------------------------------------------------------------------------------------------
# New API signature: cp_id (int or IntEnum), yield_on (required, must be parameter name).
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_checkpoint_missing_cp_id_raises():
    """``qd.checkpoint()`` with no args must raise with a message that points at the required signature."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1)):
        with qd.checkpoint():
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"qd\.checkpoint\(cp_id, yield_on=flag\)"):
        k(x)


@test_utils.test()
def test_checkpoint_missing_yield_on_raises():
    """``qd.checkpoint(cp_id)`` without ``yield_on=`` is rejected at compile time."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1)):
        with qd.checkpoint(0):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"missing required argument `yield_on`"):
        k(x)


@test_utils.test()
def test_checkpoint_non_int_cp_id_raises():
    """``cp_id`` must be statically determinable to an int / IntEnum value; a string / float / unresolved Name fails."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint("not an int", yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"must be an int literal or an IntEnum value"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_duplicate_cp_id_raises():
    """Each user-supplied ``cp_id`` must be unique within a kernel so the host loop can map ``status.checkpoint`` and
    ``kernel.resume(from_checkpoint=...)`` unambiguously."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(0, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"already used by another checkpoint"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_yield_on_nonexistent_arg_raises():
    """``yield_on`` must reference an ndarray kernel argument; typos / scope mismatches must error early."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=missing_flag):  # noqa: F821 - intentional bad reference
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="does not resolve to an ndarray kernel parameter"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_yield_on_must_be_name_or_attribute():
    """``yield_on=`` must reference an ndarray kernel argument -- either a bare ``ast.Name`` or an ``ast.Attribute``
    chain (for ``@qd.data_oriented`` member ndarrays). Arbitrary expressions are not supported; pinning the diagnostic
    so the user knows to refactor."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag if True else flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"must reference a kernel ndarray argument"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_unexpected_kwarg_raises():
    """Only ``cp_id`` and ``yield_on`` are accepted; other kwargs must error so typos surface immediately."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag, when=flag):  # type: ignore[call-arg]
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="unexpected keyword argument"):
        k(x, flag)


@test_utils.test()
def test_checkpoint_nested_raises():
    """Checkpoints inside other checkpoints are forbidden at compile time."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            with qd.checkpoint(1, yield_on=flag):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match="cannot be nested"):
        k(x, flag)


# ----------------------------------------------------------------------------------------------------------------------
# Auto-wrap pass behaviour.
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_autowrap_assigns_implicit_cp_id_per_for_loop():
    """In a ``checkpoints=True`` kernel, every top-level for-loop not inside a ``with qd.checkpoint(...)`` consumes one
    internal cp_id slot (with a ``None`` user label). Three plain for-loops -> three implicit slots."""
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1)):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
        for i in range(x.shape[0]):
            x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    x.from_numpy(np.zeros(N, dtype=np.int32))
    k(x)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))
    # All three are implicit (no user labels), no yield_on either.
    assert k._primal.checkpoint_user_labels_by_cp_id == [None, None, None]
    assert k._primal.checkpoint_yield_on_args == [None, None, None]


@test_utils.test()
def test_autowrap_mixes_explicit_and_implicit_in_source_order():
    """Mix of explicit yielders and auto-wrapped for-loops produces a dense source-order internal cp_id sequence with
    user labels interleaved with ``None`` for the implicit slots."""
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(10, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        for i in range(x.shape[0]):  # implicit cp_id=1
            x[i] = x[i] + 1
        with qd.checkpoint(20, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        for i in range(x.shape[0]):  # implicit cp_id=3
            x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 4, dtype=np.int32))
    assert k._primal.checkpoint_user_labels_by_cp_id == [10, None, 20, None]
    assert k._primal.checkpoint_yield_on_args == ["flag", None, "flag", None]


@test_utils.test()
def test_autowrap_recurses_into_graph_do_while_body():
    """The auto-wrap pass recurses into ``while qd.graph_do_while(...):`` bodies so for-loops nested inside the WHILE
    body get the same implicit-checkpoint treatment."""
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            for i in range(x.shape[0]):  # implicit
                x[i] = x[i] + 1
            with qd.checkpoint(0, yield_on=flag):
                for _ in range(1):
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
    # One implicit (the bare for-loop) + one explicit yielder.
    assert k._primal.checkpoint_user_labels_by_cp_id == [None, 0]
    assert k._primal.checkpoint_yield_on_args == [None, "flag"]


@test_utils.test()
def test_autowrap_leaves_kernel_prologue_alone():
    """Bare top-level statements (not inside any for-loop) stay in the kernel prologue with cp_id=-1 and run on every
    launch -- including resume launches. Verified indirectly by the fact that the autowrap pass produces only one
    implicit checkpoint here (for the one for-loop), not extra slots for the bare prologue stmt."""
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), n_local: qd.types.ndarray(qd.i32, ndim=0)):
        n_local[()] = x.shape[0]
        for i in range(x.shape[0]):
            x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    n_local = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    n_local.from_numpy(np.array(0, dtype=np.int32))
    k(x, n_local)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 1, dtype=np.int32))
    assert int(n_local.to_numpy()) == N
    # One implicit cp_id for the for-loop. The bare assign is kernel prologue, cp_id=-1, not in the table.
    assert k._primal.checkpoint_user_labels_by_cp_id == [None]


# ----------------------------------------------------------------------------------------------------------------------
# Bare-stmt-inside-checkpoint rule still applies to explicit blocks.
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_checkpoint_bare_assign_raises():
    """A bare ``Assign`` at the top level of an explicit ``with qd.checkpoint(...)`` body must raise
    ``QuadrantsSyntaxError`` at compile time, not be silently wrapped. The user should be steered to ``for _ in
    range(1): <stmt>`` explicitly so they see each top-level statement becomes its own offloaded task."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(c: qd.types.ndarray(qd.i32, ndim=0), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            c[()] = c[()] + 1

    c = qd.ndarray(qd.i32, shape=())
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"bare top-level Assign statement"):
        k(c, flag)


@test_utils.test()
def test_checkpoint_bare_assign_error_message_suggests_for_wrap():
    """The rejection message must show the user the exact ``for _ in range(1):`` rewrite. Pin a representative subset
    of the message so a future copy-edit can't silently drop the actionable fix-it hint."""

    @qd.kernel(graph=True, checkpoints=True)
    def k(c: qd.types.ndarray(qd.i32, ndim=0), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            c[()] = c[()] + 1

    c = qd.ndarray(qd.i32, shape=())
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(qd.QuadrantsSyntaxError, match=r"for _ in range\(1\):"):
        k(c, flag)


@test_utils.test()
def test_checkpoint_docstring_allowed():
    """A docstring at the top of a checkpoint body is allowed (it's a no-op ``Expr(Constant)``)."""
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            """This checkpoint increments x."""
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 1, dtype=np.int32))


# ----------------------------------------------------------------------------------------------------------------------
# IntEnum labels round-trip through GraphStatus and kernel.resume.
# ----------------------------------------------------------------------------------------------------------------------


# Module-level IntEnum so the AST resolver can find it via the kernel's module globals (matches the canonical user
# pattern, and is the only one the resolver supports).
class _Stage(IntEnum):
    LOAD = 0
    SIM = 1
    REDUCE = 2


@test_utils.test()
def test_intenum_label_resolves_to_internal_cp_id():
    """``qd.checkpoint(Stage.SIM, ...)`` resolves the IntEnum to its int value internally and stores the IntEnum
    instance (not the raw int) in the user-label table, so the round-trip through ``GraphStatus.checkpoint`` preserves
    enum identity."""
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(_Stage.LOAD, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(_Stage.REDUCE, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    k(x, flag)
    labels = k._primal.checkpoint_user_labels_by_cp_id
    assert labels == [_Stage.LOAD, _Stage.REDUCE]
    # Identity preserved (not just int equality).
    assert all(isinstance(lbl, _Stage) for lbl in labels)


# ----------------------------------------------------------------------------------------------------------------------
# Yield mechanics (still work end-to-end through the new API).
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_checkpoint_no_yield_when_flag_is_zero():
    """With ``yield_on`` flag == 0 the kernel completes normally and reports no yield. Sanity check that the yield-check
    kernel doesn't fire spurious yields."""
    N = 8

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    status = k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.ones(N, dtype=np.int32))
    assert status is not None and not status.yielded
    assert status.checkpoint is None
    if _is_checkpoint_if_path_native():
        assert _last_yield_cp_id_on_last_call() == -1


@test_utils.test()
def test_checkpoint_yields_when_flag_is_set():
    """A non-zero ``yield_on`` flag fires the yield-check kernel: the first checkpoint records its cp_id into
    ``yield_signal``, bumps ``resume_point`` so every later checkpoint (explicit and implicit) is skipped."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("yield semantics require the CUDA-native IF path or CPU host-branch gating")
    N = 8

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):  # implicit cp_id=0
            x[i] = x[i] + 1
        with qd.checkpoint(7, yield_on=flag):  # explicit cp_id=1, label=7
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        for i in range(x.shape[0]):  # implicit cp_id=2 -- must be skipped on yield
            x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))
    status = k(x, flag)
    # Implicit pre-yielder ran (+1), yielder ran (+1), implicit post-yielder skipped -> total 2.
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 2, dtype=np.int32))
    assert status.yielded
    assert status.checkpoint == 7


@test_utils.test()
def test_checkpoint_yield_first_wins_subsequent_skipped():
    """When two yielders both set their flag in the same launch, the *first* (in source / declaration order) wins:
    ``yield_signal`` is atomic-CAS'd from -1 only by the first writer, later writers see the slot already filled and
    no-op. The second checkpoint's body is itself skipped on the gating prologue, so the test also validates that the
    skip propagates."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("requires yield/resume support")
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        first_flag: qd.types.ndarray(qd.i32, ndim=0),
        second_flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint(_Stage.LOAD, yield_on=first_flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(_Stage.SIM, yield_on=second_flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    first_flag = qd.ndarray(qd.i32, shape=())
    second_flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    first_flag.from_numpy(np.array(1, dtype=np.int32))
    second_flag.from_numpy(np.array(1, dtype=np.int32))
    status = k(x, first_flag, second_flag)
    # First checkpoint yields; second is skipped. x increments by 1 only.
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 1, dtype=np.int32))
    assert status.yielded
    assert status.checkpoint == _Stage.LOAD


@test_utils.test()
def test_checkpoint_yield_does_not_clear_user_flag():
    """The framework never writes into the user's ``yield_on`` buffer -- after a yield the flag retains whatever value
    the body wrote. The host loop is responsible for clearing the flag before ``resume(...)`` (the canonical pattern in
    ``docs/source/user_guide/graph.md``)."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("requires yield/resume support")
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))
    status1 = k(x, flag)
    assert status1.yielded
    assert int(flag.to_numpy()) == 1, "framework must not touch the user's yield_on buffer"
    # Host clears the flag, then a fresh launch (not a resume) completes normally.
    flag.from_numpy(np.array(0, dtype=np.int32))
    status2 = k(x, flag)
    assert not status2.yielded


# ----------------------------------------------------------------------------------------------------------------------
# Resume API: label-based, skips implicit + explicit checkpoints declared before the resume target.
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_resume_by_int_label_skips_implicit_and_explicit_before():
    """``kernel.resume(from_checkpoint=N)`` with a raw int label skips every checkpoint (implicit AND explicit) declared
    earlier in source order. Three pieces of work (implicit, explicit, implicit) and resume from the explicit's label
    skips the first implicit; resume from a later sentinel skips both implicit + the explicit."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("requires yield/resume support")
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):  # implicit cp_id=0
            x[i] = x[i] + 1
        with qd.checkpoint(42, yield_on=flag):  # explicit cp_id=1, label=42
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
        for i in range(x.shape[0]):  # implicit cp_id=2
            x[i] = x[i] + 100

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())

    # Full launch: 1 + 10 + 100 = 111.
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 111, dtype=np.int32))

    # Resume from label=42: skip implicit cp_id=0, run explicit (+10), run trailing implicit (+100). 110.
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    k.resume(x, flag, from_checkpoint=42)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 110, dtype=np.int32))


@test_utils.test()
def test_resume_by_intenum_label_round_trip():
    """The canonical IntEnum-driven host loop: yield from one stage, resume from a later one. The IntEnum identity
    round-trips through ``status.checkpoint``, and ``kernel.resume(from_checkpoint=Stage.X)`` is readable at the call
    site."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("requires yield/resume support")
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(_Stage.LOAD, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(_Stage.SIM, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
        with qd.checkpoint(_Stage.REDUCE, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 100

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())

    # Trigger a yield from LOAD.
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))
    status = k(x, flag)
    assert status.yielded
    assert status.checkpoint is _Stage.LOAD or status.checkpoint == _Stage.LOAD
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 1, dtype=np.int32))

    # Resume from SIM (skip LOAD), no further yields.
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    status = k.resume(x, flag, from_checkpoint=_Stage.SIM)
    assert not status.yielded
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 110, dtype=np.int32))


@test_utils.test()
def test_resume_unknown_label_raises():
    """``kernel.resume(from_checkpoint=<unknown>)`` raises ``RuntimeError`` listing the kernel's available labels."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("requires yield/resume support")
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(10, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    # Warm up so the kernel is compiled and the label table is populated.
    k(x, flag)
    with pytest.raises(RuntimeError, match=r"does not match any qd\.checkpoint\(cp_id=\.\.\.\) in kernel"):
        k.resume(x, flag, from_checkpoint=999)


@test_utils.test()
def test_resume_non_int_arg_raises():
    """``kernel.resume(from_checkpoint=<non-int>)`` is rejected at the host wrapper before reaching compilation."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("requires yield/resume support")

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        with qd.checkpoint(0, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(4,))
    flag = qd.ndarray(qd.i32, shape=())
    with pytest.raises(RuntimeError, match=r"must be an int or IntEnum value"):
        k.resume(x, flag, from_checkpoint="bad")


@test_utils.test()
def test_canonical_yield_resume_loop():
    """The canonical host loop pattern from the docs: launch -> while yielded -> resume(from=status.checkpoint)."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("requires yield/resume support")
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def step(
        x: qd.types.ndarray(qd.i32, ndim=1),
        overflow: qd.types.ndarray(qd.i32, ndim=0),
    ):
        with qd.checkpoint(_Stage.LOAD, yield_on=overflow):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        with qd.checkpoint(_Stage.SIM, yield_on=overflow):
            for i in range(x.shape[0]):
                x[i] = x[i] + 10
        with qd.checkpoint(_Stage.REDUCE, yield_on=overflow):
            for i in range(x.shape[0]):
                x[i] = x[i] + 100

    x = qd.ndarray(qd.i32, shape=(N,))
    overflow = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))

    # Simulate the canonical pattern: yield from LOAD, host "fixes" it and resumes past LOAD; on the resume launch SIM
    # yields once, host "fixes" it and resumes past SIM; on the second resume nothing yields and the loop exits. The
    # "fix" each round is to re-arm overflow so the *next* yielder fires (and resume past LOAD/SIM into the next stage).
    # With the new label API the host advances by passing the next stage explicitly rather than computing
    # `status.checkpoint + 1`.
    resume_targets = {_Stage.LOAD: _Stage.SIM, _Stage.SIM: _Stage.REDUCE}
    overflow.from_numpy(np.array(1, dtype=np.int32))
    status = step(x, overflow)
    expected_sequence = [_Stage.LOAD, _Stage.SIM]
    actual_sequence = []
    while status.yielded:
        actual_sequence.append(status.checkpoint)
        target = resume_targets[status.checkpoint]
        # Arm overflow for the NEXT yielder if there is one left to fire, otherwise clear so the kernel can complete.
        overflow.from_numpy(np.array(1 if target != _Stage.REDUCE else 0, dtype=np.int32))
        status = step.resume(x, overflow, from_checkpoint=target)
    assert actual_sequence == expected_sequence


# ----------------------------------------------------------------------------------------------------------------------
# graph_do_while + checkpoints interaction.
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_checkpoint_inside_graph_do_while_completes_when_no_yield():
    """A checkpoint inside a ``qd.graph_do_while`` body is the canonical qipc pattern. With ``yield_on`` flag == 0 the
    WHILE loop runs to completion as if there were no checkpoint."""
    N = 8

    @qd.kernel(graph=True, checkpoints=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
            with qd.checkpoint(0, yield_on=flag):
                for _ in range(1):
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


@test_utils.test()
def test_checkpoint_yield_exits_graph_do_while_early():
    """When a checkpoint inside a ``graph_do_while`` body yields, the WHILE loop exits even though the user's loop
    condition is still true."""
    if not _supports_checkpoint_yield_resume_in_while_loop():
        pytest.skip("requires yield/resume + graph_do_while support")
    N = 4

    @qd.kernel(graph=True, checkpoints=True)
    def k(
        x: qd.types.ndarray(qd.i32, ndim=1),
        counter: qd.types.ndarray(qd.i32, ndim=0),
        flag: qd.types.ndarray(qd.i32, ndim=0),
    ):
        while qd.graph_do_while(counter):
            with qd.checkpoint(0, yield_on=flag):
                for i in range(x.shape[0]):
                    x[i] = x[i] + 1
            for _ in range(1):
                counter[()] = counter[()] - 1

    x = qd.ndarray(qd.i32, shape=(N,))
    counter = qd.ndarray(qd.i32, shape=())
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    counter.from_numpy(np.array(5, dtype=np.int32))
    flag.from_numpy(np.array(1, dtype=np.int32))
    status = k(x, counter, flag)
    # The checkpoint runs once and yields; then the WHILE exits early. So x is incremented once and counter is not
    # decremented for that iteration (the decrement was skipped post-yield).
    np.testing.assert_array_equal(x.to_numpy(), np.ones(N, dtype=np.int32))
    assert status.yielded
    assert status.checkpoint == 0


# ----------------------------------------------------------------------------------------------------------------------
# Member-ndarray support for `yield_on=` (both `@qd.data_oriented` self-members and `@dataclasses.dataclass` parameter
# members).
#
# ``qd.checkpoint(yield_on=self.flag)`` and ``qd.checkpoint(yield_on=params.flag)`` (where ``params`` is a kernel
# parameter typed as a ``@dataclasses.dataclass``) both resolve the member ndarray to a flat C++ arg-id at AST-build
# time via ``ASTTransformer._resolve_ndarray_kernel_arg_id``: it builds the expression and reads the resolved
# ``ExternalTensorExpression.arg_id``, so any attribute chain that ends up as a kernel ndarray arg works the same way
# as a bare parameter name. This frees users from having to forward flag members as bare kernel parameters when the
# rest of the kernel already operates on the dataclass / data-oriented owner.
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_checkpoint_yield_on_data_oriented_member_metadata():
    """`yield_on=self.flag` is accepted and the resolved label is stored verbatim (``"self.flag"``) in
    ``checkpoint_yield_on_args``, while ``checkpoint_yield_on_cpp_arg_ids`` carries the flat C++ arg-id the runtime
    forwards to the launch context. Verifies the AST-build-time resolution path without booting the backend."""
    N = 4

    @qd.data_oriented
    class Sim:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.flag = qd.ndarray(qd.i32, shape=())

        @qd.kernel(graph=True, checkpoints=True)
        def step(self):
            with qd.checkpoint(0, yield_on=self.flag):
                for i in range(self.x.shape[0]):
                    self.x[i] = self.x[i] + 1

    sim = Sim()
    sim.x.from_numpy(np.zeros(N, dtype=np.int32))
    sim.flag.from_numpy(np.array(0, dtype=np.int32))
    sim.step()
    np.testing.assert_array_equal(sim.x.to_numpy(), np.ones(N, dtype=np.int32))
    assert sim.step._primal.checkpoint_user_labels_by_cp_id == [0]
    assert sim.step._primal.checkpoint_yield_on_args == ["self.flag"]
    cpp_ids = sim.step._primal.checkpoint_yield_on_cpp_arg_ids
    assert len(cpp_ids) == 1 and cpp_ids[0] >= 0


@test_utils.test()
def test_checkpoint_yield_on_dataclass_member_metadata():
    """`yield_on=params.flag` for a ``@dataclasses.dataclass`` kernel parameter takes the same AST-build-time resolution
    path as ``self.flag`` for a ``@qd.data_oriented`` owner -- the resolved label round-trips into
    ``checkpoint_yield_on_args`` and the flat arg-id lands in ``checkpoint_yield_on_cpp_arg_ids``."""
    import dataclasses  # pylint: disable=import-outside-toplevel

    N = 4

    @dataclasses.dataclass
    class Params:
        x: qd.types.NDArray[qd.i32, 1]
        flag: qd.types.NDArray[qd.i32, 0]

    @qd.kernel(graph=True, checkpoints=True)
    def step(params: Params):
        with qd.checkpoint(0, yield_on=params.flag):
            for i in range(params.x.shape[0]):
                params.x[i] = params.x[i] + 1

    params = Params(
        x=qd.ndarray(qd.i32, shape=(N,)),
        flag=qd.ndarray(qd.i32, shape=()),
    )
    params.x.from_numpy(np.zeros(N, dtype=np.int32))
    params.flag.from_numpy(np.array(0, dtype=np.int32))
    step(params)
    np.testing.assert_array_equal(params.x.to_numpy(), np.ones(N, dtype=np.int32))
    assert step._primal.checkpoint_user_labels_by_cp_id == [0]
    # Dataclass-parameter member access gets pre-rewritten by the AST pipeline to a flattened parameter name
    # (`__qd_params__qd_flag`) before the checkpoint transformer sees it, so the label round-trips in the flattened
    # form. The functional contract -- a valid flat C++ arg-id is resolved and the kernel mutates the right ndarray --
    # is the same as for the bare-param / `self.flag` forms.
    labels = step._primal.checkpoint_yield_on_args
    assert len(labels) == 1 and labels[0] is not None and "flag" in labels[0]
    cpp_ids = step._primal.checkpoint_yield_on_cpp_arg_ids
    assert len(cpp_ids) == 1 and cpp_ids[0] >= 0


@test_utils.test()
def test_checkpoint_yield_on_dataclass_member_yields_and_resumes():
    """Behavioural round-trip for `yield_on=params.flag` -- mirror of the `self.flag` test below, using a
    `@dataclasses.dataclass` kernel parameter instead of a `@qd.data_oriented` owner. The dataclass-member access is
    pre-rewritten to a flattened parameter, so verifying the full yield/resume contract end-to-end is the only way to
    confirm the right ndarray is wired up at launch."""
    import dataclasses  # pylint: disable=import-outside-toplevel

    if not _supports_checkpoint_yield_resume():
        pytest.skip("backend does not implement checkpoint yield/resume")
    N = 4

    @dataclasses.dataclass
    class Params:
        x: qd.types.NDArray[qd.i32, 1]
        flag: qd.types.NDArray[qd.i32, 0]

    @qd.kernel(graph=True, checkpoints=True)
    def step(params: Params):
        with qd.checkpoint(7, yield_on=params.flag):
            for i in range(params.x.shape[0]):
                params.x[i] = params.x[i] + 1
                params.flag[()] = 1
        with qd.checkpoint(8, yield_on=params.flag):
            for i in range(params.x.shape[0]):
                params.x[i] = params.x[i] + 10

    params = Params(
        x=qd.ndarray(qd.i32, shape=(N,)),
        flag=qd.ndarray(qd.i32, shape=()),
    )
    params.x.from_numpy(np.zeros(N, dtype=np.int32))
    params.flag.from_numpy(np.array(0, dtype=np.int32))
    status = step(params)
    assert status.yielded
    assert status.checkpoint == 7
    np.testing.assert_array_equal(params.x.to_numpy(), np.ones(N, dtype=np.int32))
    params.flag.from_numpy(np.array(0, dtype=np.int32))
    # `step` is a free-function kernel (not a bound class kernel), so `params` must be passed positionally to
    # `resume` -- the data_oriented sibling test above can omit it because the dataclass member access is implicit
    # through `sim.step`'s bound `self`.
    status = step.resume(params, from_checkpoint=8)
    assert not status.yielded
    np.testing.assert_array_equal(params.x.to_numpy(), np.full(N, 11, dtype=np.int32))


@test_utils.test()
def test_checkpoint_yield_on_member_nonexistent_attribute_raises():
    """`yield_on=self.nonexistent_attr` (attribute does not exist on the `@qd.data_oriented` owner) must raise a user-
    facing `QuadrantsSyntaxError` at the `with` site -- the AST-time resolver wraps the underlying attribute lookup
    failure in the same `does not resolve to an ndarray kernel parameter` diagnostic as the bare-name nonexistent case,
    so users see one consistent error pattern."""
    N = 4

    @qd.data_oriented
    class Sim:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.flag = qd.ndarray(qd.i32, shape=())

        @qd.kernel(graph=True, checkpoints=True)
        def step(self):
            with qd.checkpoint(0, yield_on=self.nonexistent_flag):  # type: ignore[attr-defined]
                for i in range(self.x.shape[0]):
                    self.x[i] = self.x[i] + 1

    sim = Sim()
    with pytest.raises(qd.QuadrantsSyntaxError, match="does not resolve to an ndarray kernel parameter"):
        sim.step()


@test_utils.test()
def test_checkpoint_yield_on_member_non_ndarray_attribute_raises():
    """`yield_on=self.scalar` where `self.scalar` is a Python int (not an ndarray) must raise the same `does not resolve
    to an ndarray kernel parameter` diagnostic -- the AST-time resolver builds the expression but rejects it because the
    resulting Expr is not an `ExternalTensorExpression`. Pinning this so future refactors of the resolver can't silently
    accept non-ndarray attributes and crash later in the launcher."""
    N = 4

    @qd.data_oriented
    class Sim:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.scalar = 7

        @qd.kernel(graph=True, checkpoints=True)
        def step(self):
            with qd.checkpoint(0, yield_on=self.scalar):
                for i in range(self.x.shape[0]):
                    self.x[i] = self.x[i] + 1

    sim = Sim()
    with pytest.raises(qd.QuadrantsSyntaxError, match="does not resolve to an ndarray kernel parameter"):
        sim.step()


@test_utils.test()
def test_checkpoint_yield_on_data_oriented_member_yields_and_resumes():
    """Behavioural round-trip for `yield_on=self.flag`: setting the member flag from inside the kernel yields, and
    ``kernel.resume(from_checkpoint=...)`` skips ahead to the named checkpoint. Same surface contract as the
    bare-parameter form (`test_checkpoint_yield_on_yields_and_resumes`); the only difference is where the flag lives."""
    if not _supports_checkpoint_yield_resume():
        pytest.skip("backend does not implement checkpoint yield/resume")
    N = 4

    @qd.data_oriented
    class Sim:
        def __init__(self):
            self.x = qd.ndarray(qd.i32, shape=(N,))
            self.flag = qd.ndarray(qd.i32, shape=())

        @qd.kernel(graph=True, checkpoints=True)
        def step(self):
            with qd.checkpoint(7, yield_on=self.flag):
                for i in range(self.x.shape[0]):
                    self.x[i] = self.x[i] + 1
                    self.flag[()] = 1
            with qd.checkpoint(8, yield_on=self.flag):
                for i in range(self.x.shape[0]):
                    self.x[i] = self.x[i] + 10

    sim = Sim()
    sim.x.from_numpy(np.zeros(N, dtype=np.int32))
    sim.flag.from_numpy(np.array(0, dtype=np.int32))
    status = sim.step()
    # Checkpoint 7 set the flag in the first iter so the kernel yields before running checkpoint 8.
    assert status.yielded
    assert status.checkpoint == 7
    np.testing.assert_array_equal(sim.x.to_numpy(), np.ones(N, dtype=np.int32))
    # User clears the flag and resumes from the post-yield checkpoint (skipping the +1 loop entirely).
    sim.flag.from_numpy(np.array(0, dtype=np.int32))
    status = sim.step.resume(from_checkpoint=8)
    assert not status.yielded
    np.testing.assert_array_equal(sim.x.to_numpy(), np.full(N, 11, dtype=np.int32))


# Module-level kernel for the fastcache-restoration test below. Lives outside any test so the child subprocess can
# import the test module and reach it without re-creating the (closure-captured) outer scope. The kernel has to be
# annotated with `fastcache=True` (=> implies `pure`) and lifted out of any decorator-bound owner so it qualifies for
# the src_ll_cache path. We model the data_oriented owner as the `_FastcacheYieldOnSelfCheckpoint` class below.


@qd.data_oriented
class _FastcacheYieldOnSelfCheckpoint:
    def __init__(self, n: int):
        self.x = qd.ndarray(qd.i32, shape=(n,))
        self.flag = qd.ndarray(qd.i32, shape=())

    @qd.kernel(graph=True, checkpoints=True, fastcache=True)
    def step(self):
        with qd.checkpoint(0, yield_on=self.flag):
            for i in range(self.x.shape[0]):
                self.x[i] = self.x[i] + 1


class _FastcacheCheckpointArgs(pydantic.BaseModel):
    arch: str
    offline_cache_file_path: str
    expect_loaded_from_fastcache: bool


def _fastcache_checkpoint_child(args: list[str]) -> None:
    args_obj = _FastcacheCheckpointArgs.model_validate_json(args[0])
    qd.init(
        arch=getattr(qd, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=True,
    )

    N = 8
    sim = _FastcacheYieldOnSelfCheckpoint(N)
    sim.x.from_numpy(np.zeros(N, dtype=np.int32))
    sim.flag.from_numpy(np.array(0, dtype=np.int32))
    sim.step()
    np.testing.assert_array_equal(sim.x.to_numpy(), np.ones(N, dtype=np.int32))

    primal = type(sim).step._primal
    # The schema-v3 fast-cache restore path must repopulate `checkpoint_yield_on_args` and
    # `checkpoint_yield_on_cpp_arg_ids` from the cached `CacheValue` (since AST transformation is skipped on a cache
    # hit). A regression here would surface as an empty `_forward_yield_on_table_to_ctx` call, silently breaking
    # yield/resume on fast-cached checkpoint kernels.
    labels = primal.checkpoint_yield_on_args
    cpp_ids = primal.checkpoint_yield_on_cpp_arg_ids
    assert (
        labels and len(labels) == 1 and labels[0] is not None and "flag" in labels[0]
    ), f"checkpoint_yield_on_args should round-trip with one slot containing 'flag', got {labels!r}"
    assert (
        len(cpp_ids) == 1 and cpp_ids[0] >= 0
    ), f"checkpoint_yield_on_cpp_arg_ids should round-trip with one valid id, got {cpp_ids!r}"
    assert primal.checkpoint_user_labels_by_cp_id == [
        0
    ], f"checkpoint_user_labels_by_cp_id should round-trip as [0], got {primal.checkpoint_user_labels_by_cp_id!r}"
    assert primal.src_ll_cache_observations.cache_loaded == args_obj.expect_loaded_from_fastcache, (
        f"cache_loaded={primal.src_ll_cache_observations.cache_loaded!r} but expected "
        f"{args_obj.expect_loaded_from_fastcache!r}"
    )

    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@test_utils.test()
def test_checkpoint_fastcache_restores_self_member_yield_on(tmp_path: pathlib.Path):
    """After a fast-cache restore in a fresh process, a `@qd.kernel(graph=True, checkpoints=True, fastcache=True)`
    kernel with `yield_on=self.flag` must repopulate `checkpoint_yield_on_args` / `checkpoint_yield_on_cpp_arg_ids` /
    `checkpoint_user_labels_by_cp_id` from the persisted ``CacheValue`` -- not from the AST transformer, which is
    skipped on a cache hit. Without the schema-v3 round-trip the launch path's `forward_yield_on_table_to_ctx` would
    be a no-op and yield/resume would silently break for fast-cached checkpoint kernels."""
    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch.name
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    for expect_loaded in [False, True]:
        args_obj = _FastcacheCheckpointArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path / "cache"),
            expect_loaded_from_fastcache=expect_loaded,
        )
        cmd_line = [sys.executable, __file__, _fastcache_checkpoint_child.__name__, args_obj.model_dump_json()]
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
        if proc.returncode != RET_SUCCESS:
            print(" ".join(cmd_line))
            print(proc.stdout)
            print("-" * 100)
            print(proc.stderr)
        assert TEST_RAN in proc.stdout
        assert proc.returncode == RET_SUCCESS


# Module-level IntEnum so `_resolve_intenum_member` can find it via importlib from the persisted qualname
# (`tests.python.test_checkpoint._FastcacheStage.LOAD`). The kernel below uses it as the cp_id so the fast-cache
# round-trip exercises the schema-v4 enum-identity preservation path.
class _FastcacheStage(IntEnum):
    LOAD = 10
    REDUCE = 20


@qd.kernel(graph=True, checkpoints=True, fastcache=True)
def _fastcache_intenum_kernel(
    x: qd.types.ndarray(qd.i32, ndim=1),
    flag: qd.types.ndarray(qd.i32, ndim=0),
):
    with qd.checkpoint(_FastcacheStage.LOAD, yield_on=flag):
        for i in range(x.shape[0]):
            x[i] = x[i] + 1
    with qd.checkpoint(_FastcacheStage.REDUCE, yield_on=flag):
        for i in range(x.shape[0]):
            x[i] = x[i] + 10


def _fastcache_intenum_child(args: list[str]) -> None:
    args_obj = _FastcacheCheckpointArgs.model_validate_json(args[0])
    qd.init(
        arch=getattr(qd, args_obj.arch),
        offline_cache=True,
        offline_cache_file_path=args_obj.offline_cache_file_path,
        src_ll_cache=True,
    )

    N = 4
    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    _fastcache_intenum_kernel(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 11, dtype=np.int32))

    primal = _fastcache_intenum_kernel._primal
    labels = primal.checkpoint_user_labels_by_cp_id
    # The schema-v4 round-trip must rebuild the IntEnum identity, not just the int equality. A regression here would
    # show up as `labels == [10, 20]` (plain ints) breaking the documented contract that `qd.checkpoint(Stage.X, ...)`
    # surfaces as `Stage.X` (not the raw int) on `status.checkpoint`.
    assert labels == [
        _FastcacheStage.LOAD,
        _FastcacheStage.REDUCE,
    ], f"checkpoint_user_labels_by_cp_id should round-trip with IntEnum identity, got {labels!r}"
    assert all(
        isinstance(lbl, _FastcacheStage) for lbl in labels
    ), f"every label slot must be a _FastcacheStage instance, got {[type(lbl).__name__ for lbl in labels]!r}"
    assert primal.src_ll_cache_observations.cache_loaded == args_obj.expect_loaded_from_fastcache

    print(TEST_RAN)
    sys.exit(RET_SUCCESS)


@test_utils.test()
def test_checkpoint_fastcache_preserves_intenum_label_identity(tmp_path: pathlib.Path):
    """Fast-cache restore must rebuild ``checkpoint_user_labels_by_cp_id`` with the original ``IntEnum`` members, not
    just int-equal plain ints. Schema v4 adds a parallel ``checkpoint_user_label_enum_qualnames`` column so
    ``_resolve_intenum_member`` can re-import the enum class on cache hit -- pydantic coerces ``IntEnum`` to ``int`` at
    ``CacheValue`` construction, which would otherwise silently drop enum identity and break the documented contract
    that ``qd.checkpoint(Stage.X, ...)`` surfaces as ``Stage.X`` (not the raw int) on ``status.checkpoint`` after a
    fast-cache hit."""
    assert qd.lang is not None
    arch = qd.lang.impl.current_cfg().arch.name
    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    for expect_loaded in [False, True]:
        args_obj = _FastcacheCheckpointArgs(
            arch=arch,
            offline_cache_file_path=str(tmp_path / "cache"),
            expect_loaded_from_fastcache=expect_loaded,
        )
        cmd_line = [sys.executable, __file__, _fastcache_intenum_child.__name__, args_obj.model_dump_json()]
        proc = subprocess.run(cmd_line, capture_output=True, text=True, env=env)
        if proc.returncode != RET_SUCCESS:
            print(" ".join(cmd_line))
            print(proc.stdout)
            print("-" * 100)
            print(proc.stderr)
        assert TEST_RAN in proc.stdout
        assert proc.returncode == RET_SUCCESS


# ----------------------------------------------------------------------------------------------------------------------
# CUDA-native introspection (slice 1c).
# ----------------------------------------------------------------------------------------------------------------------


@test_utils.test()
def test_checkpoint_emits_if_nodes_on_cuda_native():
    """On CUDA SM 9.0+, the GraphManager wires one IF conditional node per checkpoint (explicit AND implicit).

    Three implicit + one explicit yielder = four IF nodes on the native path. On non-CUDA / pre-SM-9.0 backends the
    kernel still runs to completion; the body-kernel correctness assertion holds either way.
    """
    N = 8

    @qd.kernel(graph=True, checkpoints=True)
    def k(x: qd.types.ndarray(qd.i32, ndim=1), flag: qd.types.ndarray(qd.i32, ndim=0)):
        for i in range(x.shape[0]):  # implicit
            x[i] = x[i] + 1
        with qd.checkpoint(0, yield_on=flag):
            for i in range(x.shape[0]):
                x[i] = x[i] + 1
        for i in range(x.shape[0]):  # implicit
            x[i] = x[i] + 1

    x = qd.ndarray(qd.i32, shape=(N,))
    flag = qd.ndarray(qd.i32, shape=())
    x.from_numpy(np.zeros(N, dtype=np.int32))
    flag.from_numpy(np.array(0, dtype=np.int32))
    k(x, flag)
    np.testing.assert_array_equal(x.to_numpy(), np.full(N, 3, dtype=np.int32))
    if _is_checkpoint_if_path_native():
        assert (
            _num_checkpoints_on_last_call() == 3
        ), f"expected 3 IF conditional nodes (2 implicit + 1 explicit), got {_num_checkpoints_on_last_call()}"


# Subprocess dispatch for fast-cache restoration tests above (mirrors the pattern in `test_graph_do_while.py`). The
# parent test invokes us via `subprocess.run([sys.executable, __file__, <child_fn_name>, <json_args>])` so the child
# runs in a fresh interpreter with a clean `qd.init` -- the only way to exercise the cross-process fast-cache load
# path that ``Kernel._try_load_fastcache`` takes after a previous run has populated the on-disk cache.
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2:])
