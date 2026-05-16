"""Tests for `qd.volatile_load`.

The primitive is mostly observable via the IR / generated machine code: the user-visible behaviour of a
correctly-emitted volatile load is identical to a regular load, _except_ in spin-wait patterns where the
compiler would otherwise hoist or merge the load.  Two layers of testing here:

1. Frontend / Python guards (no GPU needed): wrong argument shapes raise ``QuadrantsSyntaxError``.
2. IR-level checks (every supported backend): the printed IR for a kernel using ``qd.volatile_load`` must
   contain ``global load volatile`` (the marker emitted by ``ir_printer.cpp``).
3. Functional smoke test (every supported backend): kernel that volatile-loads from a field set ahead of time
   returns the expected values -- catches plumbing bugs where the load fails outright.

The "would deadlock without volatile" test is the real correctness contract but cannot be expressed
deterministically (the optimiser is free to hoist or not), so we rely on the IR-level check to certify the
volatile bit reached codegen instead.
"""

import pytest

import quadrants as qd
from quadrants.lang.exception import QuadrantsSyntaxError, QuadrantsTypeError

from tests import test_utils


@test_utils.test(arch=qd.gpu)
def test_volatile_load_basic_value():
    """Functional smoke: volatile-load reads back the value the host wrote.  Catches gross plumbing bugs in any
    backend's load_buffer / create_global_load path before the IR-level test even runs."""
    flags = qd.field(dtype=qd.i32, shape=8)

    @qd.kernel
    def read_back(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(8):
            out[i] = qd.volatile_load(flags[i])

    import numpy as np

    for i in range(8):
        flags[i] = i * 7 + 3

    out = np.zeros(8, dtype=np.int32)
    read_back(out)
    for i in range(8):
        assert out[i] == i * 7 + 3, f"out[{i}]={out[i]} expected {i * 7 + 3}"


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], print_ir=True, debug=True)
def test_volatile_load_emits_volatile_marker(capfd):
    """IR-level: ``ir_printer.cpp`` prints ``global load volatile`` only when ``GlobalLoadStmt::is_volatile``
    survives the transform pipeline.  This is the primary correctness signal that the new flag actually reaches
    codegen on every backend (LLVM and SPIR-V both run ``ir_printer`` on their input)."""
    flags = qd.field(dtype=qd.i32, shape=4)

    @qd.kernel
    def k():
        for i in range(4):
            x = qd.volatile_load(flags[i])
            flags[i] = x + 1

    for i in range(4):
        flags[i] = i

    k()
    out = capfd.readouterr().out
    assert "global load volatile" in out, (
        "expected 'global load volatile' in dumped IR but did not find it; printed IR follows:\n" + out
    )


@test_utils.test(arch=qd.gpu)
def test_volatile_load_survives_loop_invariant_caching():
    """Regression: ``cache_loop_invariant_global_vars`` must not hoist a volatile load out of an enclosing
    loop.  We can't deterministically force the optimiser to *try* to hoist, but we can exercise a kernel where
    a non-volatile load would be a clear hoist target (loop-invariant index, simple body) and assert the kernel
    still produces the correct answer.  A real hoist would either change the value (host writes the cell after
    the kernel begins) or, in the worst case, produce a stale read; here we content ourselves with verifying
    the value is what we wrote."""
    counter = qd.field(dtype=qd.i32, shape=())
    out = qd.field(dtype=qd.i32, shape=16)

    @qd.kernel
    def k():
        for i in range(16):
            # `counter[None]` is loop-invariant from the optimiser's POV; a non-volatile load could legally be
            # hoisted out of the per-iteration body.  Volatile blocks that.
            out[i] = qd.volatile_load(counter[None]) + i

    counter[None] = 100
    k()
    for i in range(16):
        assert out[i] == 100 + i


def test_volatile_load_rejects_non_lvalue():
    qd.init(arch=qd.cpu)

    with pytest.raises(QuadrantsSyntaxError, match="lvalue"):

        @qd.kernel
        def k():
            x = 1
            y = qd.volatile_load(x + 1)

        k()


def test_volatile_load_rejects_bare_field():
    qd.init(arch=qd.cpu)
    flags = qd.field(dtype=qd.i32, shape=4)

    with pytest.raises(QuadrantsSyntaxError, match="Field directly"):

        @qd.kernel
        def k():
            y = qd.volatile_load(flags)

        k()


def test_volatile_load_rejects_local_array():
    """C++-only guard: a function-scope local tensor (AllocaStmt) passes the Python lvalue + non-Field checks
    but is rejected in ``VolatileLoadExpression::flatten`` because a local alloca cannot be observed by another
    thread, so volatile semantics would be meaningless.  Exercises the path between the two Python guards and
    the GlobalLoadStmt construction."""
    qd.init(arch=qd.cpu)

    with pytest.raises(QuadrantsTypeError, match="local array"):

        @qd.kernel
        def k():
            v = qd.Vector([0, 0, 0, 0])
            y = qd.volatile_load(v)

        k()
