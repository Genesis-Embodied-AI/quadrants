"""Tests for kernel code coverage instrumentation.

These tests verify that the AST rewriter correctly inserts coverage probes and that the probes fire when kernel
code executes on the device.
"""

import ast
import os
import textwrap

import pytest

import quadrants as qd

from tests import test_utils

# These tests only run when QD_KERNEL_COVERAGE=1
pytestmark = pytest.mark.skipif(
    os.environ.get("QD_KERNEL_COVERAGE", "") != "1",
    reason="QD_KERNEL_COVERAGE=1 not set",
)


# ---------------------------------------------------------------------------
# AST rewriter unit tests
# ---------------------------------------------------------------------------

_AST_REWRITER_CASES = [
    pytest.param(
        """\
        def f():
            x = 1
            y = 2
            return x + y
        """,
        {11, 12, 13},
        10,
        id="straight_line",
    ),
    pytest.param(
        """\
        def f():
            if x > 0:
                a = 1
            else:
                b = 2
        """,
        {2, 3, 5},
        1,
        id="if_else",
    ),
    pytest.param(
        """\
        def f():
            for i in range(10):
                x = i
        """,
        {2, 3},
        1,
        id="for_loop",
    ),
    pytest.param(
        """\
        def f():
            while x > 0:
                x = x - 1
            else:
                y = 0
        """,
        {2, 3, 5},
        1,
        id="while_loop_else",
    ),
    pytest.param(
        """\
        def f():
            with ctx:
                a = 1
                b = 2
        """,
        {2, 3, 4},
        1,
        id="with_statement",
    ),
    pytest.param(
        """\
        def f():
            try:
                a = 1
            except:
                b = 2
            else:
                c = 3
            finally:
                d = 4
        """,
        {3, 5, 7, 9},
        1,
        id="try_except_finally",
    ),
]


@pytest.mark.parametrize("src,expected_lines,start_lineno", _AST_REWRITER_CASES)
def test_ast_rewriter(src, expected_lines, start_lineno):
    """Verify the AST rewriter inserts probes at the expected source lines."""
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    tree = ast.parse(textwrap.dedent(src))
    rewriter = _CoverageASTRewriter(
        field_name="_qd_cov", filepath="test.py", start_lineno=start_lineno, probe_id_start=0
    )
    rewriter.visit(tree)

    covered_lines = {lineno for _, (_, lineno) in rewriter.probe_map.items()}
    assert expected_lines.issubset(covered_lines), f"Expected lines {expected_lines} to be probed, got {covered_lines}"


def test_ast_rewriter_capacity_limit():
    """Verify that probes stop being inserted when the capacity limit is hit."""
    import warnings

    import quadrants.lang._kernel_coverage as kcov
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    src = textwrap.dedent(
        """\
        def f():
            a = 1
            b = 2
            c = 3
    """
    )
    tree = ast.parse(src)
    old_warning_state = kcov._capacity_warning_emitted
    kcov._capacity_warning_emitted = False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rewriter = _CoverageASTRewriter(
                field_name="_qd_cov", filepath="test.py", start_lineno=1, probe_id_start=kcov._MAX_PROBES - 1
            )
            rewriter.visit(tree)

        assert rewriter.next_probe_id == kcov._MAX_PROBES
        assert len(rewriter.probe_map) == 1, f"Only 1 probe should fit, got {len(rewriter.probe_map)}"
        assert len(w) == 1
        assert "exceeded" in str(w[0].message).lower()
    finally:
        kcov._capacity_warning_emitted = old_warning_state


def test_ast_rewriter_deduplicates_same_line():
    """Verify that two statements on the same source line get only one probe."""
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    src = "def f():\n    a = 1; b = 2\n"
    tree = ast.parse(src)
    rewriter = _CoverageASTRewriter(field_name="_qd_cov", filepath="test.py", start_lineno=1, probe_id_start=0)
    rewriter.visit(tree)

    abs_lines = [lineno for _, (_, lineno) in rewriter.probe_map.items()]
    assert abs_lines.count(2) == 1, f"Line 2 should have exactly one probe, got {abs_lines.count(2)}"


def test_env_var_max_probes():
    """Verify that QD_COVERAGE_MAX_PROBES env var is read at import time."""
    import quadrants.lang._kernel_coverage as kcov

    assert kcov._MAX_PROBES == int(os.environ.get("QD_COVERAGE_MAX_PROBES", "100000"))


def test_harvest_field_exception_path():
    """Verify that _harvest_field handles to_numpy() failure gracefully."""
    from unittest.mock import MagicMock

    import quadrants.lang._kernel_coverage as kcov

    old_field = kcov._cov_field
    old_prog = kcov._cov_field_prog
    old_map = kcov._probe_map.copy()
    try:
        mock_field = MagicMock()
        mock_field.to_numpy.side_effect = RuntimeError("runtime destroyed")
        kcov._cov_field = mock_field
        kcov._cov_field_prog = object()
        kcov._probe_map[999999] = ("fake.py", 1)

        # Should not raise - the exception is caught and logged
        kcov._harvest_field()

        assert kcov._cov_field is None, "Field should be cleared after failure"
        assert kcov._cov_field_prog is None, "Field prog should be cleared after failure"
    finally:
        kcov._cov_field = old_field
        kcov._cov_field_prog = old_prog
        kcov._probe_map = old_map


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_branches_e2e():
    """Verify that only the taken branch has its probe fired."""
    from quadrants.lang import _kernel_coverage

    _kernel_coverage.ensure_field_allocated()

    probe_count_before = _kernel_coverage._probe_counter
    out = qd.field(dtype=qd.i32, shape=(1,))

    @qd.kernel
    def branching_kernel():
        x = 10
        if x > 5:
            out[0] = 1
        else:
            out[0] = 2

    branching_kernel()

    assert out[0] == 1

    cov_field = _kernel_coverage.get_field()
    arr = cov_field.to_numpy()

    probes_for_kernel = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_before}

    taken_probes = {pid for pid, loc in probes_for_kernel.items() if arr[pid] != 0}
    not_taken_probes = {pid for pid, loc in probes_for_kernel.items() if arr[pid] == 0}

    assert len(taken_probes) > 0, "At least some probes should have fired"
    assert len(not_taken_probes) > 0, "The else branch should not have been reached"


@test_utils.test(arch=qd.gpu)
def test_kernel_coverage_simt_e2e():
    """Verify coverage probes track branches with block.sync() and subgroup shuffle.

    The if/else is based on a runtime value read from a field, so the compiler cannot constant-fold it away.
    Only the taken branch's shuffle probe should fire.
    """
    from quadrants.lang import _kernel_coverage
    from quadrants.lang.simt import subgroup

    _kernel_coverage.ensure_field_allocated()

    N = 64
    probe_count_before = _kernel_coverage._probe_counter
    flag = qd.field(dtype=qd.i32, shape=(1,))
    a = qd.field(dtype=qd.i32, shape=(N,))
    out = qd.field(dtype=qd.i32, shape=(N,))

    flag[0] = 1  # runtime value: take the if-branch

    @qd.kernel
    def simt_kernel():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = i + 1
            qd.simt.block.sync()
            if flag[0] > 0:
                val = subgroup.shuffle(a[i], qd.u32(0))
                out[i] = val
            else:
                val = subgroup.shuffle(a[i], qd.u32(1))
                out[i] = val + 100

    simt_kernel()

    for i in range(4):
        assert out[i] == 1, f"Expected 1 at index {i}, got {out[i]}"

    cov_field = _kernel_coverage.get_field()
    arr = cov_field.to_numpy()

    probes_for_kernel = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_before}

    fired = {pid for pid in probes_for_kernel if arr[pid] != 0}
    not_fired = {pid for pid in probes_for_kernel if arr[pid] == 0}
    assert len(fired) >= 4, f"Expected at least 4 probes to fire, got {len(fired)}"
    assert len(not_fired) >= 2, "The else branch should not have been reached"


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_survives_reinit():
    """Verify that coverage data accumulated before qd.init() reset is preserved.

    Runs a kernel, then resets via qd.reset()/qd.init() (which triggers the _hooked_clear harvest), runs another
    kernel, harvests again, and checks that _accumulated_lines contains data from both sessions.
    """
    from quadrants.lang import _kernel_coverage, impl

    current_arch = impl.get_runtime()._arch
    _kernel_coverage.ensure_field_allocated()

    probe_count_before = _kernel_coverage._probe_counter
    out1 = qd.field(dtype=qd.i32, shape=(1,))

    @qd.kernel
    def kernel_before_reset():
        out1[0] = 1

    kernel_before_reset()

    cov_field = _kernel_coverage.get_field()
    assert cov_field is not None
    arr = cov_field.to_numpy()
    probes_first = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_before}
    fired_first = {pid for pid in probes_first if arr[pid] != 0}
    assert len(fired_first) > 0, "Probes from first kernel should have fired"

    # Don't call _harvest_field() manually - let qd.reset() trigger it via the _hooked_clear hook
    qd.reset()

    # Verify the hook harvested data from the first session
    files_before = set(_kernel_coverage._accumulated_lines.keys())
    assert len(files_before) > 0, "Hook should have harvested data during reset"
    lines_before = {}
    for f, lines in _kernel_coverage._accumulated_lines.items():
        lines_before[f] = set(lines)

    qd.init(arch=current_arch)

    _kernel_coverage.ensure_field_allocated()

    probe_count_mid = _kernel_coverage._probe_counter
    out2 = qd.field(dtype=qd.i32, shape=(1,))

    @qd.kernel
    def kernel_after_reset():
        out2[0] = 2

    kernel_after_reset()

    _kernel_coverage._harvest_field()

    for f in files_before:
        assert (
            f in _kernel_coverage._accumulated_lines
        ), f"File {f} from before reset should still be in _accumulated_lines"
        assert lines_before[f].issubset(
            _kernel_coverage._accumulated_lines[f]
        ), "Lines from before reset should be preserved"

    probes_second = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_mid}
    second_files = {loc[0] for loc in probes_second.values()}
    for f in second_files:
        assert f in _kernel_coverage._accumulated_lines, f"File {f} from second kernel should be in _accumulated_lines"


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_autodiff():
    """Verify that autodiff forward pass produces probes but backward does not.

    The forward compilation (AutodiffMode.NONE) should insert probes that fire. The backward compilation
    (AutodiffMode.REVERSE) should not add any probes.
    """
    from quadrants.lang import _kernel_coverage

    _kernel_coverage.ensure_field_allocated()

    x = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        loss[None] = x[None] * x[None]

    x[None] = 5.0

    probe_count_before = _kernel_coverage._probe_counter

    with qd.ad.Tape(loss):
        compute()

    probe_count_after_tape = _kernel_coverage._probe_counter
    forward_probes = probe_count_after_tape - probe_count_before
    assert forward_probes > 0, "Forward compilation should have inserted probes"

    # Verify forward probes actually fired
    cov_field = _kernel_coverage.get_field()
    assert cov_field is not None
    arr = cov_field.to_numpy()
    probes = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_before}
    fired = {pid for pid in probes if arr[pid] != 0}
    assert len(fired) > 0, "Forward pass inside Tape should produce fired coverage probes"

    # Verify backward pass computes correct gradients
    assert loss[None] == pytest.approx(25.0)
    assert x.grad[None] == pytest.approx(10.0)


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_qd_func():
    """Verify that probes fire inside a @qd.func called from a kernel."""
    from quadrants.lang import _kernel_coverage

    _kernel_coverage.ensure_field_allocated()

    probe_count_before = _kernel_coverage._probe_counter
    out = qd.field(dtype=qd.i32, shape=(1,))

    @qd.func
    def helper():
        out[0] = 99

    @qd.kernel
    def caller():
        helper()

    caller()

    assert out[0] == 99

    cov_field = _kernel_coverage.get_field()
    assert cov_field is not None
    arr = cov_field.to_numpy()

    probes = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_before}
    fired = {pid for pid in probes if arr[pid] != 0}
    # The kernel body has one statement (helper()), and the func body has one (out[0] = 99).
    # Both should produce probes that fire.
    assert (
        len(fired) >= 2
    ), f"Expected probes from both kernel and func to fire, got {len(fired)} fired out of {len(probes)}"


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_multiple_kernels_same_session():
    """Verify that probes from two different kernels both fire in the same session."""
    from quadrants.lang import _kernel_coverage

    _kernel_coverage.ensure_field_allocated()

    probe_count_before = _kernel_coverage._probe_counter
    a = qd.field(dtype=qd.i32, shape=(1,))
    b = qd.field(dtype=qd.i32, shape=(1,))

    @qd.kernel
    def kernel_a():
        a[0] = 10

    @qd.kernel
    def kernel_b():
        b[0] = 20

    kernel_a()
    probe_count_after_a = _kernel_coverage._probe_counter
    kernel_b()

    assert a[0] == 10
    assert b[0] == 20

    cov_field = _kernel_coverage.get_field()
    arr = cov_field.to_numpy()

    probes_a = {
        pid: loc for pid, loc in _kernel_coverage._probe_map.items() if probe_count_before <= pid < probe_count_after_a
    }
    probes_b = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_after_a}

    fired_a = {pid for pid in probes_a if arr[pid] != 0}
    fired_b = {pid for pid in probes_b if arr[pid] != 0}

    assert len(fired_a) > 0, "Probes from kernel_a should have fired"
    assert len(fired_b) > 0, "Probes from kernel_b should have fired"


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_qd_prefix_exemption_pure_kernel():
    """Verify that _qd_-prefixed globals don't violate pure kernel checks.

    With kernel coverage enabled, _qd_cov is injected as a global. This test verifies that a pure (fastcache)
    kernel still compiles without error. The kernel uses ndarray arguments (not global fields) because pure
    kernels prohibit non-_qd_ globals.
    """
    a = qd.ndarray(qd.i32, (1,))

    @qd.kernel(fastcache=True)
    def pure_kernel(arr: qd.types.NDArray) -> None:
        arr[0] = 42

    pure_kernel(a)
    assert a[0] == 42
