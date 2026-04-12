"""Tests for kernel code coverage instrumentation.

These tests verify that the AST rewriter correctly inserts coverage probes
and that the probes fire when kernel code executes on the device.
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


def test_ast_rewriter_inserts_probes():
    """Verify the AST rewriter inserts probes at each statement."""
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    src = textwrap.dedent(
        """\
        def f():
            x = 1
            y = 2
            return x + y
    """
    )
    tree = ast.parse(src)
    rewriter = _CoverageASTRewriter(field_name="_qd_cov", filepath="test.py", start_lineno=10, probe_id_start=0)
    tree = rewriter.visit(tree)

    assert rewriter.next_probe_id == 3
    assert (0, ("test.py", 11)) in rewriter.probe_map.items()
    assert (1, ("test.py", 12)) in rewriter.probe_map.items()
    assert (2, ("test.py", 13)) in rewriter.probe_map.items()


def test_ast_rewriter_branches():
    """Verify probes are inserted inside both if and else branches."""
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    src = textwrap.dedent(
        """\
        def f():
            if x > 0:
                a = 1
            else:
                b = 2
    """
    )
    tree = ast.parse(src)
    rewriter = _CoverageASTRewriter(field_name="_qd_cov", filepath="test.py", start_lineno=1, probe_id_start=0)
    tree = rewriter.visit(tree)

    lines_covered = {lineno for _, (_, lineno) in rewriter.probe_map.items()}
    assert 2 in lines_covered  # if x > 0
    assert 3 in lines_covered  # a = 1
    assert 5 in lines_covered  # b = 2


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


def test_ast_rewriter_for_loop():
    """Verify probes inside for loop body."""
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    src = textwrap.dedent(
        """\
        def f():
            for i in range(10):
                x = i
    """
    )
    tree = ast.parse(src)
    rewriter = _CoverageASTRewriter(field_name="_qd_cov", filepath="test.py", start_lineno=1, probe_id_start=0)
    tree = rewriter.visit(tree)

    lines_covered = {lineno for _, (_, lineno) in rewriter.probe_map.items()}
    assert 2 in lines_covered  # for i in range(10)
    assert 3 in lines_covered  # x = i


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_e2e():
    """End-to-end test: run a kernel and check that coverage probes fired."""
    from quadrants.lang import _kernel_coverage

    _kernel_coverage.ensure_field_allocated()

    result = qd.field(dtype=qd.i32, shape=(1,))

    @qd.kernel
    def simple_kernel():
        result[0] = 42

    simple_kernel()

    assert result[0] == 42

    cov_field = _kernel_coverage.get_field()
    assert cov_field is not None
    arr = cov_field.to_numpy()
    assert arr.sum() > 0


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

    The if/else is based on a runtime value read from a field, so the compiler
    cannot constant-fold it away. Only the taken branch's shuffle probe should fire.
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

    Runs a kernel, then resets via qd.reset()/qd.init() (which triggers the
    _hooked_clear harvest), runs another kernel, harvests again, and checks that
    _accumulated_lines contains data from both sessions.
    """
    from quadrants.lang import impl, _kernel_coverage

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

    # Don't call _harvest_field() manually — let qd.reset() trigger it via the _hooked_clear hook
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
        assert f in _kernel_coverage._accumulated_lines, (
            f"File {f} from before reset should still be in _accumulated_lines"
        )
        assert lines_before[f].issubset(_kernel_coverage._accumulated_lines[f]), (
            "Lines from before reset should be preserved"
        )

    probes_second = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_mid}
    second_files = {loc[0] for loc in probes_second.values()}
    for f in second_files:
        assert f in _kernel_coverage._accumulated_lines, (
            f"File {f} from second kernel should be in _accumulated_lines"
        )


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_autodiff_forward_covered():
    """Verify that kernel lines are covered during the forward pass of autodiff."""
    from quadrants.lang import _kernel_coverage

    _kernel_coverage.ensure_field_allocated()

    probe_count_before = _kernel_coverage._probe_counter

    x = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        loss[None] = x[None] * 2.0

    x[None] = 3.0

    with qd.ad.Tape(loss):
        compute()

    assert loss[None] == pytest.approx(6.0)
    assert x.grad[None] == pytest.approx(2.0)

    cov_field = _kernel_coverage.get_field()
    assert cov_field is not None
    arr = cov_field.to_numpy()

    probes_for_kernel = {pid: loc for pid, loc in _kernel_coverage._probe_map.items() if pid >= probe_count_before}
    fired = {pid for pid in probes_for_kernel if arr[pid] != 0}
    assert len(fired) > 0, "Forward pass inside Tape should produce coverage probes"


@test_utils.test(arch=[qd.cpu, qd.cuda])
def test_kernel_coverage_autodiff_no_extra_probes_for_grad():
    """Verify that the backward pass does not insert additional coverage probes.

    The kernel is compiled once for NONE mode (forward, with probes) and once for
    REVERSE mode (backward, without probes). The probe counter should only increase
    from the forward compilation.
    """
    from quadrants.lang import _kernel_coverage

    _kernel_coverage.ensure_field_allocated()

    x = qd.field(dtype=qd.f32, shape=(), needs_grad=True)
    loss = qd.field(dtype=qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        loss[None] = x[None] * x[None]

    x[None] = 5.0

    probe_count_before_forward = _kernel_coverage._probe_counter

    with qd.ad.Tape(loss):
        compute()

    probe_count_after_tape = _kernel_coverage._probe_counter

    forward_probes = probe_count_after_tape - probe_count_before_forward
    assert forward_probes > 0, "Forward compilation should have inserted probes"

    assert x.grad[None] == pytest.approx(10.0)

    probe_count_after_grad = _kernel_coverage._probe_counter
    assert probe_count_after_grad == probe_count_after_tape, (
        f"Backward pass should not insert additional probes, but probe counter went from "
        f"{probe_count_after_tape} to {probe_count_after_grad}"
    )


def test_env_var_max_probes():
    """Verify that QD_COVERAGE_MAX_PROBES env var is read at import time."""
    import quadrants.lang._kernel_coverage as kcov

    assert kcov._MAX_PROBES == int(os.environ.get("QD_COVERAGE_MAX_PROBES", "100000"))
