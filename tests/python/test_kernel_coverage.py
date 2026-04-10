"""Tests for kernel code coverage instrumentation.

These tests verify that the AST rewriter correctly inserts coverage probes
and that the probes fire when kernel code executes on the device.
"""

import os
import ast
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

    src = textwrap.dedent("""\
        def f():
            x = 1
            y = 2
            return x + y
    """)
    tree = ast.parse(src)
    rewriter = _CoverageASTRewriter(
        field_name="_qd_cov", filepath="test.py", start_lineno=10, probe_id_start=0
    )
    tree = rewriter.visit(tree)

    assert rewriter.next_probe_id == 3
    assert (0, ("test.py", 11)) in rewriter.probe_map.items()
    assert (1, ("test.py", 12)) in rewriter.probe_map.items()
    assert (2, ("test.py", 13)) in rewriter.probe_map.items()


def test_ast_rewriter_branches():
    """Verify probes are inserted inside both if and else branches."""
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    src = textwrap.dedent("""\
        def f():
            if x > 0:
                a = 1
            else:
                b = 2
    """)
    tree = ast.parse(src)
    rewriter = _CoverageASTRewriter(
        field_name="_qd_cov", filepath="test.py", start_lineno=1, probe_id_start=0
    )
    tree = rewriter.visit(tree)

    lines_covered = {lineno for _, (_, lineno) in rewriter.probe_map.items()}
    assert 2 in lines_covered  # if x > 0
    assert 3 in lines_covered  # a = 1
    assert 5 in lines_covered  # b = 2


def test_ast_rewriter_for_loop():
    """Verify probes inside for loop body."""
    from quadrants.lang._kernel_coverage import _CoverageASTRewriter

    src = textwrap.dedent("""\
        def f():
            for i in range(10):
                x = i
    """)
    tree = ast.parse(src)
    rewriter = _CoverageASTRewriter(
        field_name="_qd_cov", filepath="test.py", start_lineno=1, probe_id_start=0
    )
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

    probes_for_kernel = {
        pid: loc
        for pid, loc in _kernel_coverage._probe_map.items()
        if pid >= probe_count_before
    }

    taken_probes = {pid for pid, loc in probes_for_kernel.items() if arr[pid] != 0}
    not_taken_probes = {pid for pid, loc in probes_for_kernel.items() if arr[pid] == 0}

    assert len(taken_probes) > 0, "At least some probes should have fired"
    assert len(not_taken_probes) > 0, "The else branch should not have been reached"
