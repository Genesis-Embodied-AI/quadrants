"""Numerical regressions retained from the legacy adstack test suite.

Historical test names identify their original regressions; all backwards now use measured replay.
Legacy sizing overrides were removed, while numerical oracles and shape parameters are preserved.
"""

import math
import os
import pathlib
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

_KNOWN_LINEAR_UNARY_OPS = {"neg"}


@pytest.mark.needs_torch
@pytest.mark.parametrize(
    "op_name",
    # Pins MakeDual (forward mode) for every nonlinear unary op whose MakeAdjoint (reverse-mode) recompute audit was the
    # focus of PRs 1-4 of this chain: {tan, tanh, exp, log, sqrt, rsqrt}. Forward mode is argued safe-by-construction
    # because it runs in primal order, so the primal value is the current-iteration value by definition - there is no
    # stale-read hazard analogous to the reverse-mode one the audit targeted. This test pins that forward-order safety
    # argument per audited op so a future MakeDual refactor cannot regress it silently. The sign/absolute-value siblings
    # (abs, sin, cos, asin, acos) are covered by `test_adstack_unary_loop_carried` in the reverse-mode direction already
    # and were not part of the audit.
    ["tan", "tanh", "exp", "log", "sqrt", "rsqrt"],
)
@test_utils.test()
def test_unary_forward_mode_derivative(op_name):
    import torch

    N = 4
    x = qd.field(qd.f32, shape=N)
    loss = qd.field(qd.f32, shape=())
    qd.root.lazy_dual()

    qd_op = getattr(qd, op_name)
    torch_op = getattr(torch, op_name)

    @qd.kernel
    def kern():
        for i in x:
            loss[None] += qd_op(x[i])

    for i in range(N):
        x[i] = 0.1 * (i + 1)

    seed = [1.0] * N
    with qd.ad.FwdMode(loss=loss, param=x, seed=seed):
        kern()

    x_t = torch.tensor([0.1 * (i + 1) for i in range(N)], dtype=torch.float32, requires_grad=True)
    l_t = torch_op(x_t).sum()
    l_t.backward()
    expected = float(x_t.grad.sum().item())

    assert loss.dual[None] == pytest.approx(expected, rel=1e-5)


def test_unary_collections_audit():
    # Prevents drift between the Python unary op registry and the C++ `unary_collections` set declared in
    # `quadrants/transforms/auto_diff/auto_diff_common.h`. Every unary op whose `MakeAdjoint` branch (defined in
    # `quadrants/transforms/auto_diff/make_adjoint.cpp`) accumulates onto `stmt->operand` must be either in
    # `unary_collections` (nonlinear: needs per-iteration operand spilling on the adstack inside dynamic loops) or
    # in the local `_KNOWN_LINEAR_UNARY_OPS` allow-list (reverse formula uses a compile-time constant coefficient,
    # so the single-slot spill path is correct for it). Forgetting to classify a new diffable unary op falls back
    # to the single-slot spill and produces silently wrong gradients in dynamic loops.
    #
    # Four invariants are checked, all of them symmetric:
    #   (1) Every diffable-math op detected in MakeAdjoint is in `cpp_nonlinear` OR in `_KNOWN_LINEAR_UNARY_OPS`.
    #   (2) Every op in `cpp_nonlinear` has a matching diffable-math branch in MakeAdjoint.
    #   (3) Every op in `_KNOWN_LINEAR_UNARY_OPS` has a matching diffable-math branch in MakeAdjoint.
    #   (4) `cpp_nonlinear` and `_KNOWN_LINEAR_UNARY_OPS` are disjoint (an op cannot be both nonlinear and linear).
    auto_diff_dir = pathlib.Path(__file__).resolve().parents[2] / "quadrants" / "transforms" / "auto_diff"
    common_src = (auto_diff_dir / "auto_diff_common.h").read_text()
    make_adjoint_src = (auto_diff_dir / "make_adjoint.cpp").read_text()

    cc_match = re.search(r"unary_collections\s*\{([^}]+)\}", common_src)
    assert cc_match is not None, "unary_collections not located in auto_diff_common.h"
    cpp_nonlinear = set(re.findall(r"UnaryOpType::(\w+)", cc_match.group(1)))

    make_adjoint_start = make_adjoint_src.find("class MakeAdjoint")
    assert make_adjoint_start != -1, "class MakeAdjoint not located in make_adjoint.cpp"
    adj_start = make_adjoint_src.find("void visit(UnaryOpStmt *stmt) override", make_adjoint_start)
    assert adj_start != -1, "MakeAdjoint::visit(UnaryOpStmt*) not located in make_adjoint.cpp"
    adj_end = make_adjoint_src.find("void visit(", adj_start + 10)
    assert adj_end != -1, "next visitor method after MakeAdjoint::visit(UnaryOpStmt*) not found"
    adj_block = make_adjoint_src[adj_start:adj_end]
    # Split the visitor's if/else-if chain into per-op segments, then classify each segment as "diffable math"
    # iff its body accumulates onto `stmt->operand`. Two accumulate entry points are recognised: the raw
    # `accumulate(stmt->operand, ...)` call (used by `neg` and `cast_value`) and the `acc(...)` lambda (used by
    # every nonlinear branch; `acc` wraps `accumulate_unary_operand_checked` which validates that the adjoint
    # formula does not read the forward `stmt` directly, and then delegates to `accumulate(stmt->operand, ...)`).
    # `cast_value` is excluded: its conditional accumulate is gated on real-typed elements and is orthogonal to
    # the `unary_collections` trade-off.
    #
    # Use `re.findall` rather than `re.search` so a future branch that ORs multiple `UnaryOpType::X` conditions
    # (e.g. `op_type == floor || op_type == ceil`) still classifies every op in the branch — capturing just the
    # first match would silently skip later ops in an OR chain.
    branches = re.split(r"\belse if\b", adj_block)
    diffable_math = set()
    for seg in branches:
        ops_in_seg = re.findall(r"UnaryOpType::(\w+)", seg)
        if not ops_in_seg:
            continue
        if "accumulate(stmt->operand" in seg or re.search(r"\bacc\(", seg):
            diffable_math.update(ops_in_seg)
    diffable_math.discard("cast_value")

    overlap = cpp_nonlinear & _KNOWN_LINEAR_UNARY_OPS
    assert not overlap, (
        f"Ops appear in BOTH `unary_collections` and `_KNOWN_LINEAR_UNARY_OPS`: {sorted(overlap)}. "
        f"An op must be classified as exactly one of (nonlinear, linear); otherwise the per-op audits below "
        f"silently cancel out."
    )

    missing = diffable_math - cpp_nonlinear - _KNOWN_LINEAR_UNARY_OPS
    assert not missing, (
        f"Diffable unary ops not classified as nonlinear or linear: {sorted(missing)}. "
        f"Add each one to `unary_collections` in quadrants/transforms/auto_diff/auto_diff_common.h (if "
        f"nonlinear) or to `_KNOWN_LINEAR_UNARY_OPS` in this file (if linear)."
    )
    stray_nonlinear = cpp_nonlinear - diffable_math
    assert not stray_nonlinear, (
        f"`unary_collections` lists ops with no matching accumulate branch (`accumulate(stmt->operand, ...)` or "
        f"`acc(...)`) in MakeAdjoint: {sorted(stray_nonlinear)}. Either remove them from `unary_collections` or "
        f"restore the MakeAdjoint branch."
    )
    stray_linear = _KNOWN_LINEAR_UNARY_OPS - diffable_math
    assert not stray_linear, (
        f"`_KNOWN_LINEAR_UNARY_OPS` lists ops with no matching accumulate branch in MakeAdjoint: "
        f"{sorted(stray_linear)}. Either remove them from `_KNOWN_LINEAR_UNARY_OPS` or restore the MakeAdjoint "
        f"branch; an entry here without a visitor means the op silently has no gradient implementation."
    )


@pytest.mark.xfail(
    reason="Reverse-mode NaN/Inf poisoning semantics is TBD. PyTorch propagates forward NaN into the backward graph "
    "(e.g. `log(-0.3).backward()` gives NaN), but Quadrants evaluates the reverse formula directly and returns a "
    "finite gradient. Either behaviour is defensible; picking a consistent rule is a separate design decision.",
    strict=True,
)
@pytest.mark.parametrize(
    "op_name,x_val",
    [
        # `(log, -0.3)` puts the operand outside the op's domain so the forward result is NaN. PyTorch backward
        # poisons the gradient with NaN; Quadrants currently evaluates `1 / operand = -3.333` verbatim and returns
        # that finite number, which this test documents as the expected-failure mode. The sqrt/asin/acos siblings
        # cannot be parametrized here under `xfail(strict=True)` because their reverse formulas divide by
        # `sqrt(<=0)` which is itself NaN, so Quadrants actually does return NaN there (test would XPASS and
        # `strict=True` treats an xpass as a failure).
        ("log", -0.3),
    ],
)
@test_utils.test()
def test_adstack_nan_propagation(op_name, x_val):
    # Pins the open question about reverse-mode NaN propagation. For each out-of-domain input the forward output
    # is NaN (both Quadrants and PyTorch agree). In reverse, PyTorch's backward pass propagates the NaN into the
    # gradient (`x.grad = NaN`) because a single NaN anywhere in the forward graph poisons every dependent grad.
    # Quadrants instead runs the analytical formula and returns a finite number. Which behaviour is "correct" is a
    # design call; the test is marked `xfail(strict=True)` so a deliberate change of semantics in either direction
    # forces a reviewer decision.
    qd_op = getattr(qd, op_name)

    x = qd.field(qd.f32, shape=(), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        y[None] = qd_op(x[None])

    x[None] = x_val
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[None] = 0.0
    compute.grad()

    assert math.isnan(x.grad[None])


@pytest.mark.xfail(
    reason=(
        "Reverse-mode NaN/Inf poisoning semantics is TBD (f64 variant). Same divergence as the f32 case: PyTorch "
        "propagates NaN in the backward graph; Quadrants runs `1 / operand` verbatim and returns a finite number."
    ),
    strict=True,
)
@pytest.mark.parametrize("op_name,x_val", [("log", -0.3)])
@test_utils.test(require=qd.extension.data64, default_fp=qd.f64)
def test_adstack_nan_propagation_f64(op_name, x_val):
    # f64 counterpart of `test_adstack_nan_propagation`. The NaN-poisoning semantics question is dtype-independent
    # (PyTorch's backward graph propagates NaN regardless of dtype; Quadrants' reverse formula `1 / operand`
    # produces a finite number in both f32 and f64). Parametrizing over dtype ensures a future decision to change
    # semantics cannot accidentally fix one dtype and miss the other.
    qd_op = getattr(qd, op_name)

    x = qd.field(qd.f64, shape=(), needs_grad=True)
    y = qd.field(qd.f64, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        y[None] = qd_op(x[None])

    x[None] = x_val
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[None] = 0.0
    compute.grad()

    assert math.isnan(x.grad[None])


_REVERSE_GLOBAL_CROSS_ITER_PATTERNS = [
    # `(has_outer_for, in_func, load_kind)` toggles for `_run_reverse_global_cross_iter`. `load_kind` selects which
    # `out[...]` index expression the inner range loads from: `cross_iter` is the canonical `out[i-1]` shape with a
    # genuine cross-iteration RAW; `constant_outside_iter_range` reads from a slot of `out` that the loop never
    # writes to, so the load and store live on disjoint iteration slices and the diagnostic must not fire.
    pytest.param(False, False, "cross_iter", id="kernel_top_inline_cross_iter"),
    pytest.param(False, True, "cross_iter", id="kernel_top_in_func_cross_iter"),
    pytest.param(True, False, "cross_iter", id="outer_for_inline_cross_iter"),
    pytest.param(True, True, "cross_iter", id="outer_for_in_func_cross_iter"),
    pytest.param(False, False, "constant_outside_iter_range", id="kernel_top_inline_constant_outside"),
    pytest.param(False, True, "constant_outside_iter_range", id="kernel_top_in_func_constant_outside"),
    pytest.param(True, False, "constant_outside_iter_range", id="outer_for_inline_constant_outside"),
    pytest.param(True, True, "constant_outside_iter_range", id="outer_for_in_func_constant_outside"),
]


def _run_reverse_global_cross_iter(has_outer_for, in_func, load_kind, expect_raise):
    # Single factory covering the reverse-mode shapes that read and write the same `needs_grad` global field
    # `out` from inside an inner non-static range, parametrised on three orthogonal axes:
    #   - `has_outer_for`: when True the inner range is nested under a `range(1)`, so it is AST-nested at
    #     `loop_depth == 1`; when False the inner range becomes the offload-level loop after inlining. The
    #     toggle is materialised via the `range(1) if qd.static(...) else qd.static(range(1))` inline-if pattern
    #     (already used by `_run_unary_loop_carried` in this file) so a single kernel template covers both.
    #   - `in_func`: when True the inner range body lives inside a `@qd.func` invoked from the kernel; when
    #     False the same body is written directly inside the kernel.
    #   - `load_kind`: `cross_iter` puts the canonical `out[i-1]` cross-iteration RAW shape in the body;
    #     `constant_outside_iter_range` replaces the load with `out[n]` (a slot strictly outside the iteration
    #     range `[0, n)`), so the load and store live on disjoint iteration slices and there is no real
    #     cross-iter dependency. Pins the offload-level guard's `references_loop_index` gate.
    n = 4
    out_shape = 2 * n if load_kind == "constant_outside_iter_range" else n
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=out_shape, needs_grad=True)

    @qd.func
    def func_body_cross_iter():
        for i in range(n):
            if i == 0:
                out[i] = x[None]
            else:
                out[i] = out[i - 1] + x[None]

    @qd.func
    def func_body_constant_outside():
        for i in range(n):
            out[i] = out[n] + x[None]

    @qd.kernel
    def compute():
        for _ in range(1) if qd.static(has_outer_for) else qd.static(range(1)):
            if qd.static(in_func):
                if qd.static(load_kind == "cross_iter"):
                    func_body_cross_iter()
                else:
                    func_body_constant_outside()
            else:
                if qd.static(load_kind == "cross_iter"):
                    for i in range(n):
                        if i == 0:
                            out[i] = x[None]
                        else:
                            out[i] = out[i - 1] + x[None]
                else:
                    for i in range(n):
                        out[i] = out[n] + x[None]

    x[None] = 1.0
    if load_kind == "constant_outside_iter_range":
        out[n] = 0.5
    compute()
    for i in range(out_shape):
        out.grad[i] = 0.0
    if load_kind == "cross_iter":
        out.grad[n - 1] = 1.0
        expected_x_grad = float(n)
    else:
        for i in range(n):
            out.grad[i] = 1.0
        expected_x_grad = float(n)
    x.grad[None] = 0.0
    if expect_raise:
        with pytest.raises(
            (qd.QuadrantsCompilationError, RuntimeError),
            match=r"non static range|Cross-iteration read-after-write",
        ):
            compute.grad()
    else:
        compute.grad()
        assert float(x.grad[None]) == pytest.approx(expected_x_grad, rel=1e-6)


@pytest.mark.parametrize("has_outer_for, in_func, load_kind", _REVERSE_GLOBAL_CROSS_ITER_PATTERNS)
@test_utils.test(require=qd.extension.adstack)
def test_reverse_global_cross_iter_with_adstack(has_outer_for, in_func, load_kind):
    # Reverse-mode AD on a kernel that reads and writes the same `needs_grad` global field from inside a non-static
    # range must either produce the analytical gradient or raise a clear diagnostic. With adstack on, only the
    # kernel-top shapes with a genuine cross-iteration RAW raise; the outer-for shapes and the
    # constant-outside-iter-range shapes produce the analytical gradient.
    #
    # Internal details: `AdStackAllocaJudger` only promotes cross-iter dependencies on local `AllocaStmt`s, never
    # through global field stores; cross-iter through a global needs_grad field at the offload level is therefore
    # not handled in either adstack setting and must raise. The offload-level guard in `auto_diff.cpp` pairs same-
    # SNode `(GlobalStore, GlobalLoad)` and raises on structurally-different index `Stmt*`s, but only when both
    # indices transitively reference a `LoopIndexStmt` - which suppresses the false positive on the
    # constant-outside-iter-range shape where the load reads a slot the loop never writes to. The outer-for shapes
    # route the cross-iter chain through an AST-nested range whose backward the AD pass handles correctly via the
    # IB analysis when adstack is on.
    expect_raise = (not has_outer_for) and load_kind == "cross_iter"
    _run_reverse_global_cross_iter(has_outer_for, in_func, load_kind, expect_raise)


_REVERSE_SIBLING_COMPONENT_PATTERNS = [
    # `(has_outer_for, in_func)` toggles for `_run_reverse_sibling_component_correct_gradient`. Same axes as the
    # cross-iter matrix, minus the `load_kind` dimension (the body shape here is fixed: same loop index on
    # axis 0 of `out`, differing constants on axis 1).
    pytest.param(False, False, id="kernel_top_inline"),
    pytest.param(False, True, id="kernel_top_in_func"),
    pytest.param(True, False, id="outer_for_inline"),
    pytest.param(True, True, id="outer_for_in_func"),
]


def _run_reverse_sibling_component_correct_gradient(has_outer_for, in_func):
    # Drives a kernel that writes `out[i, 0]` and reads `out[i, 1]` on a 2-axis `needs_grad` field. The store
    # and load both reference `LoopIndexStmt(i)` on axis 0 (same `Stmt*`) and have differing `ConstStmt`s on
    # axis 1. The offload-level guard's per-axis rule treats the differing constant axis as iter-independent
    # sibling access, so the kernel must compile and produce the analytical gradient.
    n = 4
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=(n, 2), needs_grad=True)

    @qd.func
    def func_body():
        for i in range(n):
            out[i, 1] = x[None]
            out[i, 0] = out[i, 1] * 2.0

    @qd.kernel
    def compute():
        for _ in range(1) if qd.static(has_outer_for) else qd.static(range(1)):
            if qd.static(in_func):
                func_body()
            else:
                for i in range(n):
                    out[i, 1] = x[None]
                    out[i, 0] = out[i, 1] * 2.0

    x[None] = 1.0
    compute()
    for i in range(n):
        out.grad[i, 0] = 1.0
        out.grad[i, 1] = 0.0
    x.grad[None] = 0.0
    compute.grad()
    # `out[i, 0] = (out[i, 1] = x) * 2.0 = 2x` per iter; seed `out.grad[i, 0] = 1` for every i.
    # `d(out[i, 0]) / d(x) = 2` per iter -> `x.grad = 2 * n`.
    assert float(x.grad[None]) == pytest.approx(2.0 * n, rel=1e-6)


@pytest.mark.parametrize("has_outer_for, in_func", _REVERSE_SIBLING_COMPONENT_PATTERNS)
@test_utils.test(require=qd.extension.adstack)
def test_reverse_sibling_component_correct_gradient_with_adstack(has_outer_for, in_func):
    # Sibling-component access on a multi-axis `needs_grad` field must compile and produce the analytical
    # gradient. The store and load share the same `LoopIndexStmt` on axis 0 and differ only on a constant
    # axis, which is sibling access in the same iteration, not cross-iter RAW.
    #
    # Internal details: the offload-level guard's per-axis `indices_have_no_cross_iter_dependency` rule
    # treats axes where neither index references `LoopIndexStmt` as iter-independent and ignores their
    # difference, so `out[i, 0]` vs `out[i, 1]` is not flagged.
    _run_reverse_sibling_component_correct_gradient(has_outer_for, in_func)


def _run_reverse_kernel_top_indep_iters_correct_gradient():
    # Independent-iteration kernel-top range-for (no cross-iteration RAW). Both adstack settings must
    # produce the analytical gradient `2 * n` for the per-iteration store `out[i] = x[None] * 2`.
    n = 4
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=n, needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            out[i] = x[None] * 2.0

    x[None] = 1.0
    compute()
    for i in range(n):
        out.grad[i] = 1.0
    x.grad[None] = 0.0
    compute.grad()
    assert float(x.grad[None]) == pytest.approx(2.0 * n, rel=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_reverse_kernel_top_indep_iters_with_adstack():
    # Kernel-top non-static range whose body has no cross-iter RAW on a `needs_grad` field must
    # produce the analytical gradient with adstack on.
    #
    # Internal details: the offload-level guard pairs only same-SNode `(store, load)`, so a body
    # that only stores `out[i] = x[None] * 2` (no load on `out`) compiles cleanly. Adstack itself has
    # nothing to do here either - each iteration is independent.
    _run_reverse_kernel_top_indep_iters_correct_gradient()


def _run_reverse_kernel_top_in_place_accum_correct_gradient():
    # In-place accumulation `out[i] = out[i] + x[None]` at the kernel-top range-for. Sanity check that
    # the offload-level guard's pointer-equality on index `Stmt*`s allows this shape - store and
    # load on `out` share the same `LoopIndexStmt` so `indices_pointer_match` returns true and the
    # guard does not fire.
    n = 4
    x = qd.field(qd.f32, shape=(), needs_grad=True)
    out = qd.field(qd.f32, shape=n, needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            out[i] = out[i] + x[None]

    x[None] = 1.0
    for i in range(n):
        out[i] = 0.0
    compute()
    for i in range(n):
        out.grad[i] = 1.0
    x.grad[None] = 0.0
    compute.grad()
    # `out[i] = out[i] + x` reduces to `out[i] += x`; with `out[i]` initialised to 0, the per-iteration
    # adjoint accumulates 1 into `x.grad` for each of n elements when `out.grad[i] = 1`.
    assert float(x.grad[None]) == pytest.approx(float(n), rel=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_reverse_kernel_top_in_place_accum_with_adstack():
    # In-place accumulation `out[i] = out[i] + x[None]` at the offload-level loop must compile and
    # produce the analytical gradient: the offload-level guard intentionally exempts this shape.
    #
    # Internal details: the store dest `out[i]` and the load src `out[i]` reference the same
    # `LoopIndexStmt`, so `OffloadLevelGlobalCrossIterRAWChecker::indices_pointer_match` returns true
    # and the guard does not fire. The per-iteration adjoint already handles in-place accumulation
    # correctly: `x.grad` accumulates 1 per element.
    _run_reverse_kernel_top_in_place_accum_correct_gradient()


@test_utils.test(require=qd.extension.adstack)
def test_adstack_large_capacity_heap_backed():
    # Exercise heap-backed replay with a nonlinear recurrence and compare its analytical gradient.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(128):
                y[None] += qd.sin(v)
                v = v + 1.0

    x[0] = 0.1
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    expected = sum(math.cos(0.1 + k) for k in range(128))
    assert x.grad[0] == pytest.approx(expected, rel=1e-4)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_rejects_unsupported_type():
    # Per-backend support matrix for loop-carried i8 in reverse-mode AD:
    #
    #   - SPIR-V (Metal / Vulkan) hard-rejects i8 loop-carried variables at codegen time: the adstack heap-backing
    #     only packs f32 and i32 (with u1 reinterpreted as i32); wider/other types have no home there. A
    #     Function-scope fallback is not offered because it is unusable for real workloads on Metal/MoltenVK
    #     (per-thread private-memory budget is exceeded) - silently falling back would paper over a
    #     correctness/perf cliff. The guard surfaces as a `RuntimeError` whose message names the supported type
    #     set. The match below pins the message so an accidental Function-scope fallback or a rename of the
    #     error string is caught.
    #   - LLVM (CPU / CUDA / AMDGPU) supports loop-carried i8 directly: the adstack heap there is byte-indexed
    #     and `AdStackAllocaStmt::size_in_bytes()` covers i8 at one byte per slot without any special case. The
    #     backward pass runs to completion and the forward output is what the test asserts.
    #
    # Skip on Vulkan drivers without shaderInt8: those reject i8 at the SPIR-V type gate (`spirv_ir_builder`
    # `QD_ERROR("Type i8 not supported")`) well before the adstack codegen guard fires, which would fail this
    # test with the wrong error message. Metal always reports spirv_has_int8=1 so it never skips here. The LLVM
    # backends do not consult `spirv_has_int8`, so the skip only applies on SPIR-V arches.
    arch = qd.lang.impl.get_runtime().prog.config().arch
    is_spirv = arch in (qd.metal, qd.vulkan)
    if is_spirv:
        caps = qd.lang.impl.get_runtime().prog.get_device_caps()
        if not caps.get(qd._lib.core.DeviceCapability.spirv_has_int8):
            pytest.skip("device lacks shaderInt8 - i8 is rejected at the SPIR-V type gate, not the adstack guard")

    # Internal detail: i8 is the probe type rather than f64 because Metal / MoltenVK rejects f64 at the
    # field-writer stage before the adstack codegen path is reached, whereas i8 is supported on both SPIR-V
    # backends (`spirv_has_int8`) and reaches the adstack guard. Any other type outside the SPIR-V supported set
    # (f16, i16, i64, u8, u16, u32, u64, f64) would do; i8 is the widest-supported of the SPIR-V-rejected
    # options and is also a plain supported type on every LLVM backend.
    x = qd.field(qd.i8)
    y = qd.field(qd.f32, shape=(), needs_grad=True)
    qd.root.dense(qd.i, 1).place(x)
    # needs_grad on the i8 field is meaningless (gradients do not flow through integer casts); the adstack
    # appears because the reverse pass replays the loop-carried i8 `v` value across the dynamic loop.

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(4):
                v = v + qd.cast(1, qd.i8)
            y[None] += qd.cast(v, qd.f32)

    x[0] = 0
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    if is_spirv:
        with pytest.raises(RuntimeError, match=r"f32, i32, and u1"):
            compute.grad()
    else:
        compute.grad()
        # Forward: v starts at 0, increments by 1 four times, writes 4 as f32 into y. No overflow expected.
        assert y[None] == pytest.approx(4.0, rel=1e-6)


def _overflowing_compute(n_elements=1, n_iter=64):
    # Build a nonlinear recurrence, initialize its inputs, and seed the output gradient.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, n_elements).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter):
                y[None] += qd.sin(v)
                v = v + 1.0

    for i in range(n_elements):
        x[i] = 0.1 + 0.01 * i
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n_elements):
        x.grad[i] = 0.0
    return compute, x, y


@test_utils.test(require=qd.extension.adstack)
def test_adstack_large_capacity_resolves_overflow():
    # A 64-step nonlinear recurrence must receive enough measured storage to finish correctly.
    compute, x, _ = _overflowing_compute()
    compute.grad()
    qd.sync()

    # y += sin(v) iterated with v = x[0] + k for k = 0..63, so dy/dx[0] = sum_k cos(x[0] + k).
    expected = sum(math.cos(0.1 + k) for k in range(64))
    assert x.grad[0] == pytest.approx(expected, rel=1e-4)


@test_utils.test(require=qd.extension.adstack, offline_cache=False)
def test_adstack_heap_backed_exceeds_old_threadstack_budget():
    # Multiple loop-carried values must retain independent primal histories and produce the analytical gradient.
    n = 4
    n_iter = 32
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            a3 = 0.0
            a4 = 0.0
            a5 = 0.0
            a6 = 0.0
            a7 = 0.0
            for _ in range(n_iter):
                a0 = a0 * 0.9 + x[i]
                a1 = a1 * 0.9 + x[i]
                a2 = a2 * 0.9 + x[i]
                a3 = a3 * 0.9 + x[i]
                a4 = a4 * 0.9 + x[i]
                a5 = a5 * 0.9 + x[i]
                a6 = a6 * 0.9 + x[i]
                a7 = a7 * 0.9 + x[i]
            y[None] += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7

    for i in range(n):
        x[i] = 0.1 * (i + 1)
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()
    qd.sync()

    geom = sum(0.9**j for j in range(n_iter))
    for i in range(n):
        assert x.grad[i] == pytest.approx(8.0 * geom, rel=1e-5)


@pytest.mark.parametrize("n_iter", [30, 100])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_near_capacity(n_iter):
    # Exercise both short and longer field-controlled loops with the same analytical gradient check.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    n_iter_fld = qd.field(qd.i32, shape=())
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter_fld[None]):
                y[None] += qd.sin(v)
                v = v + 1.0

    x[0] = 0.1
    n_iter_fld[None] = n_iter
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0

    compute.grad()
    qd.sync()
    expected = sum(math.cos(0.1 + k) for k in range(n_iter))
    # `rel=1e-4` rather than 1e-5: n_iter=100 accumulates a ~1e-5 relative drift on AMD Vulkan (RADV) that the
    # tighter bound catches but that is within f32 accumulation noise for a 100-term oscillating cosine sum.
    assert x.grad[0] == test_utils.approx(expected, rel=1e-4)


def _run_sum_linear(
    qd_dtype, use_static_loop, use_varying_coeff, n_iter, rel_tol, approx=test_utils.approx, abs_tol=None
):
    n = 4
    x = qd.field(qd_dtype, shape=n, needs_grad=True)
    y = qd.field(qd_dtype, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for a in qd.static(range(n_iter)) if qd.static(use_static_loop) else range(n_iter):
                if qd.static(use_varying_coeff):
                    y[None] += v * qd.cast(a + 1, qd_dtype)
                else:
                    y[None] += v

    x_vals = [0.1, 0.3, 0.5, 0.8]
    for i, v in enumerate(x_vals):
        x[i] = v
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    expected = sum(a + 1 for a in range(n_iter)) if use_varying_coeff else float(n_iter)
    approx_kwargs = {"rel": rel_tol}
    if abs_tol is not None:
        approx_kwargs["abs"] = abs_tol
    for i in range(n):
        assert x.grad[i] == approx(expected, **approx_kwargs)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("use_static_loop", [True, False])
@pytest.mark.parametrize("use_varying_coeff", [True, False])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_sum_linear(use_static_loop, use_varying_coeff, n_iter):
    # Check linear accumulation for static and dynamic loops, with constant and varying coefficients.
    _run_sum_linear(qd.f32, use_static_loop, use_varying_coeff, n_iter, rel_tol=1e-6)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("use_static_loop", [True, False])
@pytest.mark.parametrize("use_varying_coeff", [True, False])
@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_sum_linear_f64(use_static_loop, use_varying_coeff, n_iter):
    # f64 counterpart uses `pytest.approx` so the tight rel_tol is not floored by `test_utils.approx`, and
    # `abs_tol=0` disables pytest.approx's default `abs=1e-12` floor so the 1e-14 relative tolerance is actually
    # honored. Note: the expected gradients here are integers in `{1, 3, 6, 10, 55}` which are exactly representable
    # in both f32 and f64, so this test cannot catch an f32-narrowing regression of the backward pass regardless of
    # tolerance - the value coverage comes from `test_adstack_basic_gradient_f64` whose non-integer expected values
    # genuinely exercise the f64 precision floor. Kept here to preserve the shape coverage (static/dynamic inner
    # loop x varying/constant coefficient) at f64 since a future type-narrowing bug that depends on shape might
    # still surface through compile-time / structural differences between f32 and f64 codegen paths.
    _run_sum_linear(qd.f64, use_static_loop, use_varying_coeff, n_iter, rel_tol=1e-14, approx=pytest.approx, abs_tol=0)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_runtime_if_wrapping_loop_with_carried_var():
    # Pins the MakeAdjoint::visit(RangeForStmt) current_block-restore behaviour. Reverse-mode AD through a dynamic
    # for-loop with a loop-carried float, nested inside a runtime-guarded `if`, must emit its post-loop reverse
    # stmts (stack-underflow cleanup and the `accumulate x.grad[i]` on the initial-value stmt) as siblings *after*
    # the reversed for-loop, not inside its body. Without the save/restore of `current_block` around the per-stmt
    # iteration, these stmts land inside the body and the gradient is silently wrong
    # (e.g. `1 + 1 + 1 = 3.0` instead of `1 + 0.95 + 0.95**2 = 2.8525`). Compile-time-true `if` branches do not
    # trigger the pattern because simplify folds them away before reverse-mode is applied.
    #
    # This shape is common in user code: a reverse-mode kernel reads fields through runtime index-range guards
    # around dynamic-loop bodies that carry floats across iterations. Without the save/restore this produces NaN
    # on Metal and zero-valued gradients on CPU.
    n_iter = 4
    n_active = 3
    n_max = n_active + 2  # outer loop iterates past n_active; body guarded by `i < n_active`.

    x = qd.field(qd.f32, shape=n_max, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)
    n_arr = qd.field(qd.i32, shape=())

    @qd.kernel
    def compute():
        for i in range(n_max):
            if i < n_arr[None]:
                v = x[i]
                acc = 0.0
                for _ in range(n_iter):
                    acc = acc + v
                    v = v * 0.95 + 0.01
                y[None] += acc

    for i in range(n_max):
        x[i] = 1.0 + 0.1 * i
    n_arr[None] = n_active
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n_max):
        x.grad[i] = 0.0
    compute.grad()

    expected = sum(0.95**k for k in range(n_iter))
    for i in range(n_active):
        assert x.grad[i] == pytest.approx(expected, rel=1e-6)
    for i in range(n_active, n_max):
        assert x.grad[i] == 0.0


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_if_cond_snapshot_through_dynamic_for():
    # Pins MakeAdjoint::visit(IfStmt) cond-snapshot behaviour. Reverse-mode AD through a runtime `if` whose
    # cond is a stack-backed alloca load, nested inside a dynamic-range for-loop, must evaluate the reverse
    # if-cond at the forward-time value - not by re-running `stack_load_top` at reverse-time. Without the
    # snapshot, BackupSSA's cross-block clone of `if_stmt->cond` re-reads the cond's backing adstack at
    # reverse time, where the top has advanced due to the short-circuit push emitted inside the forward if
    # body; the reverse branch flips, the accumulation never runs, and gradients silently come out all-zero.
    #
    # Internal details: `cfg_optimization=False` is load-bearing - with it enabled, store-to-load forwarding
    # collapses the tautological `if (stack_load_top after push_true) { ... }` short-circuit wrapper before
    # MakeAdjoint sees it, and the multi-push-per-iter pattern that drives the bug vanishes. The outer
    # `qd.ndrange` (not a plain Python `for`) is required: the plain-`for` variant keeps the outer index as
    # a direct loop index rather than a cast alloca, and the enclosing-loop cast is what forces the inner
    # cond alloca to be stack-promoted. `qd.cast(i_b, qd.i32)` is also load-bearing for the same reason -
    # without it the cond alloca stays unpromoted and the bug does not surface. The inner-loop bound `n[None]`
    # pulled from a field, not a Python literal, forces the inner for to be compiled as a runtime-dynamic
    # range rather than statically unrolled; the buggy stack clone cannot arise on the fully-unrolled path.
    # The min shape is 2 iterations (`i == 0` writes a vector from `x`; else reads a constant `c[i]`); the
    # expected grad `[1, 1, 1, 0]` makes a flipped reverse branch visible immediately because flipping drops
    # the `x[0..2]` accumulation entirely and the whole grad comes out `[0, 0, 0, 0]`.
    vec3 = qd.types.vector(3, qd.f32)

    outputs = qd.field(dtype=vec3, shape=(2, 1), needs_grad=True)
    constants = qd.field(dtype=vec3, shape=(2,))
    n_iter = qd.field(dtype=qd.i32, shape=())
    inputs = qd.field(dtype=qd.f32, shape=(4, 1), needs_grad=True)

    @qd.kernel
    def my_kernel():
        for i_batch in qd.ndrange(outputs.shape[1]):
            i_batch = qd.cast(i_batch, qd.i32)
            for i_inner in range(n_iter[None]):
                if i_inner == 0:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[0, i_batch], inputs[1, i_batch], inputs[2, i_batch]], dt=qd.f32
                    )
                else:
                    outputs[i_inner, i_batch] = constants[i_inner]

    outputs.grad.from_numpy(np.ones((2, 1, 3), dtype=np.float32))
    n_iter[None] = 2

    my_kernel.grad()

    grad = inputs.grad.to_numpy().squeeze()
    assert grad[0] == 1.0
    assert grad[1] == 1.0
    assert grad[2] == 1.0
    assert grad[3] == 0.0


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_if_cond_snapshot_adaptive_sizing():
    # Keep cfg_optimization=False to retain branch snapshots and multiple pushes per iteration.
    vec3 = qd.types.vector(3, qd.f32)

    outputs = qd.field(dtype=vec3, shape=(4, 1), needs_grad=True)
    constants = qd.field(dtype=vec3, shape=(4,))
    n_iter = qd.field(dtype=qd.i32, shape=())
    inputs = qd.field(dtype=qd.f32, shape=(4, 1), needs_grad=True)

    @qd.kernel
    def my_kernel():
        for i_batch in qd.ndrange(outputs.shape[1]):
            i_batch = qd.cast(i_batch, qd.i32)
            for i_inner in range(n_iter[None]):
                if i_inner == 0:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[0, i_batch], inputs[1, i_batch], inputs[2, i_batch]], dt=qd.f32
                    )
                elif i_inner == 1:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[1, i_batch], inputs[2, i_batch], inputs[3, i_batch]], dt=qd.f32
                    )
                elif i_inner == 2:
                    outputs[i_inner, i_batch] = qd.Vector(
                        [inputs[0, i_batch], inputs[2, i_batch], inputs[3, i_batch]], dt=qd.f32
                    )
                else:
                    outputs[i_inner, i_batch] = constants[i_inner]

    outputs.grad.from_numpy(np.ones((4, 1, 3), dtype=np.float32))
    n_iter[None] = 4

    my_kernel.grad()

    grad = inputs.grad.to_numpy().squeeze()
    assert grad[0] == 2.0
    assert grad[1] == 2.0
    assert grad[2] == 3.0
    assert grad[3] == 2.0


@test_utils.test(require=qd.extension.adstack)
def test_adstack_sibling_for_loops_reverse_order():
    # Reverse-mode AD through two sibling dynamic for-loops in the same container, where the second loop reads a global
    # that the first loop wrote, must execute the second loop's reverse before the first loop's reverse. Otherwise the
    # first-loop reverse reads an uninitialised (zero) adjoint of the intermediate global, clears it, and the gradient
    # the second-loop reverse later populates propagates nowhere. Left unfixed, `inputs.grad` comes out all-zeros
    # despite a well-defined non-zero analytic derivative.
    #
    # Internal details: MakeAdjoint runs per-IB, and for this shape both sibling fors' bodies are their own IBs
    # (innermost loops with global ops). The reverse-mode transform therefore never visits the container block that
    # holds them, so nothing flips their order. ReverseOuterLoops flips each loop's `reversed` iteration direction and
    # also pairwise-swaps sibling for-loops inside every non-IB container block the pass walks through. Non-loop
    # statements (range-bound loads, alloca, etc.) stay at their original positions so SSA operands still dominate both
    # swapped fors. The outer `for _ in range(1)` dummy is the smallest shape that places the two siblings inside a
    # non-IB container (the frontend rejects a bare sequence of top-level for-loops as "mixed usage of for-loops and
    # statements without looping"); `n[None]` from a field forces the inner ranges to be dynamic so the bug manifests
    # (static-unrolled ranges go through a different path that already works).
    size = 3
    n = qd.field(qd.i32, shape=())

    inputs = qd.field(qd.f32, shape=size, needs_grad=True)
    weights = qd.field(qd.f32, shape=size)
    scratch = qd.field(qd.f32, shape=size, needs_grad=True)
    outputs = qd.field(qd.f32, shape=size, needs_grad=True)

    @qd.kernel
    def my_kernel():
        for _ in range(1):
            for i in range(n[None]):
                scratch[i] = inputs[i]
            for i in range(n[None]):
                outputs[i] = scratch[i] * weights[i]

    n[None] = size
    for i in range(size):
        inputs[i] = float(i + 1)
        weights[i] = float(i + 1) * 0.5

    my_kernel()
    outputs.grad.from_numpy(np.ones(size, dtype=np.float32))
    my_kernel.grad()

    grad = inputs.grad.to_numpy()
    for i in range(size):
        assert grad[i] == float(i + 1) * 0.5


@pytest.mark.parametrize("n", [3, 5])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_inner_for_bound_is_enclosing_loop_index(n):
    # Reverse-mode AD must handle a triangular-nested loop, where an inner for-loop's upper bound is itself an
    # enclosing loop's index (a per-iteration value, not a loop invariant). The kernel mirrors a classic
    # lower-triangular sweep like an in-place Cholesky update. `w` additionally accumulates a linear function of
    # the outer counter alongside the inner sum, so both a per-iteration value and its reverse-mode gradient flow
    # are exercised. `x` entries are distinct (0.1, 0.2, ...) so the inner for's contribution to each `x.grad[k]`
    # differs per iteration; a uniform `x` would collapse several contributions into the same number and would
    # let a reverse pass over the wrong iteration range still match the expected sum.
    #
    # Internal details: two pieces of the autodiff pipeline are load-bearing together.
    # (1) AdStackAllocaJudger::visit(RangeForStmt) recognises allocas whose LocalLoad feeds a RangeForStmt begin or
    #     end and promotes them to an adstack, so each forward iteration pushes the current bound and each reverse
    #     iteration pops the matching one. This is the only promotion path for a pure loop-counter alloca (LOAD-only
    #     into the inner bound, no LOAD-then-STORE cycle), so the `local_loaded_` short-circuit in visit(LocalStoreStmt)
    #     cannot cover it. The comparison has to resolve the LocalLoad chain (compare `ll->src` to the backup, not
    #     the operand itself or the cursor) because `begin`/`end` are always value-producing stmts, not the alloca,
    #     and the mutable cursor only names the first matching load's instance.
    # (2) BackupSSA::visit(RangeForStmt) spills cross-block operands on the for-stmt itself (not just its body), so
    #     the reverse clone's `end` operand - pointing at the forward-scope AdStackLoadTop - is rematerialised via
    #     op->clone() inside the reverse scope (the existing AdStackLoadTopStmt branch in generic_visit).
    # Without either piece, this kernel fails: IR-verify reports `RangeForStmt cannot have operand LocalLoadStmt`
    # without (2); LLVM codegen hits "Instruction does not dominate all uses" without (2) after (1); or the reverse
    # inner-loop iteration count is the last forward value without (1), silently corrupting gradients for the
    # earliest inner indices (those visited most often across outer iterations) - bigger `n` exposes more affected
    # indices, which is why the parametrize sweep catches a regression that a single small `n` can alias past.
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)
    w = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            for j in range(n):
                if i < n and j < i + 1:
                    s = 0.0
                    for k in range(j):
                        s = s + x[k] * x[k]
                    w[None] += qd.cast(j, qd.f32) * x[i]
                    y[None] += qd.sqrt(qd.math.clamp(x[i] + 1.0 - s, 0.01, 1e9))

    x_vals = [0.1 * (k + 1) for k in range(n)]
    for k in range(n):
        x[k] = x_vals[k]
    y[None] = 0.0
    w[None] = 0.0
    compute()
    y.grad[None] = 1.0
    w.grad[None] = 1.0
    for k in range(n):
        x.grad[k] = 0.0
    compute.grad()

    expected = [0.0] * n
    for i in range(n):
        for j in range(i + 1):
            s = sum(x_vals[k] * x_vals[k] for k in range(j))
            arg = x_vals[i] + 1.0 - s
            d_arg = 1.0 / (2.0 * arg**0.5)
            expected[i] += d_arg
            for k in range(j):
                expected[k] += d_arg * (-2.0 * x_vals[k])
            # w_contribution = cast(j) * x[i]: d/dx[i] += j
            expected[i] += float(j)
    for k in range(n):
        assert x.grad[k] == test_utils.approx(expected[k], rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_ndrange_over_ndarray_shape_does_not_oversize_heap():
    # A multidimensional ndarray loop with ten carried values must fit and produce correct per-cell gradients.
    rows, cols = 2, 3

    @qd.kernel
    def compute(arr: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i, j in qd.ndrange(arr.shape[0], arr.shape[1]):
            v0 = arr[i, j]
            v1 = arr[i, j] + 0.1
            v2 = arr[i, j] + 0.2
            v3 = arr[i, j] + 0.3
            v4 = arr[i, j] + 0.4
            v5 = arr[i, j] + 0.5
            v6 = arr[i, j] + 0.6
            v7 = arr[i, j] + 0.7
            v8 = arr[i, j] + 0.8
            v9 = arr[i, j] + 0.9
            for _ in range(3):
                v0 = v0 * 1.01 + 0.01
                v1 = v1 * 1.02 + 0.02
                v2 = v2 * 1.03 + 0.03
                v3 = v3 * 1.04 + 0.04
                v4 = v4 * 1.05 + 0.05
                v5 = v5 * 1.06 + 0.06
                v6 = v6 * 1.07 + 0.07
                v7 = v7 * 1.08 + 0.08
                v8 = v8 * 1.09 + 0.09
                v9 = v9 * 1.10 + 0.10
            out[0] += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9

    arr_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    arr = qd.ndarray(qd.f32, shape=(rows, cols), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    arr.from_numpy(arr_np)
    arr.grad.from_numpy(np.zeros_like(arr_np))
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))

    compute(arr, out)
    compute.grad(arr, out)
    qd.sync()

    got_grad = arr.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"ndrange-over-shape grad returned NaN: {got_grad}"

    # Analytic oracle. The kernel is affine in `arr[i, j]` (each `v_k` is `v_k * c_k + d_k` for three iterations, so
    # `d(v_k_final) / d(arr[i, j]) = c_k^3`), and `out[0]` sums all ten recurrences, so the closed-form gradient per
    # cell is `sum_k c_k^3`. Independent of the backward emission so a wrong-but-non-NaN gradient (the failure mode when
    # the adstack heap was bound to Metal's nil-fallback and reads came back as zero) still trips the assertion;
    # tolerance bounded by f32 accumulation roundoff only, not finite-difference cancellation.
    coeffs = np.array([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10], dtype=np.float64)
    expected_per_cell = float((coeffs**3).sum())
    # `rtol=1e-4` rather than tight-to-backward-roundoff because on AMD Vulkan (RADV) the adjoint accumulation
    # through ten loop-carried recurrences drifts by a few parts in 1e5 relative to the analytic value; the
    # tighter bound catches the drift without a corresponding correctness signal. The regression this test
    # guards against (nil device buffer -> zero read -> NaN adjoint) trips any tolerance at all.
    np.testing.assert_allclose(got_grad, np.full_like(arr_np, expected_per_cell), rtol=1e-4, atol=0)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_bounded_inner_loop_sized_by_structural_prepass():
    # Constant inner-loop bounds must produce the analytical gradient through a nonlinear recurrence.
    x = qd.field(qd.f32, shape=(1,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(5):
                v = qd.sin(v) + 0.1
            y[None] += v

    n_iter = 5

    x[0] = 0.3
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    v = 0.3
    dv_dx = 1.0
    for _ in range(n_iter):
        dv_dx = math.cos(v) * dv_dx
        v = math.sin(v) + 0.1
    # `pytest.approx` rather than `test_utils.approx` so the tight f32 tolerance isn't floored to the
    # backend `get_rel_eps()` - a wrong gradient from an unresolved adstack would drift by orders of
    # magnitude, not parts per million, but a tight bound still catches subtler regressions.
    assert x.grad[0] == pytest.approx(dv_dx, rel=1e-6)


@test_utils.test(require=qd.extension.adstack, default_ip=qd.i64)
def test_adstack_bounded_inner_loop_pre_pass_handles_i64_bounds():
    # Compile integer bounds as i64 while keeping the nonlinear primal and gradient arithmetic at f32.
    x = qd.field(qd.f32, shape=(1,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(3):
                v = qd.sin(v) + 0.1
            y[None] += v

    x[0] = 0.2
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()

    v = 0.2
    dv_dx = 1.0
    for _ in range(3):
        dv_dx = math.cos(v) * dv_dx
        v = math.sin(v) + 0.1
    # `default_ip=qd.i64` only forces i64 loop bounds; floating-point arithmetic stays at
    # `default_fp` (f32). `pytest.approx` bypasses the backend `get_rel_eps()` floor so the tight
    # bound actually bites.
    assert x.grad[0] == pytest.approx(dv_dx, rel=1e-6)


@pytest.mark.parametrize("trip_mode", ["element", "shape"])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_nested_tripcount_gradient(trip_mode):
    # Pins the reverse-mode gradient through three nested range-fors whose inner bounds toggle between the
    # two shapes the structural pre-pass must recognise as trip-count sources: an ndarray element read
    # (`range(n[i])`, lowering to a `MaxOverRange`-wrapped `ExternalTensorRead`) and a shape-only reference
    # (`range(n.shape[0])`, lowering to a bare `ExternalTensorShape`). Both forms drive a nested adstack with
    # `x[j] * x[j]` pushes on every innermost iteration; a trip-count bound that underestimates the true
    # push count trips an overflow at `qd.sync()` rather than a silent numerical drift.
    #
    # Internal details: `qd.static(IS_ELEMENT)` picks the range at compile time so each parametrisation emits
    # a different `SizeExpr` tree without any runtime branch in the kernel. The `element` leg is the shape
    # the `ExternalTensorRead`-as-trip-body grammar gap is narrowed around; the `shape` leg is the
    # always-resolvable baseline and pins that shape-only nesting keeps working when the element form is
    # rejected or widened.
    IS_ELEMENT = trip_mode == "element"
    N_X = 32
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(n: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i_e in range(n.shape[0]):
            for i_l in range(n[i_e]) if qd.static(IS_ELEMENT) else range(n.shape[0]):
                accum = 0.0
                for j in range(n[i_l]) if qd.static(IS_ELEMENT) else range(n.shape[0]):
                    accum = accum + x[j] * x[j]
                loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1
    n_np = np.array([2, 3, 4, 1], dtype=np.int32)
    compute(n_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(n_np)
    qd.sync()

    n_shape = int(n_np.shape[0])
    expected_loss = 0.0
    expected_grad = [0.0] * N_X
    for i_e in range(n_shape):
        middle_end = int(n_np[i_e]) if IS_ELEMENT else n_shape
        for i_l in range(middle_end):
            inner_end = int(n_np[i_l]) if IS_ELEMENT else n_shape
            for j in range(inner_end):
                expected_loss += 0.1 * 0.1
                expected_grad[j] += 2.0 * 0.1

    assert loss[None] == pytest.approx(expected_loss, rel=1e-5)
    for k in range(N_X):
        if expected_grad[k] == 0.0:
            assert x.grad[k] == pytest.approx(0.0, abs=1e-7)
        else:
            assert x.grad[k] == pytest.approx(expected_grad[k], rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_field_load_bounded_loop_evaluated_per_launch():
    # Changing a field-controlled loop bound between invocations must change both sizing and gradients.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    n_iter_fld = qd.field(qd.i32, shape=())
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter_fld[None]):
                v = v * 0.95 + 0.01
            y[None] += v

    for n_iter in (1, 20, 50):
        x[0] = 0.1
        n_iter_fld[None] = n_iter
        y[None] = 0.0
        compute()
        y.grad[None] = 1.0
        x.grad[0] = 0.0
        compute.grad()
        qd.sync()
        expected = 0.95**n_iter
        assert x.grad[0] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("ndarray_kind", ["numpy", "qd_ndarray"])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_inner_range_bounded_by_ndarray_read_at_outer_index(ndarray_kind):
    # Pins the `ExternalTensorRead`-over-`LoopIndex` `MaxOverRange` wrap in the `SizeExpr` pre-pass: a reverse-mode
    # adstack whose inner range `range(a[i])` is bounded by a scalar ndarray read at the enclosing outer loop index. The
    # pre-pass must build `MaxOverRange(var, 0, outer_end, ExternalTensorRead(a, [var]))` for the alloca's multiplier;
    # the launch-time evaluator enumerates the outer range, reads `a[var]` at each iteration, and takes the max to size
    # the per-thread adstack heap exactly.
    #
    # Internal details: runs on every backend. On CPU the launch-time SizeExpr is evaluated host-side via
    # `evaluate_adstack_size_expr`, with ndarray element reads going through the real host pointer
    # `set_host_accessible_ndarray_ptrs` mirrored into `array_ptrs`. On CUDA / AMDGPU the host encodes the tree into
    # device-side bytecode (`encode_adstack_size_expr_device_bytecode`) and calls `runtime_eval_adstack_size_expr` to
    # run the interpreter on the device. On Metal / Vulkan the bytecode is emitted as a SPIR-V compute shader launched
    # from `GfxRuntime::launch_kernel`. All three paths are the only way to resolve an `ExternalTensorRead` against a
    # GPU-private ndarray without round-tripping the whole allocation to host. Asserts the analytical gradient `0.95 **
    # a[i]` per outer iteration so a regression in the wrap or in either the host or device evaluator shows up as a
    # value mismatch rather than an overflow crash. Parametrised over the ndarray argument kind because `numpy`/torch
    # inputs lower through `set_arg_external_array_with_shape` (writes the raw host pointer straight into `array_ptrs`)
    # while `qd.ndarray` inputs lower through `set_arg_ndarray_impl` + `set_ndarray_ptrs` (stashes a `DeviceAllocation
    # *` first, then the launcher resolves it). The CPU launcher mirrors the resolved pointer back into `array_ptrs`;
    # the CUDA / AMDGPU launchers don't need to because the device interpreter reads the ndarray data pointer straight
    # out of `ctx->arg_buffer` at the offset the encoder precomputed from `args_type`, sidestepping the host-side
    # `array_ptrs` map entirely.
    N = 4
    arr_data = np.array([2, 3, 1, 2], dtype=np.int32)

    x = qd.field(qd.f32, shape=(N,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in x:
            v = x[i]
            n = a[i]
            for _ in range(n):
                v = v * 0.95 + 0.01
            y[None] += v

    for i in range(N):
        x[i] = 0.1

    if ndarray_kind == "numpy":
        a = arr_data
    else:
        a = qd.ndarray(qd.i32, shape=(N,))
        a.from_numpy(arr_data)

    compute(a)
    y.grad[None] = 1.0
    for i in range(N):
        x.grad[i] = 0.0
    compute.grad(a)
    qd.sync()

    for i in range(N):
        expected = 0.95 ** int(arr_data[i])
        assert x.grad[i] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "trip_count_source",
    ["qd_ndarray", "field"],
    ids=["qd_ndarray", "field"],
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_metadata_cache_invalidates_on_host_mutation(trip_count_source):
    # Changing host-side trip counts must affect the next backward, including after a prior launch.
    N = 4
    N_X = 16

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    if trip_count_source == "qd_ndarray":
        n_obj = qd.ndarray(qd.i32, shape=(N,))

        @qd.kernel
        def compute(n: qd.types.ndarray(dtype=qd.i32, ndim=1)):
            for i_e in range(n.shape[0]):
                accum = 0.0
                for j in range(n[i_e]):
                    accum = accum + x[j] * x[j]
                loss[None] += accum

        def set_n(val):
            for i in range(N):
                n_obj[i] = val

        def call_compute():
            compute(n_obj)

        def call_compute_grad():
            compute.grad(n_obj)

    else:
        n_obj = qd.field(qd.i32, shape=(N,))

        @qd.kernel
        def compute():
            for i_e in range(N):
                accum = 0.0
                for j in range(n_obj[i_e]):
                    accum = accum + x[j] * x[j]
                loss[None] += accum

        def set_n(val):
            for i in range(N):
                n_obj[i] = val

        def call_compute():
            compute()

        def call_compute_grad():
            compute.grad()

    set_n(8)
    for i in range(N_X):
        x[i] = 0.1
    loss[None] = 0.0
    call_compute()
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    call_compute_grad()
    qd.sync()
    assert loss[None] == pytest.approx(N * 8 * 0.01, rel=1e-5)
    for k in range(8):
        assert x.grad[k] == pytest.approx(N * 2 * 0.1, rel=1e-5)
    for k in range(8, N_X):
        assert x.grad[k] == pytest.approx(0.0, abs=1e-7)

    set_n(16)
    for i in range(N_X):
        x[i] = 0.1
    loss[None] = 0.0
    call_compute()
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    call_compute_grad()
    qd.sync()
    assert loss[None] == pytest.approx(N * 16 * 0.01, rel=1e-5)
    for k in range(N_X):
        assert x.grad[k] == pytest.approx(N * 2 * 0.1, rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_inner_range_bounded_by_multidim_ndarray_read():
    # A multidimensional array read must select the right per-iteration loop bound.
    rows, cols = 3, 5
    arr_np = np.zeros((rows, cols), dtype=np.int32)
    arr_np[2, 2] = 100  # sole non-zero cell; stride-1 sum never visits (i=2, j=2) -> max evaluated to 0
    # Sanity: stride-1 sum over `(i, j)` with `i + j < rows + cols - 1` stays within the zero band.
    n_max_true = int(arr_np.max())
    n_max_stride_one = 0
    for i in range(rows):
        for j in range(cols):
            if i + j < rows + cols - 1:
                n_max_stride_one = max(n_max_stride_one, int(arr_np.flatten()[min(i + j, rows * cols - 1)]))
    assert n_max_true == 100 and n_max_stride_one == 0

    N_X = 8
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=2)):
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                v = x[0]
                for _ in range(a[i, j]):
                    v = v * 0.95 + 0.01
                y[None] += v

    for i in range(N_X):
        x[i] = 0.1

    compute(arr_np)
    y.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_np)
    qd.sync()

    # `y = sum_{i, j} v_final(a[i, j])` where `v_final(n) = 0.95 ** n * x[0] + const_terms(n)`. Gradient wrt
    # x[0] is `sum_{i, j} 0.95 ** a[i, j]`; with `a[2, 2] = 100` and all other cells zero the dominant
    # contribution is `14 + 0.95 ** 100 ~ 14.006`.
    expected = float(np.sum(0.95 ** arr_np.astype(np.float64)))
    assert x.grad[0] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("outer_bound", ["const", "dynamic"])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_ext_tensor_read_indexed_by_stashed_outer_loop_var(outer_bound):
    # An external-array trip count indexed through a stashed outer-loop index must produce correct gradients.
    N_ENT = 1
    link_start_np = np.array([0], dtype=np.int32)
    link_end_np = np.array([2], dtype=np.int32)
    joint_start_np = np.array([0, 1], dtype=np.int32)
    joint_end_np = np.array([1, 3], dtype=np.int32)
    parent_idx_np = np.array([-1, 0], dtype=np.int32)

    N_X = 4
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    IS_CONST = outer_bound == "const"

    @qd.kernel
    def compute(
        link_start: qd.types.ndarray(dtype=qd.i32, ndim=1),
        link_end: qd.types.ndarray(dtype=qd.i32, ndim=1),
        joint_start: qd.types.ndarray(dtype=qd.i32, ndim=1),
        joint_end: qd.types.ndarray(dtype=qd.i32, ndim=1),
        parent_idx: qd.types.ndarray(dtype=qd.i32, ndim=1),
    ):
        for i_e in range(N_ENT) if qd.static(IS_CONST) else range(link_start.shape[0]):
            for i_l_ in range(link_start[i_e], link_end[i_e]):
                i_l = qd.cast(i_l_, qd.i32)
                accum = x[i_l] * 0.5
                if parent_idx[i_l] != -1:
                    accum = accum + x[parent_idx[i_l]] * 0.3
                n_joints = joint_end[i_l] - joint_start[i_l]
                for i_j_ in range(n_joints):
                    i_j = i_j_ + joint_start[i_l]
                    accum = accum + x[i_j] * x[i_j]
                loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1

    compute(link_start_np, link_end_np, joint_start_np, joint_end_np, parent_idx_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(link_start_np, link_end_np, joint_start_np, joint_end_np, parent_idx_np)
    qd.sync()

    # Analytical gradients at x[i] = 0.1:
    #   x[0]: 0.5 (i_l=0 self) + 0.3 (i_l=1 parent) + 2*x[0] (i_j=0, i_l=0) = 1.0
    #   x[1]: 0.5 (i_l=1 self) + 2*x[1] (i_j=1, i_l=1)                      = 0.7
    #   x[2]:                                   2*x[2] (i_j=2, i_l=1)      = 0.2
    #   x[3]: 0
    assert x.grad[0] == pytest.approx(1.0, rel=1e-6)
    assert x.grad[1] == pytest.approx(0.7, rel=1e-6)
    assert x.grad[2] == pytest.approx(0.2, rel=1e-6)
    assert x.grad[3] == pytest.approx(0.0, abs=1e-7)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_field_ptr_indexed_by_stashed_outer_loop_var():
    # A field trip count indexed through a stashed outer-loop index must produce correct gradients.
    N_OUTER = 4
    link_start = qd.field(qd.i32, shape=(N_OUTER,))
    link_end = qd.field(qd.i32, shape=(N_OUTER,))

    N_X = 6
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(arr: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i_outer in range(arr.shape[0]):
            i_oc = qd.cast(i_outer, qd.i32)
            for i_inner in range(link_start[i_oc], link_end[i_oc]):
                v = x[i_inner] * 0.5
                for _ in range(2):
                    v = v * 0.95 + 0.01
                loss[None] += v

    link_start[0] = 0
    link_start[1] = 1
    link_start[2] = 3
    link_start[3] = 4
    link_end[0] = 1
    link_end[1] = 3
    link_end[2] = 4
    link_end[3] = 6
    for i in range(N_X):
        x[i] = 0.1

    arr = np.zeros(N_OUTER, dtype=np.float32)
    compute(arr)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr)
    qd.sync()

    # Analytical gradient at each `x[i_inner]`: each slot is read once, each read contributes
    # `d(loss)/d(x) = 0.5 * 0.95 * 0.95 = 0.45125`.
    for i in range(N_X):
        assert x.grad[i] == pytest.approx(0.45125, rel=1e-5)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_triangular_ndrange_self_referential_push_idempotency():
    # Triangular nested ranges and repeated stack snapshots must compile and differentiate consistently.
    outer_size = qd.field(qd.i32, shape=(1,))
    group_begin = qd.field(qd.i32, shape=(1,))
    group_end = qd.field(qd.i32, shape=(1,))
    group_base = qd.Vector.field(3, qd.f32, shape=(1,))
    sub_begin = qd.field(qd.i32, shape=(1,))
    sub_end = qd.field(qd.i32, shape=(1,))
    src_offset = qd.field(qd.i32, shape=(1,))
    dst_offset = qd.field(qd.i32, shape=(1,))
    dst_buf = qd.field(qd.f32, shape=(6, 1), needs_grad=True)
    src_buf = qd.field(qd.f32, shape=(7, 1), needs_grad=True)
    batch_probe = qd.Vector.field(3, qd.f32, shape=(1, 1), needs_grad=True)
    dst_vec_buf = qd.Vector.field(4, qd.f32, shape=(1, 1), needs_grad=True)
    src_vec_buf = qd.Vector.field(4, qd.f32, shape=(1, 2, 1), needs_grad=True)

    @qd.kernel
    def compute(
        batch_probe: qd.template(),
        dst_vec_buf: qd.template(),
        src_vec_buf: qd.template(),
        group_base: qd.template(),
        sub_begin: qd.template(),
        sub_end: qd.template(),
        src_offset: qd.template(),
        dst_offset: qd.template(),
        dst_buf: qd.template(),
        outer_size: qd.template(),
        group_begin: qd.template(),
        group_end: qd.template(),
        src_buf: qd.template(),
    ):
        for i_e, i_b in qd.ndrange(outer_size.shape[0], batch_probe.shape[1]):
            for j_e in range(i_e, outer_size.shape[0]):
                for i_l in range(group_begin[j_e], group_end[j_e]):
                    base = group_base[i_l]
                    n_sub = sub_end[i_l] - sub_begin[i_l]
                    for k_ in range(n_sub):
                        k = k_ + sub_begin[i_l]
                        s = src_offset[k]
                        a = src_buf[s + 4, i_b]
                        b = src_buf[s + 5, i_b]
                        c = src_buf[s + 6, i_b]
                        packed = qd.Vector([a, b, c], dt=qd.f32)
                        d = dst_offset[k]
                        for j in qd.static(range(3)):
                            dst_buf[d + j, i_b] = base[j]
                            dst_buf[d + 3 + j, i_b] = packed[j]
                    dst_vec_buf[i_l, i_b] = src_vec_buf[i_l, n_sub, i_b]

    outer_size[0] = 1
    group_begin[0] = 0
    group_end[0] = 1
    sub_begin[0] = 0
    sub_end[0] = 1
    src_offset[0] = 0
    dst_offset[0] = 0

    # The grad compile must complete without raising RuntimeError("stash data-flow cycle ..."). The assertion is the
    # absence of that RuntimeError - no gradient value is checked because the minimal-shape fields have a single element
    # and the regression is purely compile-time cycle detection.
    compute.grad(
        batch_probe,
        dst_vec_buf,
        src_vec_buf,
        group_base,
        sub_begin,
        sub_end,
        src_offset,
        dst_offset,
        dst_buf,
        outer_size,
        group_begin,
        group_end,
        src_buf,
    )
    qd.sync()


@pytest.mark.parametrize("inner_loop_shape", ["begin_end", "sub_then_range"])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_structural_pre_pass_fuses_sub_of_max_over_range_with_matching_shape_ends(inner_loop_shape):
    # Check gradients when begin and end bounds are read from arrays of matching shape.
    IS_BEGIN_END = inner_loop_shape == "begin_end"
    N_X = 8

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(start_arr: qd.types.ndarray(dtype=qd.i32, ndim=1), end_arr: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i_o in range(start_arr.shape[0]):
            s = start_arr[i_o]
            e = end_arr[i_o]
            accum = 0.0
            for i_j_ in range(s, e) if qd.static(IS_BEGIN_END) else range(e - s):
                i_j = i_j_ if qd.static(IS_BEGIN_END) else (i_j_ + s)
                accum = accum + x[i_j] * x[i_j] * qd.cast(e - s, qd.f32)
            loss[None] += accum

    # `start = [0, 3]`, `end = [3, 4]`: per-slot trips are (3, 1). `e - s` per slot is (3, 1) so the inner
    # body adds `trip * x[i_j]^2` across 4 inner iterations.
    for i in range(N_X):
        x[i] = 0.1
    start_np = np.array([0, 3], dtype=np.int32)
    end_np = np.array([3, 4], dtype=np.int32)

    compute(start_np, end_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(start_np, end_np)
    qd.sync()

    # loss = sum over i_j in [0, 3) of `3 * x[i_j]^2` + sum over i_j in [3, 4) of `1 * x[i_j]^2`
    #      = 3 * 3 * 0.01 + 1 * 0.01 = 0.10.
    assert loss[None] == pytest.approx(0.10, rel=1e-5)
    # d(loss)/dx[j] = 2 * trip * x[j]; j in 0..2 has trip=3, j=3 has trip=1, j>=4 never visited.
    for j in range(3):
        assert x.grad[j] == pytest.approx(2 * 3 * 0.1, rel=1e-5)
    assert x.grad[3] == pytest.approx(2 * 1 * 0.1, rel=1e-5)
    for j in range(4, N_X):
        assert x.grad[j] == pytest.approx(0.0, abs=1e-7)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_structural_pre_pass_fuses_sub_of_max_over_range_with_mismatched_shape_ends():
    # Check gradients when begin and end bounds are read from arrays with different shapes.
    N_X = 16

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(arr_a: qd.types.ndarray(dtype=qd.i32, ndim=1), arr_b: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for _dummy in range(1):
            accum = 0.0
            for i_a in range(arr_a.shape[0]):
                for i_b in range(arr_b.shape[0]):
                    n = arr_a[i_a] - arr_b[i_b]
                    if n > 0:
                        for k in range(n):
                            accum = accum + x[k] * x[k]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1
    arr_a_np = np.array([1, 5], dtype=np.int32)
    arr_b_np = np.array([4, 0], dtype=np.int32)

    compute(arr_a_np, arr_b_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_a_np, arr_b_np)
    qd.sync()

    # For (i_a, i_b) in {(0,0), (0,1), (1,0), (1,1)} the inner trip n = max(0, arr_a[i_a] - arr_b[i_b]) is {0, 1, 1, 5}.
    # Total push count is 7 (the (1,1) pair contributes 5); the pre-fusion adstack bound of 4 overflows. loss = (0 + 1 +
    # 1 + 5) * x[0..4]^2 = 0.07 with every x[i] = 0.1. x.grad[k] = 2 * (count of inner iterations that visit index k) *
    # 0.1. k = 0 is visited by every non-empty pair (3 visits), k in [1, 4] is visited only by the (1, 1) pair (1 visit
    # each), k >= 5 is never visited.
    assert loss[None] == pytest.approx(0.07, rel=1e-5)
    assert x.grad[0] == pytest.approx(0.6, rel=1e-5)
    for k in range(1, 5):
        assert x.grad[k] == pytest.approx(0.2, rel=1e-5)
    for k in range(5, N_X):
        assert x.grad[k] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    "arr_a_values, arr_b_values",
    [
        ([5, 5, 5, 5], [4, 0]),
        ([0, 0, 0, 7], [1, 1]),
    ],
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_fuses_sub_of_max_over_range_with_mismatched_lengths_is_safe(arr_a_values, arr_b_values):
    # Bounds from arrays with different lengths must preserve the intended inner iteration range.
    N_X = 16

    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(arr_a: qd.types.ndarray(dtype=qd.i32, ndim=1), arr_b: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for _dummy in range(1):
            accum = 0.0
            for i_a in range(arr_a.shape[0]):
                for i_b in range(arr_b.shape[0]):
                    n = arr_a[i_a] - arr_b[i_b]
                    if n > 0:
                        for k in range(n):
                            accum = accum + x[k] * x[k]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1
    arr_a_np = np.array(arr_a_values, dtype=np.int32)
    arr_b_np = np.array(arr_b_values, dtype=np.int32)

    compute(arr_a_np, arr_b_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_a_np, arr_b_np)
    qd.sync()

    expected_pushes = 0
    expected_visits = [0] * N_X
    for ia in range(len(arr_a_values)):
        for ib in range(len(arr_b_values)):
            n = max(0, int(arr_a_values[ia]) - int(arr_b_values[ib]))
            expected_pushes += n
            for k in range(min(n, N_X)):
                expected_visits[k] += 1
    assert loss[None] == pytest.approx(expected_pushes * 0.01, rel=1e-5)
    for k in range(N_X):
        if expected_visits[k] == 0:
            assert x.grad[k] == pytest.approx(0.0, abs=1e-7)
        else:
            assert x.grad[k] == pytest.approx(2 * expected_visits[k] * 0.1, rel=1e-5)


@pytest.mark.parametrize(
    "x_unused_val",
    [0.1, 100.0],
    ids=["uniform_x", "amplified_unused_x"],
)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_sub_of_max_over_range_fusion_does_not_mix_fieldload_and_extread(x_unused_val):
    # Field and external-array bounds must remain distinct even when their index expressions look alike.
    N = 4
    N_X = 16

    fld = qd.field(qd.i32, shape=(N,))
    x = qd.field(qd.f32, shape=(N_X,), needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(arr: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for _dummy in range(1):
            accum = 0.0
            for i in range(arr.shape[0]):
                n = fld[i] - arr[i]
                if n > 0:
                    for k in range(n):
                        accum = accum + x[k] * x[k]
            loss[None] += accum

    for i in range(N_X):
        x[i] = 0.1 if i < 8 else x_unused_val
    for i in range(N):
        fld[i] = 10
    arr_np = np.array([2, 2, 2, 2], dtype=np.int32)

    compute(arr_np)
    loss.grad[None] = 1.0
    for i in range(N_X):
        x.grad[i] = 0.0
    compute.grad(arr_np)
    qd.sync()

    # Each of the 4 outer iterations runs `fld[i] - arr[i] = 10 - 2 = 8` inner iters. Total pushes = 32.
    # x[0..7] is visited 4 times (once per outer iter); x[8..] is never visited, so the loss and gradients
    # are independent of `x_unused_val`.
    assert loss[None] == pytest.approx(4 * 8 * 0.01, rel=1e-5)
    for k in range(8):
        assert x.grad[k] == pytest.approx(4 * 2 * 0.1, rel=1e-5)
    for k in range(8, N_X):
        assert x.grad[k] == pytest.approx(0.0, abs=1e-7)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_spirv_metadata_per_task_buffer():
    # Keep cfg_optimization=False and two sibling tasks with different stack layouts. The second task
    # computes non-differentiable scratch values; counting must discard their unused recurrence while
    # preserving the stack counts and the first task's gradient.
    tri_mat = qd.field(dtype=qd.f32, shape=(2, 7, 7, 1))
    src_mat = qd.field(dtype=qd.types.matrix(3, 3, qd.f32), shape=(1, 1), needs_grad=True)
    dst_mat = qd.field(dtype=qd.types.matrix(3, 3, qd.f32), shape=(1, 1), needs_grad=True)
    state = qd.field(dtype=qd.types.vector(3, qd.f32), shape=(1, 1), needs_grad=True)
    group_offset = qd.field(dtype=qd.i32, shape=(1,))
    group_size = qd.field(dtype=qd.i32, shape=(1,))
    group_size[0] = 7

    @qd.kernel
    def kernel_two_offloads_with_tri_reduce():
        for i_0, i_b in qd.ndrange(state.shape[0], state.shape[1]):
            dst_mat[i_0, i_b] = src_mat[i_0, i_b]

        for i_g, i_b in qd.ndrange(state.shape[0], state.shape[1]):
            base = group_offset[i_g]
            for p_i0 in range(group_size[i_g]):
                for p_j0 in range(p_i0 + 1):
                    i_pr = base + p_i0
                    j_pr = base + p_j0
                    acc = qd.f32(0.0)
                    for p_k0 in range(p_j0):
                        k_pr = base + p_k0
                        acc = acc + (tri_mat[1, i_pr, k_pr, i_b] * tri_mat[1, j_pr, k_pr, i_b])
                    tri_mat[1, i_pr, j_pr, i_b] = acc

    seeds = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    dst_mat.grad.from_numpy(seeds)
    kernel_two_offloads_with_tri_reduce.grad()
    qd.sync()
    np.testing.assert_array_equal(src_mat.grad.to_numpy(), seeds)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [4, 32])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_linear_only_accumulator_with_nonlinear_operand(n_iter):
    # Cross-check `d/dx sum_i sum_j sin(x[i] + j*step)` against PyTorch autograd for the kernel shape
    # that mixes a loop-carried accumulator (`acc = acc + ...`) with a non-linear operand
    # (`sin(a)` where `a = x[i] + j*step`) inside an unrolled inner loop.
    #
    # Internal details: the operand `a` feeds `sin`, which is in `NonLinearOps::unary_collections`, so
    # `AdStackAllocaJudger::visit(UnaryOpStmt)` promotes its alloca to an adstack and the reverse pass
    # reads `a` back via `AdStackLoadTopStmt`. The accumulator `acc` only feeds linear `add`, never
    # reaches the non-linear / index / control-flow consumer visitors, but its adjoint chain still has
    # to weave through the adstack push / pop sites the operand introduces. `n_iter=32` keeps the
    # unrolled body wide enough that any miscalculation in the slot offset for the operand's stack -
    # e.g. an off-by-one in the inline `count - 1` saturation, a misaligned slot stride, or a missed
    # primal+adjoint zero-init on push - surfaces here as a wrong gradient long before it would surface
    # as an overflow at runtime.
    import torch

    n = 4
    step = 0.07
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for j in qd.static(range(n_iter)):
                a = x[i] + qd.cast(j, qd.f32) * step
                acc = acc + qd.sin(a)
            y[None] += acc

    for i in range(n):
        x[i] = 0.18 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.18 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch.float32)
        for j in range(n_iter):
            a_t = x_t[i] + float(j) * step
            acc_t = acc_t + torch.sin(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    assert y[None] == pytest.approx(y_t.item(), rel=1e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=1e-6)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_inner", [4, 32])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_repeated_load_top_of_outer_value_in_unrolled_inner_loop(n_inner):
    # Cross-check `d/dx sum_i sum_j cos(sin(x[i])) * (1 + j*0.01)` against PyTorch autograd for the
    # kernel shape where an outer-loop value (`a = sin(x[i])`) is consumed by a non-linear unary op
    # `cos(a)` repeatedly inside an unrolled inner loop, producing N consecutive `AdStackLoadTopStmt`
    # reads of the same stack in the reverse pass with no intervening Push / Pop / AccAdjoint.
    #
    # Internal details: each inner-loop iteration's reverse-pass adjoint formula reads `a`'s primal
    # (for `d/da cos(a) = -sin(a)`) and adjoint via the inline slot-pointer math. With `n_inner=32`
    # straight-line LoadTops on the same stack inside one block, the inline codegen emits 32
    # independent count-load + slot-GEP + value-load sequences. A regression that miscalculates the
    # saturating `count - 1` index under repeated reads, races the count alloca's mem2reg promotion
    # across the LoadTops, or short-circuits the slot offset math, surfaces here as a wrong gradient.
    import torch

    n = 3
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            a = qd.sin(x[i])
            acc = 0.0
            for j in qd.static(range(n_inner)):
                acc = acc + qd.cos(a) * (1.0 + qd.cast(j, qd.f32) * 0.01)
            y[None] += acc

    for i in range(n):
        x[i] = 0.21 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.21 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        a_t = torch.sin(x_t[i])
        acc_t = torch.zeros((), dtype=torch.float32)
        for j in range(n_inner):
            acc_t = acc_t + torch.cos(a_t) * (1.0 + float(j) * 0.01)
        y_t = y_t + acc_t
    y_t.backward()

    # Tolerance is `rel=5e-6` rather than the f32 floor `1e-6` because the kernel is unusually deep on the
    # numeric side: each contribution is `cos(sin(x[i])) * (1 + j*0.01)` and the test sums up to 32 of them per
    # outer iteration, so the per-sum drift accumulates to a few ULPs at the order of magnitude of the final
    # value across every backend that auto-promotes to FMA or drops guard bits (CUDA's NVPTX and Vulkan's
    # SPIR-V both observed). The test's purpose is to pin slot-pointer / count-recurrence correctness under
    # repeated LoadTop, not bit-exact float behavior; 5e-6 is the smallest tolerance that still passes on every
    # backend while staying inside the f32 noise band.
    assert y[None] == pytest.approx(y_t.item(), rel=5e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=5e-6)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [4, 16])
@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], require=qd.extension.adstack)
def test_adstack_f16_unrolled_pushes_alignment(n_iter):
    # Cross-check `d/dx sum_i sum_j sin(x[i] + j*step)` against PyTorch autograd for an `f16` field whose
    # operand `a = x[i] + j*step` adstack-promotes through `sin`. Each adstack slot is `2 * element_size`
    # = 4 bytes wide, so slot N starts at byte offset `8 + 4*N` from the per-thread slab base. Slot 0 is
    # 8-aligned but every odd slot index (1, 3, 5, ...) lands on a 4-byte-aligned address that is NOT
    # 8-aligned.
    #
    # Internal details: `AdStackPushStmt`'s inline IR emits a `llvm.memset` zeroing the primal+adjoint
    # slot pair before storing the pushed value. The destination alignment passed to `CreateMemSet` must
    # reflect the actual slot pointer alignment - `min(8, 2 * element_size)` = 4 bytes for f16. An
    # over-stated 8-byte alignment lets NVPTX / AMDGCN lowering pick a wider store (`st.b64`) than the
    # 4-aligned slot pointer can satisfy, which traps as a misaligned-address fault on stricter GPU
    # backends or silently corrupts adjoint state on tolerant ones - either way producing wrong
    # gradients. `n_iter=16` ensures multiple non-zero slot offsets get exercised, including odd
    # slot indices where the alignment claim matters most. Tolerance `rel=4e-3` is the f16 precision
    # floor for sums of ~16 `sin` values. SPIR-V backends (Metal / MoltenVK / Vulkan) are excluded
    # from the arch list because their MSL / Vulkan shader compilation fails on the reverse-mode
    # `y[None] += acc` adstack shape with `qd.f16` fields - the host-side type-check warns
    # `Atomic add may lose precision: f16 <- f32` and the underlying compute pipeline build then
    # rejects the shader, so the alignment regression this test pins on the LLVM CUDA / AMDGPU
    # backends cannot run on the SPIR-V backends regardless of slot alignment correctness.
    import torch

    n = 4
    step = 0.05
    x = qd.field(qd.f16, shape=n, needs_grad=True)
    y = qd.field(qd.f16, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for j in qd.static(range(n_iter)):
                a = x[i] + qd.cast(j, qd.f16) * qd.f16(step)
                acc = acc + qd.sin(a)
            y[None] += acc

    for i in range(n):
        x[i] = 0.21 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.21 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch.float32)
        for j in range(n_iter):
            a_t = x_t[i] + float(j) * step
            acc_t = acc_t + torch.sin(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    assert float(y[None]) == pytest.approx(y_t.item(), rel=4e-3)
    for i in range(n):
        assert float(x.grad[i]) == pytest.approx(x_t.grad[i].item(), rel=4e-3)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [4, 16, 64])
@pytest.mark.parametrize("n_stacks", [1, 3])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_unrolled_many_pushes_across_multiple_stacks(n_iter, n_stacks):
    # Cross-check `d/dx sum_i sum_s sum_j sin(x[i] + j*step + s*offset)` against PyTorch autograd for the
    # kernel shape where `n_stacks` parallel adstacks each receive `n_iter` straight-line pushes inside an
    # unrolled inner loop, fanning out to a separate non-linear `sin` per stack per iteration.
    #
    # Internal details: this is the regime the LLVM release-build inline AdStack codegen targets - each `s`
    # promotes its operand `a = x[i] + j*step + s*offset` onto its own per-stack `alloca i64` count, and
    # the unrolled inner loop body emits `n_iter` consecutive `count++` increments per stack. After mem2reg
    # lifts the alloca to SSA and GVN folds the increment chain to constants `0, 1, ..., n_iter - 1`, the
    # only memory ops left in the unrolled body are the slot stores. `n_iter=64` is well above the adstack
    # capacity floor of 32, so any regression that miscalculates the slot offset under multiple sibling
    # stacks (e.g. by hoisting the count alloca shared across stacks instead of per-stack), that
    # short-circuits / silently drops pushes via a wrong saturating subtract on count, or that races a
    # cross-stack increment, surfaces here as a wrong gradient long before it would surface as an overflow
    # at runtime. The cross-check tolerance `rel=1e-6` is the f32 floor; a few-ULP drift per `sin` over up
    # to `n_stacks * n_iter <= 192` summed terms still fits within it.
    import torch

    n = 4
    step = 0.07
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for s in qd.static(range(n_stacks)):
                s_offset = qd.cast(s, qd.f32) * 0.13
                for j in qd.static(range(n_iter)):
                    a = x[i] + qd.cast(j, qd.f32) * step + s_offset
                    acc += qd.sin(a)
            y[None] += acc

    for i in range(n):
        x[i] = 0.21 + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.21 + i * 0.05 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch.float32)
        for s in range(n_stacks):
            s_offset_t = float(s) * 0.13
            for j in range(n_iter):
                a_t = x_t[i] + float(j) * step + s_offset_t
                acc_t = acc_t + torch.sin(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    assert y[None] == pytest.approx(y_t.item(), rel=1e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=1e-6)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_inner", [4, 32])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_min_loop_carried_serial_range_for(n_inner):
    # Cross-check `d/dx sum_i acc_n` where `acc` is initialized to 1.0 and updated per iteration of a serial
    # `for j in range(n_inner)` body via `acc = qd.min(acc * 0.5 + 0.05, x[i] + j*0.05)`. The min winner flips
    # between the lhs and rhs across iterations (rhs wins on iter 0 because acc starts at 1.0 vs x[i]=~0.3,
    # then lhs wins on later iterations as acc shrinks). The reverse pass must use the per-iteration forward
    # lhs/rhs to route the gradient correctly.
    #
    # Internal details: pins the snap-stack fix in `MakeAdjoint::visit(BinaryOpStmt)`'s min/max branch. The
    # forward cmp `lhs < rhs` (min) / `rhs < lhs` (max) is computed at forward time and pushed onto a
    # dedicated 1-push-per-bin-execution adstack, then read back in reverse with a matching pop. Without the
    # snap-stack, BackupSSA spills `bin->lhs` / `bin->rhs` to single overwrite-each-iteration allocas and
    # every reverse iteration reads the last forward iteration's values, so the cmp flips on iterations where
    # the actual winner changed and the gradient routes through the wrong branch (visible as `x.grad=0`
    # instead of the analytical `0.125` for `x[i]=0.31`, `n_inner=4`).
    import torch

    n = 4
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 1.0
            for j in range(n_inner):
                acc = qd.min(acc * 0.5 + 0.05, x[i] + qd.cast(j, qd.f32) * 0.05)
            y[None] += acc

    for i in range(n):
        x[i] = 0.31 + i * 0.07
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([0.31 + i * 0.07 for i in range(n)], dtype=torch.float32, requires_grad=True)
    y_t = torch.zeros((), dtype=torch.float32)
    for i in range(n):
        acc_t = torch.tensor(1.0, dtype=torch.float32)
        for j in range(n_inner):
            acc_t = torch.minimum(acc_t * 0.5 + 0.05, x_t[i] + float(j) * 0.05)
        y_t = y_t + acc_t
    y_t.backward()

    assert y[None] == pytest.approx(y_t.item(), rel=1e-6)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=1e-4)


@pytest.mark.parametrize("gated_fraction", [0.0, 0.05, 0.5, 1.0])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_ndarray_gate_grad_correct(gated_fraction):
    # Asserts gradient correctness for reverse-mode kernels of shape `for i in range(n): if selector[i] > eps:
    # <adstack-using gradient work>` where `selector` is an ndarray argument.
    n = 256
    n_iter = 8
    eps = 1e-9

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    np.random.seed(0)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = int(round(gated_fraction * n))
    selector_np = np.zeros(n, dtype=np.float32)
    if n_gated > 0:
        gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
        selector_np[gated_indices] = 1.0
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"static-bound-expr grad returned NaN: {got_grad}"

    # Analytic oracle. For gated i, the inner recurrence `v = v*c + d` over `n_iter` steps is linear in v with slope
    # `c^n_iter`, where `c = 1.05`. So `d(out[0])/d(x[i]) = c^n_iter` for gated i, 0 otherwise. `gated_fraction == 0` is
    # the per-task-reducer-count-zero edge case: every dispatched thread misses the gate, the reducer publishes capacity
    # = 0, the codegen-emitted clamp at the LCA-block claim site has to keep the row id at 0 (a naive `capacity - 1`
    # underflow to UINT32_MAX leaves the clamp inert and a divergent over-claim writes past the float-heap end).
    # Float-heap allocation is floored at one row precisely so the single-row fallback is always backed by real storage.
    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > eps, np.float32(expected_per_gated), np.float32(0.0))
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=[qd.extension.adstack, qd.extension.data64])
def test_adstack_static_bound_expr_f64_gate_grad_correct():
    # Asserts gradient correctness for reverse-mode kernels with an f64-typed gating ndarray (`if selector_f64[i] >
    # 0.5`) above f32 adstack pushes.
    n = 256
    n_iter = 8
    threshold = 0.5

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f64, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > threshold:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    np.random.seed(0)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = np.full(n, 0.25, dtype=np.float64)
    gated_indices = np.sort(np.random.choice(n, size=n // 2, replace=False))
    selector_np[gated_indices] = 1.0

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"f64-gate static-bound-expr grad returned NaN: {got_grad}"

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > threshold, np.float32(expected_per_gated), np.float32(0.0))
    for i in range(n):
        assert got_grad[i] == pytest.approx(expected[i], rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("alloca_outside_gate", [False, True])
@test_utils.test(require=qd.extension.adstack, debug=True)
def test_adstack_static_bound_expr_ndarray_gate_debug_build_grad_correct(alloca_outside_gate):
    # Asserts gradient correctness for reverse-mode kernels with a captured ndarray-backed gate under `debug=True`.
    n = 256
    n_iter = 8
    eps = 1e-9
    gated_fraction = 0.5

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))

    if alloca_outside_gate:

        @qd.kernel
        def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
            for i in range(n):
                v = qd.cast(0.0, qd.f32)
                if selector[i] > eps:
                    v = x[i]
                    for _ in range(n_iter):
                        v = v * 1.05 + 0.05
                out[0] += v

    else:

        @qd.kernel
        def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
            for i in range(n):
                if selector[i] > eps:
                    v = x[i]
                    for _ in range(n_iter):
                        v = v * 1.05 + 0.05
                    out[0] += v

    np.random.seed(2)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = max(1, int(round(gated_fraction * n)))
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"debug-build static-bound-expr grad returned NaN: {got_grad}"

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > eps, np.float32(expected_per_gated), np.float32(0.0))
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("gated_fraction", [0.05, 0.5, 1.0])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_grad_correct(gated_fraction):
    # Asserts gradient correctness for reverse-mode kernels of shape `for i in selector: if selector[i] > eps:
    # <adstack-using gradient work>` where `selector` is a `qd.field(...)` placed under `qd.root.dense(...)` -the
    # layout  most sparse-grid workloads use.
    n = 256
    n_iter = 8
    eps = 1e-9

    selector = qd.field(qd.f32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in selector:
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(1)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = max(1, int(round(gated_fraction * n)))
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    for i in range(n):
        x[i] = float(x_np[i])
        selector[i] = float(selector_np[i])
    out[None] = 0.0
    out.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0

    compute()
    compute.grad()
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.where(selector_np > eps, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = np.array([x.grad[i] for i in range(n)], dtype=np.float32)
    assert not np.isnan(got_grad).any(), f"static-bound-expr snode grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_ndarray_gate_compound_index_grad_correct():
    # Pins gradient correctness when an ndarray-backed gating array is indexed by a compound expression
    # (`selector[i % K]` with K < n).
    n = 256
    K = 64
    n_iter = 8
    eps = 1e-9

    selector = qd.ndarray(qd.f32, shape=(K,))
    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i % K] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    np.random.seed(3)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = (np.random.rand(K) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    gated_per_iter = selector_np[np.arange(n) % K] > eps
    expected = np.where(gated_per_iter, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"compound-index ndarray grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "gate_shape",
    [
        "compound_mod",
        "affine_div",
        "constant_index",
        "dynamic_load_index",
        "folding_two_axis_decomp",
    ],
)
@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal], require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_non_bijective_index_grad_correct(gate_shape):
    # Pins gradient correctness on parallel-dispatched backends for `qd.field`-backed gates whose index expression
    # is not a per-iteration bijection with the SNode cell space.
    n = 256
    K = 64
    n_iter = 8
    eps = 1e-9

    affine_div_size = n  # snode size = n keeps `loop_iter <= snode_iter_count`, so the iter-count check passes
    fold_axis = 8  # `folding_two_axis_decomp` joint space is `fold_axis * fold_axis = 64 < n = 256`
    if gate_shape == "affine_div":
        selector_shape = (affine_div_size,)
    elif gate_shape == "folding_two_axis_decomp":
        selector_shape = (fold_axis, fold_axis)
    else:
        selector_shape = (K,)
    selector = qd.field(qd.f32, shape=selector_shape)
    idx_field = qd.field(qd.i32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            gate_idx = (
                (i % K,)
                if qd.static(gate_shape == "compound_mod")
                else (
                    (i // 2,)
                    if qd.static(gate_shape == "affine_div")
                    else (
                        (K // 2,)
                        if qd.static(gate_shape == "constant_index")
                        else (
                            (i % fold_axis, (i // fold_axis) % fold_axis)
                            if qd.static(gate_shape == "folding_two_axis_decomp")
                            else (idx_field[i],)
                        )
                    )
                )
            )
            if selector[gate_idx] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(2)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = (np.random.rand(*selector_shape) < 0.3).astype(np.float32)
    idx_field_np = (np.arange(n) % K).astype(np.int32)
    selector.from_numpy(selector_np)
    for i in range(n):
        x[i] = float(x_np[i])
        idx_field[i] = int(idx_field_np[i])
    out[None] = 0.0
    out.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0

    compute()
    compute.grad()
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    if gate_shape == "compound_mod":
        gated_per_iter = selector_np[np.arange(n) % K] > eps
    elif gate_shape == "affine_div":
        gated_per_iter = selector_np[np.arange(n) // 2] > eps
    elif gate_shape == "constant_index":
        gated_per_iter = np.full(n, bool(selector_np[K // 2] > eps))
    elif gate_shape == "folding_two_axis_decomp":
        idx = np.arange(n)
        gated_per_iter = selector_np[idx % fold_axis, (idx // fold_axis) % fold_axis] > eps
    else:
        gated_per_iter = selector_np[idx_field_np] > eps
    expected = np.where(gated_per_iter, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = np.array([x.grad[i] for i in range(n)], dtype=np.float32)
    assert not np.isnan(got_grad).any(), f"non-bijective-index snode grad returned NaN ({gate_shape}): {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_bijective_decomposed_index_grad_correct():
    # Bijective multi-axis gate decomposed from a single flat loop variable: `for i in range(n): if  selector[i //
    # K, i % K] > eps:`.
    K = 8
    n_rows = 32
    n = n_rows * K
    n_iter = 8
    eps = 1e-9

    selector = qd.field(qd.f32, shape=(n_rows, K))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            if selector[i // K, i % K] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(11)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = (np.random.rand(n_rows, K) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    idx = np.arange(n)
    gated_per_iter = selector_np[idx // K, idx % K] > eps
    expected = np.where(gated_per_iter, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"bijective decomposed-index snode grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_bijective_linear_range_grad_correct():
    # Bijective single-axis gate: `for i in range(n): if field[i] > eps:`.
    n = 256
    n_iter = 8
    eps = 1e-9
    field = qd.field(qd.f32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            if field[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    field_np = (np.random.rand(n) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    field.from_numpy(field_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.where(field_np > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"linear-range bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_bijective_multi_axis_structfor_grad_correct():
    # Bijective multi-axis StructFor gate: `for I, J, K in field3d: if field3d[I, J, K] > eps:`.
    n = 16
    n_iter = 8
    eps = 1e-9
    field3d = qd.field(qd.f32, shape=(n, n, n))
    x = qd.field(qd.f32, shape=(n, n, n), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for I, J, K in field3d:
            if field3d[I, J, K] > eps:
                v = x[I, J, K]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n**3).reshape(n, n, n)).astype(np.float32)
    field_np = (np.random.rand(n, n, n) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    field3d.from_numpy(field_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.where(field_np > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"multi-axis StructFor bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_bijective_multi_axis_ndrange_grad_correct():
    # Bijective multi-axis ndrange gate: `for ii, jj, kk in qd.ndrange(n, n, n): if grid[ii, jj, kk] > eps:`.
    n = 16
    n_iter = 8
    eps = 1e-9
    grid = qd.field(qd.f32, shape=(n, n, n))
    x = qd.field(qd.f32, shape=(n, n, n), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for ii, jj, kk in qd.ndrange(n, n, n):
            if grid[ii, jj, kk] > eps:
                v = x[ii, jj, kk]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n**3).reshape(n, n, n)).astype(np.float32)
    grid_np = (np.random.rand(n, n, n) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    grid.from_numpy(grid_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.where(grid_np > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"multi-axis ndrange bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_bijective_slice_with_iter_grad_correct():
    # Bijective gate with one iterating axis plus a constant slice: `for i in range(n): if grid[i, 0] > eps:`.
    n = 256
    cols = 4
    n_iter = 8
    eps = 1e-9
    grid = qd.field(qd.f32, shape=(n, cols))
    x = qd.field(qd.f32, shape=(n, cols), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in range(n):
            if grid[i, 0] > eps:
                v = x[i, 0]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(7)
    x_np = (0.1 + 0.001 * np.arange(n * cols).reshape(n, cols)).astype(np.float32)
    grid_np = (np.random.rand(n, cols) < 0.3).astype(np.float32)
    x.from_numpy(x_np)
    grid.from_numpy(grid_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))
    compute()
    compute.grad()
    qd.sync()
    expected = np.zeros_like(x_np)
    expected[:, 0] = np.where(grid_np[:, 0] > eps, np.float32(1.05**n_iter), np.float32(0.0))
    got = x.grad.to_numpy()
    assert not np.isnan(got).any(), f"slice-with-iter bijective grad returned NaN: {got}"
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_snode_gate_multi_axis_ndrange_with_arg_index_grad_correct():
    # Pins SNode-arm bound-expr capture for the canonical sparse-grid kernel shape: `for ii, jj, kk, ib in
    # qd.ndrange(...): if grid[f, ii, jj, kk, ib] > eps:`.
    F = 2
    n = 4
    B = 1
    n_iter = 4
    eps = 1e-9

    grid = qd.field(qd.f32, shape=(F, n, n, n, B))
    x = qd.field(qd.f32, shape=(F, n, n, n, B), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(f: qd.i32) -> None:
        for ii, jj, kk, ib in qd.ndrange(n, n, n, B):
            if grid[f, ii, jj, kk, ib] > eps:
                v = x[f, ii, jj, kk, ib]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[None] += v

    np.random.seed(3)
    x_np = (0.1 + 0.001 * np.arange(F * n * n * n * B).reshape(F, n, n, n, B)).astype(np.float32)
    grid_np = (np.random.rand(F, n, n, n, B) < 0.4).astype(np.float32)
    grid.from_numpy(grid_np)
    x.from_numpy(x_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros_like(x_np))

    f_active = 0
    compute(f_active)
    compute.grad(f_active)
    qd.sync()

    coeff = 1.05
    expected_per_gated = coeff**n_iter
    expected = np.zeros_like(x_np)
    expected[f_active] = np.where(grid_np[f_active] > eps, np.float32(expected_per_gated), np.float32(0.0))
    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"multi-axis ndrange grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack, debug=False)
def test_adstack_static_bound_expr_snode_gate_primal_dependent_grad_correct():
    # Asserts gradient correctness on the LLVM CPU host reducer for SNode-backed gates with a primal-dependent inner
    # recurrence.
    n = 4096
    n_iter = 8
    eps = 1e-9

    selector = qd.field(qd.f32, shape=(n,))
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute() -> None:
        for i in selector:
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * v + 0.05
                out[None] += v

    np.random.seed(1)
    x_np = (0.001 * np.ones(n)).astype(np.float32)
    n_gated = max(1, n // 2)
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out[None] = 0.0
    out.grad[None] = 1.0
    x.grad.from_numpy(np.zeros(n, dtype=np.float32))

    compute()
    compute.grad()
    qd.sync()

    expected = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if selector_np[i] <= eps:
            continue
        v = float(x_np[i])
        primals = [v]
        for _ in range(n_iter):
            v = v * v + 0.05
            primals.append(v)
        d = 1.0
        for k in range(n_iter):
            d = d * (2.0 * primals[n_iter - 1 - k])
        expected[i] = np.float32(d)

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any()
    assert not np.isinf(got_grad).any()
    for i in range(n):
        assert got_grad[i] == pytest.approx(expected[i], rel=1e-5, abs=1e-7)


@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], debug=False)
def test_adstack_static_bound_expr_snode_gate_multileaf_dense_grad_correct():
    # Asserts gradient correctness on the LLVM static-bound-expr SNode resolver for dense parents with multiple
    # mixed-size leaves.
    n = 256
    n_iter = 6
    eps = 1e-9

    field_f64 = qd.field(qd.f64)
    field_f32 = qd.field(qd.f32)
    x = qd.field(qd.f32, shape=(n,), needs_grad=True)
    out = qd.field(qd.f32, shape=(), needs_grad=True)
    qd.root.dense(qd.i, n).place(field_f64, field_f32)

    @qd.kernel
    def compute() -> None:
        for i in field_f32:
            if field_f32[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = v * v + 0.05
                out[None] += v

    np.random.seed(1)
    # `x` varies with `i` so any cross-row aliasing under a mis-sized adstack heap surfaces as a gradient mismatch (the
    # reverse pop reads back a different thread's primal). A constant `x` would mask aliasing because every gated thread
    # pushes the same primal sequence and the pop comes back identical.
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    n_gated = max(1, n // 2)
    selector_np = np.zeros(n, dtype=np.float32)
    gated_indices = np.sort(np.random.choice(n, size=n_gated, replace=False))
    selector_np[gated_indices] = 1.0
    for i in range(n):
        x[i] = float(x_np[i])
        field_f32[i] = float(selector_np[i])
        field_f64[i] = 1.0
    out[None] = 0.0
    out.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0

    compute()
    compute.grad()
    qd.sync()

    expected = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if selector_np[i] <= eps:
            continue
        v = float(x_np[i])
        primals = [v]
        for _ in range(n_iter):
            v = v * v + 0.05
            primals.append(v)
        d = 1.0
        for k in range(n_iter):
            d = d * (2.0 * primals[n_iter - 1 - k])
        expected[i] = np.float32(d)

    got_grad = np.array([x.grad[i] for i in range(n)], dtype=np.float32)
    assert not np.isnan(got_grad).any()
    assert not np.isinf(got_grad).any()
    for i in range(n):
        assert got_grad[i] == pytest.approx(expected[i], rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("bound_shape", ["int_const", "scalar_field", "ndarray_shape", "ndarray_read", "two_arg_range"])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_memory_savings_runs_clean(bound_shape):
    # Check per-element gradients for each supported loop-bound shape under a runtime gate.
    n = 256
    n_iter = 16
    eps = 1e-9

    np.random.seed(0)
    x_np = (0.1 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = np.zeros(n, dtype=np.float32)
    selector_np[: max(1, int(round(0.5 * n)))] = 1.0
    np.random.shuffle(selector_np)

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))
    bound_arr = qd.ndarray(qd.i32, shape=(n,))
    n_field = qd.field(qd.i32, shape=())
    start_arr = qd.ndarray(qd.i32, shape=(1,))
    stop_arr = qd.ndarray(qd.i32, shape=(1,))
    n_field[None] = n
    bound_arr.from_numpy(np.full(n, n, dtype=np.int32))
    start_arr.from_numpy(np.array([0], dtype=np.int32))
    stop_arr.from_numpy(np.array([n], dtype=np.int32))

    @qd.kernel
    def compute(
        x: qd.types.NDArray,
        selector: qd.types.NDArray,
        out: qd.types.NDArray,
        bound_arr: qd.types.NDArray,
        start_arr: qd.types.NDArray,
        stop_arr: qd.types.NDArray,
    ) -> None:
        # `qd.static(bound_shape == ...)` evaluates the comparison at kernel-compile time (`bound_shape` is a Python
        # closure constant), so the AST that reaches the codegen has only one of the five `range` forms surviving - no
        # helper has to materialise per parametrisation.
        for i in (
            range(n)
            if qd.static(bound_shape == "int_const")
            else (
                range(n_field[None])
                if qd.static(bound_shape == "scalar_field")
                else (
                    range(selector.shape[0])
                    if qd.static(bound_shape == "ndarray_shape")
                    else (
                        range(bound_arr[0])
                        if qd.static(bound_shape == "ndarray_read")
                        else range(start_arr[0], stop_arr[0])
                    )
                )
            )
        ):
            if selector[i] > eps:
                v = x[i] * x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out, bound_arr, start_arr, stop_arr)
    compute.grad(x, selector, out, bound_arr, start_arr, stop_arr)
    qd.sync()

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"sparse-adstack-heap [{bound_shape}] grad returned NaN: {got_grad}"
    coeff = 1.05
    # `v = x[i] * x[i]` then `v = v * 1.05 + 0.05` repeated n_iter times. v_final = x[i]^2 * c^n + S where S is a
    # constant. d(v_final)/d(x[i]) = 2 * x[i] * c^n. Gated only.
    expected = np.where(selector_np > eps, np.float32(2.0 * x_np * coeff**n_iter), np.float32(0.0))
    np.testing.assert_allclose(got_grad, expected, rtol=1e-4, atol=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_primal_dependent_inner_recurrence_grad_correct():
    # Asserts gradient correctness for reverse-mode kernels with a captured ndarray-backed gate above a primal-
    # dependent  inner recurrence (`v = qd.sin(v) + 0.01`, whose chain rule `d(sin(v))/dv = cos(v)` depends on the
    # stored primal).
    n = 512
    n_iter = 4
    eps = 1e-9

    np.random.seed(0)
    x_np = (0.05 + 0.001 * np.arange(n)).astype(np.float32)
    selector_np = np.ones(n, dtype=np.float32)

    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f32, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = qd.sin(v) + 0.01
                out[0] += v

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    # numpy reference: chain rule for `v_k = sin(v_{k-1}) + 0.01` is `cos(v_{k-1})`. d(v_n)/d(x[i]) is the product of
    # `cos(v_k)` for k = 0..n_iter-1, where the v_k sequence is generated forward from x[i].
    v_np = x_np.copy()
    grad_np = np.ones(n, dtype=np.float64)
    for _ in range(n_iter):
        grad_np *= np.cos(v_np.astype(np.float64))
        v_np = np.sin(v_np) + np.float32(0.01)
    expected = grad_np.astype(np.float32)

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"primal-dependent inner-recurrence grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, expected, rtol=2e-4, atol=2e-6)


@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_static_bound_expr_non_loop_var_index_falls_back_to_worst_case():
    # A gate indexed by a non-loop variable must preserve distinct f64 gradients for active elements.
    n = 64
    K = 4
    n_iter = 8
    eps = 1e-12

    np.random.seed(0)
    # Spread `x` widely across the f64 representable range so per-`i` `cos(x[i])` differs by O(0.1) between adjacent
    # indices; under f64 precision the multi-thread CPU race produces a clearly observable drift in the per-`i`
    # chain-rule product when the gate-capture pretends `selector[i % K]` is loop-index-shaped.
    x_np = (0.5 + 0.05 * np.arange(n)).astype(np.float64)
    selector_np = np.zeros(n, dtype=np.float64)
    selector_np[:K] = 1.0  # first K cells gated; rest zero

    x = qd.ndarray(qd.f64, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f64, shape=(1,), needs_grad=True)
    selector = qd.ndarray(qd.f64, shape=(n,))

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i % K] > eps:
                v = x[i]
                for _ in range(n_iter):
                    v = qd.sin(v) + 0.01
                out[0] += v

    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float64))
    out.grad.from_numpy(np.ones((1,), dtype=np.float64))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    # `v = sin(v) + c` has a primal-dependent chain rule `cos(v_{k-1})`. Each iteration's reverse pass multiplies
    # adjoints by `cos(stored_primal)`, so a slot read corrupted by a different iteration's push produces a
    # primal-dependent wrong factor. With selector[:K] = 1.0 every iteration is gated; numpy reference computes the
    # chain forward then products `cos(v_k)` for k = 0..n_iter-1.
    v_np = x_np.copy()
    grad_np = np.ones(n, dtype=np.float64)
    for _ in range(n_iter):
        grad_np *= np.cos(v_np)
        v_np = np.sin(v_np) + 0.01

    got_grad = x.grad.to_numpy()
    assert not np.isnan(got_grad).any(), f"non-loop-var-index grad returned NaN: {got_grad}"
    np.testing.assert_allclose(got_grad, grad_np, rtol=1e-12, atol=1e-14)


@test_utils.test(arch=[qd.cuda, qd.amdgpu], require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_gpu_dispatch_cap_uses_floor_division():
    # Use 65700 elements and block_dim=192 to exercise a partial final block and grid-stride reuse.
    # The f64 analytical oracle distinguishes gradients across threads after six nonlinear iterations.
    n = 65700
    block_dim = 192
    n_inner = 6

    x_np = (0.5 + 0.001 * np.arange(n)).astype(np.float64)

    x = qd.ndarray(qd.f64, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f64, shape=(1,), needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.NDArray, out: qd.types.NDArray) -> None:
        qd.loop_config(block_dim=block_dim)
        for i in range(n):
            v = x[i]
            for _ in range(n_inner):
                v = qd.sin(v) + 0.01
            out[0] += v

    x.from_numpy(x_np)
    out.from_numpy(np.zeros((1,), dtype=np.float64))
    out.grad.from_numpy(np.ones((1,), dtype=np.float64))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, out)
    compute.grad(x, out)
    qd.sync()

    v_np = x_np.copy()
    grad_ref = np.ones(n, dtype=np.float64)
    for _ in range(n_inner):
        grad_ref *= np.cos(v_np)
        v_np = np.sin(v_np) + 0.01

    got_grad = x.grad.to_numpy()
    np.testing.assert_allclose(got_grad, grad_ref, rtol=1e-12, atol=1e-14)


@test_utils.test(require=qd.extension.adstack, debug=False)
def test_adstack_static_bound_expr_device_sizer_per_kind_offsets_grad_correct():
    # Mixed floating-point and integer stacks must have independent offsets and correct gradients.
    n_outer = 8
    a_np = np.array([2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int32)

    x = qd.field(qd.f32, shape=(n_outer,), needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute(a: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in x:
            v0 = x[i] * 1.0
            j = 0
            v1 = x[i] * 2.0
            n = a[i]
            for _ in range(n):
                v0 = v0 * 0.95 + 0.01
                j = j + 1
                v1 = v1 * 0.9 + 0.02
            y[None] += v0 + v1 + qd.cast(j, qd.f32) * 0.0

    for i in range(n_outer):
        x[i] = 0.1 + 0.05 * i

    compute(a_np)
    y.grad[None] = 1.0
    for i in range(n_outer):
        x.grad[i] = 0.0
    compute.grad(a_np)
    qd.sync()

    for i in range(n_outer):
        # d(v0_n + v1_n) / dx[i] = 1.0 * 0.95**a[i] + 2.0 * 0.9**a[i].
        expected = 1.0 * (0.95 ** int(a_np[i])) + 2.0 * (0.9 ** int(a_np[i]))
        assert x.grad[i] == pytest.approx(expected, rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_static_bound_expr_resolve_length_walks_full_ndarray():
    # A loop bound stored near the end of an ndarray must contribute to the measured depth.
    n_gated_past_cap = 64  # enough to alias multiple iterations into a single row if the heap mis-sizes to one row
    advisory_cap = 131072  # SPIR-V kMaxNumThreadsGridStrideLoop
    n = advisory_cap + n_gated_past_cap
    n_iter = 4

    selector = qd.ndarray(qd.f32, shape=(n,))
    x = qd.ndarray(qd.f32, shape=(n,), needs_grad=True)
    out = qd.ndarray(qd.f32, shape=(1,), needs_grad=True)

    @qd.kernel
    def compute(x: qd.types.NDArray, selector: qd.types.NDArray, out: qd.types.NDArray) -> None:
        for i in range(n):
            if selector[i] > 1e-9:
                v = x[i]
                for _ in range(n_iter):
                    v = v * 1.05 + 0.05
                out[0] += v

    x_np = (0.001 * np.arange(n) + 0.1).astype(np.float32)
    selector_np = np.zeros(n, dtype=np.float32)
    selector_np[advisory_cap : advisory_cap + n_gated_past_cap] = 1.0  # all gated cells past the advisory cap
    x.from_numpy(x_np)
    selector.from_numpy(selector_np)
    out.from_numpy(np.zeros((1,), dtype=np.float32))
    out.grad.from_numpy(np.ones((1,), dtype=np.float32))
    x.grad.from_numpy(np.zeros_like(x_np))

    compute(x, selector, out)
    compute.grad(x, selector, out)
    qd.sync()

    got = x.grad.to_numpy()
    expected_per_gated = np.float32(1.05**n_iter)
    expected = np.where(selector_np > 1e-9, expected_per_gated, np.float32(0.0)).astype(np.float32)
    assert not np.isnan(got).any(), f"resolve_length grad returned NaN: {got[advisory_cap:advisory_cap + 8]}"
    for i in range(advisory_cap, advisory_cap + n_gated_past_cap):
        assert got[i] == pytest.approx(expected[i], rel=1e-5, abs=1e-7), (
            f"gated index {i} (past advisory_total_num_threads={advisory_cap}) gradient diverged: "
            f"got={got[i]} expected={expected[i]}"
        )


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_scalarized_tensor_component_loop_trip_grad_correct():
    # Every component of a loop-carried vector must preserve its history after stack scalarization.
    n_iter_val = 3
    denom_val = 2.0
    x_np = np.array([0.3, 0.7], dtype=np.float32)
    n_env = x_np.size

    n_iter = qd.field(qd.i32, shape=())
    denom = qd.field(qd.f32, shape=())
    x = qd.field(qd.f32, shape=n_env, needs_grad=True)
    out = qd.field(qd.f32, shape=n_env, needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n_env):
            q = qd.Vector([x[i], x[i] * 0.5], dt=qd.f32)
            s = denom[None]
            for _ in range(n_iter[None]):
                q = qd.Vector([q[0] + q[1] * 0.1, q[1] + q[0] * 0.2], dt=qd.f32)
                q = q / s
            out[i] = q[0] + q[1]

    n_iter[None] = n_iter_val
    denom[None] = denom_val
    x.from_numpy(x_np)
    compute()
    out.grad.from_numpy(np.ones(n_env, dtype=np.float32))
    x.grad.fill(0.0)
    compute.grad()

    # The recurrence q <- (M @ q) / s with M = [[1, 0.1], [0.2, 1]] is linear, so d(out)/d(x[i]) is the same for
    # every env: sum of the columns of (M / s)^n_iter applied to the initial jacobian (1, 0.5).
    mat = np.array([[1.0, 0.1], [0.2, 1.0]], dtype=np.float64) / denom_val
    jac = np.array([1.0, 0.5], dtype=np.float64)
    for _ in range(n_iter_val):
        jac = mat @ jac
    expected = float(jac.sum())
    grad = x.grad.to_numpy()
    for i in range(n_env):
        assert grad[i] == pytest.approx(expected, rel=1e-5)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_loop_carried_vector_component_grad_correct():
    # A multi-component `qd.Vector` carried across a field-bounded loop and rebuilt from its own previous components
    # each iteration must have its reverse-mode gradient read each iteration's component value, not the last one. The
    # loop-carried tensor local is read component-wise through a `MatrixPtr`, so before the fix the adstack judge never
    # saw a load of it and left it a single overwrite-each-iteration slot; the reverse then read the final iteration's
    # component for every reverse step, scaling the second-order term of the gradient by a spurious extra factor. The
    # scalar-valued analogue was already correct, so this guards specifically the tensor component read off the stack.
    n_env = 2
    x_np = np.array([0.3, 0.7], dtype=np.float32)

    n_iter = qd.field(qd.i32, shape=())
    x = qd.field(qd.f32, shape=n_env, needs_grad=True)
    out = qd.field(qd.f32, shape=n_env, needs_grad=True)

    @qd.kernel
    def compute():
        for i_env in range(n_env):
            q = qd.Vector([x[i_env], x[i_env] * 0.5], dt=qd.f32)
            for _ in range(n_iter[None]):
                q = qd.Vector([q[0] * q[0], q[1] + q[0] * 0.1], dt=qd.f32)
            out[i_env] = q[1]

    n_iter[None] = 2
    x.from_numpy(x_np)
    compute()
    out.grad.from_numpy(np.ones(n_env, dtype=np.float32))
    x.grad.fill(0.0)
    compute.grad()

    # After two iterations out = 0.6 * x + 0.1 * x^2, so d(out)/dx = 0.6 + 0.2 * x.
    grad = x.grad.to_numpy()
    for i in range(n_env):
        assert grad[i] == pytest.approx(0.6 + 0.2 * x_np[i], rel=1e-5)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_adstack_offset_array_difference_loop_trip_grad_correct():
    # Adjacent offset-array reads describe ragged segments of lengths 2, 1, and 3. Check every gradient.
    starts_np = np.array([0, 2, 3, 6], dtype=np.int32)  # segment lengths 2, 1, 3
    n_seg = starts_np.size - 1
    total = int(starts_np[-1])
    n_env = 2
    x_np = np.linspace(0.2, 0.9, total * n_env).astype(np.float32).reshape(total, n_env)

    starts = qd.field(qd.i32, shape=n_seg + 1)
    x = qd.field(qd.f32, shape=(total, n_env), needs_grad=True)
    out = qd.field(qd.f32, shape=(n_seg, n_env), needs_grad=True)

    @qd.kernel
    def compute():
        for i, i_env in qd.ndrange(n_seg, n_env):
            seg_start = starts[i]
            seg_end = starts[i + 1]
            acc = qd.f32(0.0)
            for j in range(seg_end - seg_start):
                acc = acc + x[seg_start + j, i_env] * x[seg_start + j, i_env]
            out[i, i_env] = acc

    starts.from_numpy(starts_np)
    x.from_numpy(x_np)
    compute()
    out.grad.from_numpy(np.ones((n_seg, n_env), dtype=np.float32))
    x.grad.fill(0.0)
    compute.grad()

    # out[i, e] = sum_{j in segment i} x[j, e]^2, so d(out)/d(x[j, e]) = 2 * x[j, e] for the owning segment.
    grad = x.grad.to_numpy()
    for j in range(total):
        for e in range(n_env):
            assert grad[j, e] == pytest.approx(2.0 * x_np[j, e], rel=1e-5)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_stride_load_dominates_sibling_branch_grad_distinct_per_thread():
    # The reverse pass of a kernel whose first f32 adstack heap access sits inside one arm of a runtime `if` while
    # the sibling arm also hits the heap must keep per-thread gradients distinct. The memoized per-thread stride
    # load from the AdStackMetadata buffer is emitted at the task entry block so its SSA id dominates both arms;
    # emitting it lazily at the first heap access would define it inside the first arm and reuse it from the
    # sibling arm, which violates SPIR-V dominance and lands on Metal as an uninitialized stride register: every
    # thread then computes the same heap row base and the whole dispatch races on one row, collapsing the per-env
    # adjoints to a single value. The taken `else` arm below runs a field-bounded loop whose multiply spills the
    # operand on the f32 heap; the compiled-but-untaken `if` arm reads both a reduction and a component of a
    # vector, which is what places the first f32 heap access of the module inside that arm.
    n_env = 2
    dt = 0.01

    jtype = qd.field(qd.i32, shape=())
    q_end = qd.field(qd.i32, shape=())
    qpos = qd.field(qd.f32, shape=(2, n_env), needs_grad=True)
    vel = qd.field(qd.f32, shape=(2, n_env), needs_grad=True)

    @qd.kernel
    def compute():
        for i_env in range(n_env):
            if jtype[None] == 3:
                rv = qd.Vector([vel[0, i_env], vel[1, i_env]], dt=qd.f32)
                qpos[0, i_env] = rv.norm_sqr()
                qpos[1, i_env] = rv[0]
            else:
                for j in range(q_end[None]):
                    qpos[j, i_env] = vel[j, i_env] * dt

    jtype[None] = 2
    q_end[None] = 1
    qpos.fill(0.0)
    vel.fill(0.0)
    vel.grad.fill(0.0)
    seed = np.zeros((2, n_env), dtype=np.float32)
    seed[0, :] = np.arange(1, n_env + 1, dtype=np.float32)
    qpos.grad.from_numpy(seed)
    compute.grad()

    grad = vel.grad.to_numpy()[0]
    for i in range(n_env):
        assert grad[i] == pytest.approx((i + 1) * dt, rel=1e-6)


def test_adstack_vector_subscript_selfop_no_warnings(tmp_path):
    # Exercises reverse-mode differentiation of a common Vector pattern: a small Vector is built with a literal
    # initializer, one slot is overwritten by static subscript, and the whole Vector is then used in an in-place
    # op whose right-hand side reads the same Vector (e.g. a self-normalization `q *= q.norm_sqr()`). The test
    # guards that the backward compile completes without emitting any "Loading variable N before anything is
    # stored to it" UD-chain warnings, which is the signature of a reverse-grad kernel that would read
    # uninitialized adjoint slots at runtime.
    #
    # Internal details: `ReplaceLocalVarWithStacks` lowers each stack-backed static subscript store into a
    # full-tensor push.
    # Devs modifying `ReplaceLocalVarWithStacks::visit(LocalStoreStmt)` or the reach-in analysis in
    # `ir/control_flow_graph.cpp` must preserve the invariant that every non-target slot in that rebuilt tensor
    # has a reaching definition the store-to-load-forwarding walker can see - otherwise the warning fires here.
    # The warning is emitted by the C++ logger during `kernel.grad()` compilation, so the check runs in a
    # subprocess to capture stderr reliably regardless of log-sink state in the parent test session.
    child_script = textwrap.dedent(
        """
        import quadrants as qd

        qd.init(arch=qd.cpu)


        @qd.func
        def f(x):
            q = qd.Vector([0.0, 0.0, 0.0, 0.0], dt=qd.f32)
            q[1] = x
            q *= q.norm_sqr()
            return q


        x = qd.field(qd.f32, shape=(), needs_grad=True)
        y = qd.field(qd.f32, shape=(), needs_grad=True)


        @qd.kernel
        def k():
            q = f(x[None])
            y[None] = q[0] + q[1] + q[2] + q[3]


        x[None] = 1.5
        k()
        y.grad[None] = 1.0
        k.grad()
        assert abs(x.grad[None] - 6.75) < 1e-6
        """
    )
    script_path = tmp_path / "vector_subscript_selfop.py"
    script_path.write_text(child_script)
    env_no_cache = {"QD_OFFLINE_CACHE": "0"}
    env = {**os.environ, **env_no_cache}
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, check=True, env=env)
    stderr = result.stderr.decode()
    assert "Loading variable" not in stderr, (
        "reverse-mode AD emitted 'Loading variable N before anything is stored to it' warnings for a Vector "
        "subscript-assign + self-referencing in-place op pattern; stderr was:\n" + stderr
    )
