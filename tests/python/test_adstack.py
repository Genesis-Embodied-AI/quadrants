import math

import pytest

import quadrants as qd

from tests import test_utils

_UNARY_OPS_PARAMS = [
    # Ops whose domain is all real values share a single `(step, offset)` pair that keeps the operand inside
    # the domain and, for `abs`/`sin`/`cos`, forces it to cross zero as `j` advances so the per-iteration
    # gradient sign actually varies. `abs` especially needs the sign-crossing: its derivative is piecewise-
    # constant, so a non-crossing operand would pass trivially. `exp`/`tanh` do not need the crossing (their
    # derivatives stay positive) but the same parameters work because their domain is all reals; they live
    # in this group on that basis alone.
    ("sin", 0.3, -0.4, 1e-4),
    ("cos", 0.3, -0.4, 1e-4),
    ("abs", 0.3, -0.4, 1e-4),
    ("tanh", 0.3, -0.4, 1e-4),
    ("exp", 0.3, -0.4, 1e-4),
    # Ops restricted to positive/subunit operands use a smaller step and zero offset to
    # stay inside their domain across every `x_val` and `n_iter` combination.
    # `tan` joins this positive-domain group because its singularity at pi/2 ~= 1.57 lies outside
    # the positive-path operand's reach for every `x_val` and `n_iter` combination.
    ("tan", 0.05, 0.0, 1e-4),
    ("log", 0.05, 0.0, 1e-4),
    ("sqrt", 0.05, 0.0, 1e-4),
    ("rsqrt", 0.05, 0.0, 1e-4),
    # asin/acos use a looser 1e-3 at f32 because native Vulkan asin/acos intrinsics on AMDGPU drift from the
    # CPU/PyTorch reference by ~1e-4 in single precision at n_iter=10. A per-iteration replay regression (the
    # one this test pins against) offsets the result by orders of magnitude, not parts per ten thousand, so
    # 1e-3 still catches it with plenty of margin. Other arches (CPU, Metal, CUDA) comfortably hit 1e-4 for
    # asin/acos too; the looser tolerance is just to keep the test green across drivers.
    ("asin", 0.05, 0.0, 1e-3),
    ("acos", 0.05, 0.0, 1e-3),
]


def _run_unary_loop_carried(qd_dtype, op_name, step, offset, x_val, n_iter, rel_tol):
    import torch

    qd_op = getattr(qd, op_name)
    torch_op = getattr(torch, op_name)
    torch_dtype = torch.float64 if qd_dtype == qd.f64 else torch.float32

    n = 4
    x = qd.field(qd_dtype, shape=n, needs_grad=True)
    y = qd.field(qd_dtype, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            acc = 0.0
            for j in range(n_iter):
                a = x[i] + qd.cast(j, qd_dtype) * step + offset
                acc += qd_op(a)
            y[None] += acc

    for i in range(n):
        x[i] = x_val + i * 0.05
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    x_t = torch.tensor([x_val + i * 0.05 for i in range(n)], dtype=torch_dtype, requires_grad=True)
    y_t = torch.zeros((), dtype=torch_dtype)
    for i in range(n):
        acc_t = torch.zeros((), dtype=torch_dtype)
        for j in range(n_iter):
            a_t = x_t[i] + float(j) * step + offset
            acc_t = acc_t + torch_op(a_t)
        y_t = y_t + acc_t
    y_t.backward()

    # Use `pytest.approx` directly rather than `test_utils.approx` so the caller's `rel_tol` is honoured as-is.
    # `test_utils.approx` floors `rel` to `max(rel, get_rel_eps())` which is >= 1e-6 on every backend (1e-4 on
    # Metal); that floor silently defeats the f64 variant's tight tolerance and makes it validate identical
    # precision to a loose f32 assertion. The backend-aware floor is not wanted here because the f64 variant
    # specifically exists to detect an f32-precision regression in an f64 backward pass.
    assert y[None] == pytest.approx(y_t.item(), rel=rel_tol)
    for i in range(n):
        assert x.grad[i] == pytest.approx(x_t.grad[i].item(), rel=rel_tol)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("x_val", [0.001, 0.15, 0.26, 0.399])
@pytest.mark.parametrize("op_name,step,offset,tol", _UNARY_OPS_PARAMS)
@test_utils.test(require=qd.extension.adstack)
def test_adstack_unary_loop_carried(op_name, step, offset, tol, x_val, n_iter):
    # Cross-check `d/dx sum_j op(x + j * step + offset)` against PyTorch autograd for a parametrized unary `op`.
    # Each op is sampled at interior values and at the edge of its domain: `x_val = 0.001` drives the positive-path
    # operand against 0 (log/sqrt gradients blow up there), `x_val = 0.399` drives it against 1 (asin/acos
    # gradient `1/sqrt(1 - x^2)` diverges), and `0.15` / `0.26` are interior samples. `0.26` (rather than `0.25`)
    # avoids `x[3] + offset = 0` at `j=0`, where `abs`'s derivative is undefined. The `(step, offset)` per op
    # keeps the operand inside the op's domain for every `x_val` and `n_iter`, while making the operand cross zero
    # for abs/sin/cos so the per-iteration sign actually varies.
    #
    # Extend the parametrize list below when a new unary op becomes differentiable through a dynamic loop: add
    # the op together with a `(step, offset)` pair that keeps the operand inside the op's domain. `n_iter = 1`
    # only covers the single-iteration path and will not on its own catch a regression in the multi-iteration
    # case; `abs` additionally needs the sign-crossing operand, otherwise its piecewise-constant derivative makes
    # the test pass trivially.
    #
    # Internal details: when reverse-mode AD walks a loop-variant unary op backwards it needs the exact forward
    # operand from each iteration to compute the gradient. If the op is not in the set the AD transform knows how
    # to back up per iteration, the operand falls back to a single slot overwritten each forward step, and the
    # reversed loop then reads the last-iteration value for every backward step - which at `n_iter >= 3` produces
    # a wrong gradient. That is the regression this test pins against: any unary op dropped from the supported set
    # causes the multi-iteration parametrize variants to fail.
    _run_unary_loop_carried(qd.f32, op_name, step, offset, x_val, n_iter, rel_tol=tol)


@pytest.mark.needs_torch
@pytest.mark.parametrize("n_iter", [1, 3, 10])
@pytest.mark.parametrize("x_val", [0.001, 0.15, 0.26, 0.399])
@pytest.mark.parametrize("op_name,step,offset,tol", _UNARY_OPS_PARAMS)
@test_utils.test(arch=[qd.cpu, qd.cuda], require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_unary_loop_carried_f64(op_name, step, offset, tol, x_val, n_iter):
    # f64 uses the same parametrize as f32 to keep ops in lockstep, but ignores the per-op f32 tolerance: f64
    # hits near-machine-precision on every backend, so a single tight global tolerance catches every drift.
    del tol
    _run_unary_loop_carried(qd.f64, op_name, step, offset, x_val, n_iter, rel_tol=1e-12)


def _run_basic_gradient(qd_dtype, n_iter, rel_tol, approx=test_utils.approx, abs_tol=None):
    # Builds the kernel, runs forward + backward, and asserts a correct gradient. `approx` defaults to
    # `test_utils.approx` which is correct for f32 (its backend-specific floor kicks in); f64 callers must pass
    # `pytest.approx` to honor a tight `rel_tol=1e-14` that `test_utils.approx` would otherwise floor to 1e-6.
    # `abs_tol` is forwarded as `abs=` to the approx; f64 callers must pass `abs_tol=0` because pytest.approx's
    # default `abs=1e-12` would otherwise dominate for the expected magnitudes here (~0.6-0.95), making the
    # effective tolerance ~1e-12 absolute rather than 1e-14 relative and missing f32-narrowing regressions.
    # The adstack is structurally required here so the backward compiler can reverse the dynamic `range(n_iter)`
    # at all; the companion `test_adstack_basic_gradient_negative` pins that disabling the adstack raises
    # `QuadrantsCompilationError("non static range")`. Value-correctness of the per-iteration `v` spilled on the
    # adstack is NOT exercised by the linear body `v = v * 0.95 + 0.01` - see `test_adstack_basic_gradient`'s
    # docstring for the details.
    n = 4
    x = qd.field(qd_dtype, shape=n, needs_grad=True)
    y = qd.field(qd_dtype, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            for _ in range(n_iter):
                v = v * 0.95 + 0.01
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

    # `v = v * 0.95 + 0.01` iterated n_iter times gives v_final = 0.95**n_iter * x[i] + const, so
    # dv_final/dx[i] == 0.95**n_iter independent of x[i], and dy/dx[i] equals the same quantity.
    expected = 0.95**n_iter
    approx_kwargs = {"rel": rel_tol}
    if abs_tol is not None:
        approx_kwargs["abs"] = abs_tol
    for i in range(n):
        assert x.grad[i] == approx(expected, **approx_kwargs)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@test_utils.test(require=qd.extension.adstack)
def test_adstack_basic_gradient(n_iter):
    # Smallest possible "does reverse-mode AD through a for-loop work at all" check. The kernel runs `n_iter`
    # iterations of `v = v * 0.95 + 0.01` per element and asserts that `dy/dx[i]` matches the analytical gradient
    # `0.95 ** n_iter` for every element.
    #
    # Internal details: the adstack is structurally required here so the backward compiler can reverse the
    # dynamic `range(n_iter)` at all - the companion `test_adstack_basic_gradient_negative` pins that disabling
    # the adstack raises `QuadrantsCompilationError` in exactly this kernel shape. Value-correctness of the
    # stored v, on the other hand, is NOT exercised: the loop body `v = v * 0.95 + 0.01` is linear, so the
    # backward chain `adj(v_prev) = 0.95 * adj(v_next)` only uses the compile-time constant 0.95 and never reads
    # v from the adstack. A broken push/load/pop that returned garbage for v would still produce the same exact
    # gradient. For push/load/pop value-correctness coverage, see `test_adstack_unary_loop_carried` (non-linear
    # unary ops in the loop body). `n_iter = 1` exercises the single-push adstack code path; `n_iter = 10`
    # exercises repeated push/pop under one forward invocation; multi-element coverage (n = 4) guards against
    # per-element accumulation bugs that a single-element variant would miss.
    _run_basic_gradient(qd.f32, n_iter=n_iter, rel_tol=1e-6)


@pytest.mark.parametrize("n_iter", [1, 3, 10])
@test_utils.test(arch=[qd.cpu, qd.cuda], require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_adstack_basic_gradient_f64(n_iter):
    # f64 counterpart of `test_adstack_basic_gradient`. Uses `pytest.approx` so the tight `rel_tol=1e-14` is honored;
    # `test_utils.approx` would floor it to `get_rel_eps()` (typically 1e-6) and silently pass an f32-precision
    # regression. `abs_tol=0` disables pytest.approx's default `abs=1e-12` floor, which would otherwise dominate for
    # the expected magnitudes `0.95**n_iter in [~0.6, 0.95]` and make the effective tolerance ~1e-12 absolute rather
    # than 1e-14 relative; that is still ~100x looser than f64 roundoff and would miss an f32-narrowing regression.
    _run_basic_gradient(qd.f64, n_iter=n_iter, rel_tol=1e-14, approx=pytest.approx, abs_tol=0)


@test_utils.test(require=qd.extension.adstack)
def test_eliminate_recomputable_pushes_preserves_zero_body_store():
    # Cross-checks `dloss/dc` against the analytic value `2 * cos(c) * cos(sin(c))` for an adstack-mode kernel where
    # `tmp` receives one recomputable body push (`tmp = qd.sin(c[None])`) plus one conditional zero body store
    # (`if reset[i] != 0: tmp = 0.0`). The reset mask `[0, 1, 0, 1]` zeros `tmp` for two of four iterations so only
    # the other two contribute to the gradient.
    #
    # Internal details: the adstack promotion of `tmp` comes from `AdStackAllocaJudger::visit(UnaryOpStmt)` (`tmp`
    # feeds the non-linear `qd.sin(tmp)` accumulator); the load+store rule does not fire because the offload-level
    # for-loop sits outside the IB and `dynamic_for_depth_` stays 0 throughout the judger walk. The body push value
    # `sin(GlobalLoad(c))` is a recomputable chain by `RecomputableChainAnalyzer`, which makes the stack a candidate
    # for `EliminateRecomputableAdStackPushes`. The eligibility gate must count the conditional `tmp = 0.0` as a
    # body push: two body pushes for one stack disqualify the stack and the original IR survives. A weaker gate that
    # classifies the conditional zero as the prologue init - e.g. detecting init by literal-zero value rather than
    # by position relative to the alloca - drops the user's zero store, rewires every `load_top` to `sin(c)`, and
    # produces `c.grad` exactly 2x the analytic value because the gradient flows through every iteration regardless
    # of the reset mask.
    n = 4
    c = qd.field(qd.f32, shape=(), needs_grad=True)
    reset = qd.field(qd.i32, shape=n)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in range(n):
            # Recomputable body push: sin of a globally-loaded scalar. The chain `sin(GlobalLoad(c))` has interior
            # side-effect-free ops only and no LoopIndex / LocalLoad leaves, so RecomputableChainAnalyzer returns
            # true.
            tmp = qd.sin(c[None])
            if reset[i] != 0:
                # Real `tmp = 0.0` body store. Lowers to AdStackPushStmt with value zero. A weaker eligibility gate
                # misclassifies this push as an init prologue push and silently erases it.
                tmp = 0.0
            # Use tmp as a non-linear unary op operand so AdStackAllocaJudger marks the alloca stack-needed via its
            # visit(UnaryOpStmt) rule, which is what makes this an AdStack in the first place (the load+store rule
            # does not fire here because the offload-level for-loop is outside the IB and `dynamic_for_depth_` stays
            # 0).
            loss[None] += qd.sin(tmp)

    c[None] = 0.5
    reset[0] = 0
    reset[1] = 1
    reset[2] = 0
    reset[3] = 1
    loss[None] = 0.0
    c.grad[None] = 0.0

    compute()
    loss.grad[None] = 1.0
    compute.grad()

    # Forward semantics: tmp_i = sin(c) when reset[i]==0, 0 when reset[i]!=0. Two of four iterations are not reset,
    # so loss == 2 * sin(sin(c)). Gradient: dloss/dc contributes cos(c)*cos(sin(c)) per non-reset iteration, so
    # dloss/dc == 2 * cos(c) * cos(sin(c)).
    n_no_reset = 2
    expected_loss = n_no_reset * math.sin(math.sin(0.5))
    expected_grad = n_no_reset * math.cos(0.5) * math.cos(math.sin(0.5))
    assert loss[None] == pytest.approx(expected_loss, rel=1e-5)
    assert c.grad[None] == pytest.approx(expected_grad, rel=1e-5)


@test_utils.test(require=qd.extension.adstack, cfg_optimization=False)
def test_backup_ssa_load_top_with_subsequent_pushes():
    # Pins `BackupSSA::is_load_top_stable` against the reverse-mode aliasing bug where cloning an
    # `AdStackLoadTopStmt` at the reverse cursor reads the stack's live top - which differs from the original
    # forward-time top whenever the same stack received another push between the original load_top and the end of
    # the IB. The minimal IR shape that exhibits this:
    #
    #   - A multi-axis `qd.ndrange(...)` whose flat loop index is decomposed into per-axis indices via repeated
    #     `floordiv` / `sub` over an adstack-backed value (each step pushes its result, then a subsequent
    #     `load_top` reads that pushed value to feed the next decomposition step).
    #   - A guarded division `out = in / mass` inside the kernel: `MakeAdjoint` emits two reverse formulas
    #     (`adj(in) += adj(out) / mass`, `adj(mass) += -adj(out) * in / mass^2`) that re-evaluate `mass` and `in`
    #     at the reverse cursor; that re-evaluation walks the per-axis index DAG, which BackupSSA was happy to
    #     reconstruct from `floordiv(load_top(stack), const)` clones - the load_top clones read the stack's
    #     final top, which is the LAST push (the inner axis) instead of the intermediate flat index.
    #
    # With the stable-load-top guard, BackupSSA detects the unsafe load_top, falls back to the per-IB alloca
    # spill (`load(op)`), and the reverse re-evaluation reads the per-iteration forward-time index, producing
    # the analytic gradients. Without the guard, the reverse pass reads `g_mass[j/2, j]` and `g_in[j/2, j]`
    # instead of `g_mass[i, j]` / `g_in[i, j]`, so cells whose forward gate was true at `(i=1, j=1)` end up
    # dividing `adj(out)` by `g_mass[0, 1]` (= 0) and the gradient comes out as `inf`/`nan`.
    #
    # Internal details: `cfg_optimization=False` keeps the compound `if true && (g_mass[i, j] > 0)` form alive
    # (the `&&` lowering inserts the multi-push into the gate stack that the chain analyzer must clone through);
    # Counting enables the stack transformation where the cloned-load_top path fires.
    g_mass = qd.field(qd.f32, shape=(2, 2), needs_grad=True)
    g_in = qd.field(qd.f32, shape=(2, 2), needs_grad=True)
    g_out = qd.field(qd.f32, shape=(2, 2), needs_grad=True)

    @qd.kernel
    def gated_div():
        for i, j in qd.ndrange(2, 2):
            if g_mass[i, j] > 0.0:
                g_out[i, j] = g_in[i, j] / g_mass[i, j]

    g_mass[1, 1] = 2.0
    g_in[1, 1] = 4.0
    g_out.grad[1, 1] = 1.0
    gated_div()
    gated_div.grad()

    # Forward: g_out[1, 1] = g_in[1, 1] / g_mass[1, 1] = 2.0; gate is false at every other cell so all other
    # gradients are zero. With dy = 1.0 on the active cell:
    #   d(g_in[1, 1])   = 1 / g_mass[1, 1]                = 0.5
    #   d(g_mass[1, 1]) = -g_in[1, 1] / g_mass[1, 1]^2    = -1.0
    assert g_in.grad[1, 1] == pytest.approx(0.5, rel=1e-6)
    assert g_mass.grad[1, 1] == pytest.approx(-1.0, rel=1e-6)
    # Inactive cells must not have received any gradient. A regression where `BackupSSA` re-clones the
    # load_top at the reverse cursor reads `g_mass[j/2, j] = g_mass[0, 1] = 0`, so 1/g_mass[0, 1] = inf
    # would land on `g_in.grad[0, 1]` via the wrong index. Probing every other cell catches both the inf
    # spill and any silent off-by-one accumulation.
    for i in range(2):
        for j in range(2):
            if (i, j) == (1, 1):
                continue
            assert g_in.grad[i, j] == pytest.approx(0.0, abs=1e-6)
            assert g_mass.grad[i, j] == pytest.approx(0.0, abs=1e-6)


@test_utils.test(
    require=qd.extension.adstack,
    cfg_optimization=False,
    force_scalarize_matrix=True,
)
def test_eliminate_recomputable_pushes_rejects_mutated_snode_chain_leaf():
    # Pins the `mutated_snodes` guard inside `RecomputableChainAnalyzer::is_recomputable` against the chain-clone
    # post-write read miscompile: a forward `GlobalLoadStmt` chain leaf reading a SNode the same kernel mutates
    # cannot be re-cloned at the reverse cursor - the cloned load re-issues the read after the forward writes have
    # updated the SNode, producing wrong gradients (`nan` / `inf`).
    #
    # Internal details:
    #   - Kernel shape: three top-level for-loops over the same `field` SNode.
    #     1. Atomic-add into `field[base, 0]` keyed by `base = floor(data[0]).cast(i32)`.
    #     2. Gated divide-by-self over the whole `field`, gate predicate `field[I, 0][0] > 0.0`.
    #     3. Consumer that reads `field[base, 0]` and accumulates into `data[1]` (the gradient endpoint).
    #   - `mutated_snodes(IB) = {field}`.
    #   - Without the guard: the analyzer admits the chain `GlobalLoad(GlobalPtr(field, base))` as recomputable;
    #     ERAP drops the adstack carrying that chain's value; `BackupSSA::generic_visit` re-clones the chain at
    #     the reverse cursor; the cloned `GlobalLoadStmt` reads `field` POST stage 2 (the divide-by-self has set
    #     every gated cell to the all-ones vector), so the adjoint flowing through `data[0]` blows up.
    #   - With the guard: the chain is rejected, ERAP keeps the original push/pop, reverse pops the iter-k value
    #     verbatim, and `data.grad[0]` matches the analytic all-ones vector.
    #   - `force_scalarize_matrix=True` is structural: the matrix-typed `field` value path is what ERAP latches
    #     onto; with scalarization off the chain's leaf shape changes and the bug no longer fires.
    #   - `cfg_optimization=False` keeps the gate's compound `&&` lowering alive so the reverse clone path
    #     actually triggers; with cfg-opt the cond folds and the clone never happens.
    #   - The trailing `1` axis on `field` matches `MakeAdjoint::visit(RangeForStmt)`'s reverse iteration;
    #     collapsing it folds the loop into a shape that misses ERAP's eligibility entirely.
    vec3 = qd.types.vector(3, qd.f32)
    data = qd.field(dtype=vec3, shape=(2,), needs_grad=True)
    field = qd.field(dtype=vec3, shape=(2, 2, 2, 1), needs_grad=True)

    data[0] = qd.Vector([1.0, 1.0, 1.0])

    @qd.kernel
    def k(data: qd.template(), field: qd.template()):
        for _ in qd.ndrange(1):
            base = qd.floor(data[0]).cast(qd.i32)
            field[base, 0] += data[0]
        for ii, jj, kk, i_b in qd.ndrange(2, 2, 2, 1):
            I = (ii, jj, kk)
            if field[I, i_b][0] > 0.0:
                field[I, i_b] = field[I, i_b] / field[I, i_b]
        for _ in qd.ndrange(1):
            base = qd.floor(data[0]).cast(qd.i32)
            data[1] = data[0] + field[base, 0]

    field[1, 1, 1, 0] = qd.Vector([1.0, 1.0, 1.0])
    data.grad[1] = qd.Vector([1.0, 1.0, 1.0])
    k(data, field)
    # Restore the initial primal state for the backward replay.
    field[1, 1, 1, 0] = qd.Vector([1.0, 1.0, 1.0])
    data[1] = qd.Vector([0.0, 0.0, 0.0])
    k.grad(data, field)

    expected = [1.0, 1.0, 1.0]
    for d in range(3):
        assert math.isfinite(data.grad[0][d]), f"non-finite at axis {d}: {data.grad[0][d]}"
        assert data.grad[0][d] == pytest.approx(expected[d], rel=1e-6, abs=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_mixed_f32_and_non_f32():
    # Runs a backward pass through a dynamic loop that carries both a float (`v`) and an integer counter (`j`), and
    # asserts the gradient on the float comes out correctly. Mixing the two types in one kernel exercises both
    # adstack storage paths at once.
    #
    # Internal detail: on SPIR-V, f32 adstacks live in BufferType::AdStackHeapFloat while i32/u1 adstacks share
    # BufferType::AdStackHeapInt (u1 reinterpreted as i32 via `ir_->cast(...)` at push/load). A codegen regression
    # in either path (e.g. pre-scan miscounting into the wrong stride, or the Push/Pop visitors routing to the
    # wrong heap_kind) would surface as a wrong gradient.
    x = qd.field(qd.f32)
    y = qd.field(qd.f32)
    qd.root.dense(qd.i, 1).place(x, x.grad)
    qd.root.place(y, y.grad)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            j = 0
            for _ in range(5):
                y[None] += v * qd.cast(j + 1, qd.f32)
                j = j + 1
                v = v + 0.1

    x[0] = 1.0
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    x.grad[0] = 0.0
    compute.grad()
    # d y / d x[0] = sum_{k=0..4} (k+1): v at iter k is x[0] + 0.1*k, weight is k+1, so coefficient on x[0] is
    # sum_{k=0..4} (k+1) = 15. Five exactly-representable f32 accumulations so the result is exact up to a handful of
    # ULPs.
    assert x.grad[0] == pytest.approx(15.0, rel=1e-6)


@test_utils.test(require=qd.extension.adstack)
def test_adstack_many_non_f32_stacks_heap_backed():
    # Regression test for macOS Metal. Deeply nested reverse-mode kernels create one i32 adstack per dynamic loop
    # (to replay the counter) and one u1 adstack per data-dependent if (to replay the branch). Function-scope
    # storage makes the per-thread private-memory footprint grow linearly with the number of adstacks, and the
    # Apple MSL compiler rejects shaders past a few dozen slots; at ~100+ slots the pipeline creation XPC-times
    # out. The heap-backing path keeps Function-scope memory bounded so such kernels compile and run correctly.
    #
    # The test packs many sibling loops + ifs into a single kernel so the reverse pass allocates several i32 and u1
    # adstacks at once, then asserts the gradient is still correct. Correctness is the load-bearing assertion: a
    # mis-offset between adstacks on the shared int heap would alias one stack's primal slice onto another and
    # produce a wrong gradient. CPU backends store adstacks as regular memory and are unaffected; this primarily
    # guards the SPIR-V heap_int path.
    n = 3
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            v = x[i]
            # Six sibling dynamic loops + six data-dependent ifs inside each. Each dynamic for-loop creates an i32
            # adstack for its counter; each branch on a float-derived predicate creates a u1 adstack for the flag.
            # Written as explicit duplicated loops rather than a meta-for so the adstack count is unambiguous
            # (meta-loops with `qd.static` would unroll to one adstack per instantiation anyway, but expanding
            # manually keeps the intent auditable when this test is inspected during a failure).
            for a in range(4):
                if qd.cast(a, qd.f32) < v:
                    v = v * 1.1
                else:
                    v = v + 0.01
            for b in range(4):
                if v > qd.cast(b, qd.f32):
                    v = v * 1.05
                else:
                    v = v - 0.01
            for c in range(4):
                if v * 0.5 > qd.cast(c, qd.f32):
                    v = v + 0.1
                else:
                    v = v * 0.99
            for d in range(4):
                if v + qd.cast(d, qd.f32) > 0.0:
                    v = v * 1.02
                else:
                    v = v + 0.02
            for e in range(4):
                if v - qd.cast(e, qd.f32) > 0.0:
                    v = v * 0.97
                else:
                    v = v + 0.03
            for f in range(4):
                if v * 2.0 > qd.cast(f, qd.f32):
                    v = v + 0.05
                else:
                    v = v * 1.03
            y[None] += v

    x_vals = [1.0, 2.0, 3.0]
    for i in range(n):
        x[i] = x_vals[i]
    y[None] = 0.0
    compute()
    y.grad[None] = 1.0
    for i in range(n):
        x.grad[i] = 0.0
    compute.grad()

    # Finite-difference reference. A symbolic gradient would require tracking which branch each if took at every
    # iteration, which is exactly what the adstack replays for us; FD keeps the test oracle independent of the
    # code under test. `h = 1e-2` sits comfortably inside every branch-flip margin at the chosen `x_vals` (the
    # `v > cast(b, f32)` thresholds are integers and the local slope is O(1), so perturbation of 1e-2 never
    # crosses a branch) while keeping f32 cancellation roughly ULP-of-y / (2h) = ~1e-5 relative. Each branch is
    # affine in `v`, so the composite function is piecewise-affine in `x[i]`; FD central diff has zero
    # truncation error on that class and the only irreducible contribution is the rounding floor above.
    h = 1e-2
    for i in range(n):
        x[i] = x_vals[i] + h
        y[None] = 0.0
        compute()
        y_plus = y[None]
        x[i] = x_vals[i] - h
        y[None] = 0.0
        compute()
        y_minus = y[None]
        x[i] = x_vals[i]
        expected = (y_plus - y_minus) / (2.0 * h)
        assert x.grad[i] == pytest.approx(expected, rel=1e-3, abs=1e-4)
