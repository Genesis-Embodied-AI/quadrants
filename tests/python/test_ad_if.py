import pytest

import quadrants as qd
from quadrants.lang import impl
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(require=qd.extension.adstack)
def test_ad_if_simple():
    x = qd.field(qd.f32, shape=())
    y = qd.field(qd.f32, shape=())

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        if x[None] > 0.0:
            y[None] = x[None]

    x[None] = 1
    y.grad[None] = 1

    func()
    func.grad()

    assert x.grad[None] == 1


@test_utils.test(require=qd.extension.adstack)
def test_ad_if():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)

    qd.root.lazy_grad()

    @qd.kernel
    def func(i: qd.i32):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 2 * x[i]

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0)
    func.grad(0)
    func(1)
    func.grad(1)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(require=qd.extension.adstack)
def test_ad_if_nested():
    n = 20
    x = qd.field(qd.f32, shape=n)
    y = qd.field(qd.f32, shape=n)
    z = qd.field(qd.f32, shape=n)

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in x:
            if x[i] < 2:
                if x[i] == 0:
                    y[i] = 0
                else:
                    y[i] = z[i] * 1
            else:
                if x[i] == 2:
                    y[i] = z[i] * 2
                else:
                    y[i] = z[i] * 3

    z.fill(1)

    for i in range(n):
        x[i] = i % 4

    func()
    for i in range(n):
        assert y[i] == i % 4
        y.grad[i] = 1
    func.grad()

    for i in range(n):
        assert z.grad[i] == i % 4


@test_utils.test(require=qd.extension.adstack)
def test_ad_if_mutable():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)

    qd.root.lazy_grad()

    @qd.kernel
    def func(i: qd.i32):
        t = x[i]
        if t > 0:
            y[i] = t
        else:
            y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func(0)
    func.grad(0)
    func(1)
    func.grad(1)

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(require=qd.extension.adstack)
def test_ad_if_parallel():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_ad_if_parallel_f64():
    x = qd.field(qd.f64, shape=2)
    y = qd.field(qd.f64, shape=2)

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        for i in range(2):
            t = x[i]
            if t > 0:
                y[i] = t
            else:
                y[i] = 2 * t

    x[0] = 0
    x[1] = 1
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 2
    assert x.grad[1] == 1


@test_utils.test(require=qd.extension.adstack)
def test_ad_if_parallel_complex():
    x = qd.field(qd.f32, shape=2)
    y = qd.field(qd.f32, shape=2)

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        qd.loop_config(parallelize=1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 0
    assert x.grad[1] == -0.25


@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_ad_if_parallel_complex_f64():
    x = qd.field(qd.f64, shape=2)
    y = qd.field(qd.f64, shape=2)

    qd.root.lazy_grad()

    @qd.kernel
    def func():
        qd.loop_config(parallelize=1)
        for i in range(2):
            t = 0.0
            if x[i] > 0:
                t = 1 / x[i]
            y[i] = t

    x[0] = 0
    x[1] = 2
    y.grad[0] = 1
    y.grad[1] = 1

    func()
    func.grad()

    assert x.grad[0] == 0
    assert x.grad[1] == -0.25


@test_utils.test(arch=get_host_arch_list())
def test_stack():
    @qd.kernel
    def func():
        impl.call_internal("test_stack")

    func()


@test_utils.test(require=[qd.extension.adstack])
def test_if_condition_depend_on_for_loop_index():
    scalar = lambda: qd.field(dtype=qd.f32)
    vec = lambda: qd.Vector.field(3, dtype=qd.f32)

    pos = vec()
    F = vec()
    f_bend = scalar()
    loss_n = scalar()
    qd.root.dense(qd.ij, (10, 10)).place(pos, F)
    qd.root.dense(qd.i, 1).place(f_bend)
    qd.root.place(loss_n)
    qd.root.lazy_grad()

    @qd.kernel
    def simulation(t: qd.i32):
        for i, j in pos:
            coord = qd.Vector([i, j])
            for n in range(12):
                f = qd.Vector([0.0, 0.0, 0.0])
                if n < 4:
                    f = qd.Vector([1.0, 1.0, 1.0])
                else:
                    f = f_bend[0] * pos[coord]
                F[coord] += f
            pos[coord] += 1.0 * t

    with qd.ad.Tape(loss=loss_n):
        simulation(5)


def _run_nested_if_inside_for_loop(qd_dtype):
    w_vals = [-2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0]
    n = len(w_vals)
    w = qd.field(qd_dtype, shape=n, needs_grad=True)
    loss = qd.field(qd_dtype, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in w:
            if w[i] > 0:
                if w[i] < 10:
                    loss[None] += w[i] * w[i]

    for i, v in enumerate(w_vals):
        w[i] = v
    loss[None] = 0.0
    loss.grad[None] = 1.0
    compute()
    compute.grad()

    for i, v in enumerate(w_vals):
        # d/dw[i] (w[i] * w[i]) == 2 * w[i] when both conditions hold; 0 otherwise.
        expected = 2.0 * v if 0.0 < v < 10.0 else 0.0
        assert w.grad[i] == expected


@test_utils.test(require=qd.extension.adstack)
def test_ad_nested_if_inside_for_loop():
    # Regression test for adjoint-alloca placement when a field read (`w[i]`) appears inside nested `if` blocks
    # within a for-loop being differentiated. Before the fix, the gradient accumulator for `w[i]` was placed inside
    # the forward `if` body, but the reverse pass generates its backward code in a separate, parallel `if` block
    # that can't see variables defined in the forward one. The accumulator was silently eliminated as dead code,
    # and `w.grad[i]` came out as zero instead of the correct `2 * w[i]`.
    #
    # Some inputs deliberately fail the outer (`w[i] > 0`) or inner (`w[i] < 10`) condition so the accumulator is
    # never written for them; `w.grad[i]` must be exactly 0 on those elements, otherwise the backward pass is
    # accumulating a contribution from an untaken branch.
    #
    # Internal detail: MakeAdjoint placed the adjoint alloca inside the forward if-body; the reverse pass emits
    # the backward code into a brand-new sibling IfStmt whose SSA does not dominate that alloca, so DCE stripped
    # it.
    _run_nested_if_inside_for_loop(qd.f32)


@test_utils.test(require=[qd.extension.adstack, qd.extension.data64], default_fp=qd.f64)
def test_ad_nested_if_inside_for_loop_f64():
    _run_nested_if_inside_for_loop(qd.f64)


@test_utils.test(require=qd.extension.adstack)
def test_ad_nested_if_elif_else_inside_for_loop():
    # Exercises the same adjoint-alloca placement fix as `test_ad_nested_if_inside_for_loop` but with explicit
    # `else` / `elif` arms: the outer `if` has an `else` that reads `w[i]`, and the inner structure is
    # `if / elif / else` with each branch reading `w[i]`. Three sibling IfStmts with distinct reversed-branch
    # SSA must all see the same adjoint alloca; placement must hoist above every one of them.
    #
    # Per-element expected gradient depends on which arm fires:
    #   v >  0 and v <  5 : loss += 2 * w[i] * w[i]          -> grad = 4 * v
    #   v >  0 and 5 <= v < 10 : loss +=     w[i] * w[i]     -> grad = 2 * v
    #   v >  0 and v >= 10: loss += 3 * w[i]                 -> grad = 3
    #   v <= 0            : loss += -w[i]                    -> grad = -1
    w_vals = [-2.0, -0.5, 1.0, 3.0, 5.0, 7.5, 10.0, 20.0]
    n = len(w_vals)
    w = qd.field(qd.f32, shape=n, needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in w:
            if w[i] > 0:
                if w[i] < 5:
                    loss[None] += 2 * w[i] * w[i]
                elif w[i] < 10:
                    loss[None] += w[i] * w[i]
                else:
                    loss[None] += 3 * w[i]
            else:
                loss[None] += -w[i]

    for i, v in enumerate(w_vals):
        w[i] = v
    loss[None] = 0.0
    loss.grad[None] = 1.0
    compute()
    compute.grad()

    for i, v in enumerate(w_vals):
        if v > 0 and v < 5:
            expected = 4.0 * v
        elif v > 0 and v < 10:
            expected = 2.0 * v
        elif v > 0:
            expected = 3.0
        else:
            expected = -1.0
        assert w.grad[i] == expected


@test_utils.test(require=qd.extension.adstack)
def test_ad_nested_for_loops_global_load():
    # Exercises the walk-up path when the GlobalLoadStmt lives in an outer for-loop whose alloca must be hoisted
    # above an inner for-loop. Original bug description: "GlobalLoadStmt outside the currently-processed range-for"
    # - the walk-up must not stop at the inner range-for but must continue past its body to the outer for.
    #
    # Kernel shape: for i in x: a = x[i]; for _ in range(n_inner): y += a. The adjoint of `x[i]` must persist
    # across every iteration of the inner reversed loop; otherwise the alloca gets placed inside the inner
    # reversed body and the accumulation is applied n_inner times per outer iteration, producing grad = n_inner^2
    # instead of n_inner.
    n = 4
    n_inner = 3
    x = qd.field(qd.f32, shape=n, needs_grad=True)
    y = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in x:
            a = x[i]
            for _ in range(n_inner):
                y[None] += a

    for i in range(n):
        x[i] = 1.0
    y[None] = 0.0
    y.grad[None] = 1.0
    compute()
    compute.grad()

    for i in range(n):
        assert x.grad[i] == float(n_inner)


@pytest.mark.xfail(
    reason="Reverse-mode AD does not yet support while loops (auto_diff.cpp visit(WhileStmt) -> QD_NOT_IMPLEMENTED).",
    strict=True,
    raises=RuntimeError,
)
@test_utils.test(require=qd.extension.adstack)
def test_ad_nested_if_inside_while_loop():
    # Same placement regression as `test_ad_nested_if_inside_for_loop`, but the nested `if` sits inside a dynamic
    # `while` loop rather than a range-for. Currently xfails because the reverse-mode AD transform does not yet
    # have a `visit(WhileStmt)` implementation.
    #
    # Internal details: the IR shape (while wrapping nested ifs wrapping a field read) is the one the alloca-
    # placement fix needs to hold on; that is a different control-flow construct from the range-for currently
    # exercised. The `while` body runs a single iteration per element - the point is the IR shape, not the trip
    # count.
    w_vals = [-2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0]
    n = len(w_vals)
    w = qd.field(qd.f32, shape=n, needs_grad=True)
    loss = qd.field(qd.f32, shape=(), needs_grad=True)

    @qd.kernel
    def compute():
        for i in w:
            step = 0
            while step < 1:
                if w[i] > 0:
                    if w[i] < 10:
                        loss[None] += w[i] * w[i]
                step = step + 1

    for i, v in enumerate(w_vals):
        w[i] = v
    loss[None] = 0.0
    loss.grad[None] = 1.0
    compute()
    compute.grad()

    for i, v in enumerate(w_vals):
        expected = 2.0 * v if 0.0 < v < 10.0 else 0.0
        assert w.grad[i] == expected
