import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(require=qd.extension.assertion)
def test_local_matrix_non_constant_index_real_matrix():
    N = 1
    x = qd.Vector.field(3, float, shape=1)

    @qd.kernel
    def test_invariant_cache():
        for i in range(1):
            x[i][1] = x[i][1] + 1.0
            for j in range(1):
                x[i][1] = x[i][1] - 5.0
                for z in range(1):
                    idx = 0
                    if z == 0:
                        idx = 1
                    x_print = x[i][idx]

                    assert x_print == x[i][1]

    test_invariant_cache()


@pytest.mark.parametrize("use_ndarray", [False, True])
@test_utils.test()
def test_atomic_dest_not_cached(use_ndarray: bool) -> None:
    """Regression: cache_loop_invariant must skip fields written by AtomicOpStmt.

    On SPIR-V backends (Metal/Vulkan), atomics in serial tasks are kept as real
    atomic operations (not demoted to load-op-store).  The cache pass must
    recognise these AtomicOpStmt destinations and refuse to cache loads from the
    same field, otherwise reads inside the loop return stale pre-loop values.
    """
    n = 4
    m = 8

    TensorType = qd.ndarray if use_ndarray else qd.field

    AnnotationType = qd.types.ndarray() if use_ndarray else qd.template()

    @qd.kernel
    def k(x: AnnotationType, result: AnnotationType):
        qd.loop_config(serialize=True)
        for i in range(n):
            x[i] = 0
            for j in range(m):
                qd.atomic_add(x[i], 1)
                result[i] = x[i]

    x = TensorType(dtype=qd.i32, shape=(n,))
    result = TensorType(dtype=qd.i32, shape=(n,))

    k(x, result)
    for i in range(n):
        assert result[i] == m, f"result[{i}] = {result[i]}, expected {m}"


@pytest.mark.parametrize("use_ndarray", [False, True])
@test_utils.test()
def test_conditional_store_to_loop_invariant_global(use_ndarray: bool) -> None:
    """Regression: a loop-invariant global written *conditionally* inside an ``if`` must not read stale.

    ``flag[i]`` is invariant w.r.t. the inner ``j`` loop, so the cache_loop_invariant_global_vars pass may hoist its
    load out of the loop.  It is written conditionally (``if j >= threshold``) inside that loop, and the read must
    observe the store.  Caching is only sound when the read and write pointers to the same address are the same
    statement; otherwise ``flag_access`` stamps the hoisted read ``activate=false``, the caching pass serves the
    pre-loop value, and the store is lost -- ``acc`` then sums the stale ``0`` instead of the stored ``1`` (this broke
    the rigid solver's convergence break-flag, an ~88% runtime regression / non-terminating loop).

    Whole-kernel CSE unifies those pointers on upstream.  Per-task CSE restores the same precondition via the
    ``merge_global_ptrs`` pass for the field path (pre-offload ``GlobalPtrStmt``s) and ``cse_offloaded_tasks`` for the
    ndarray path (``ExternalPtrStmt``s, which only exist post-offload).  Both are exercised here.
    """
    n = 4
    m = 8
    threshold = 3

    AnnotationType = qd.types.ndarray() if use_ndarray else qd.template()
    TensorType = qd.ndarray if use_ndarray else qd.field

    @qd.kernel
    def k(flag: AnnotationType, result: AnnotationType):
        for i in range(n):  # offloaded task
            flag[i] = 0
            acc = 0
            for j in range(m):  # inner loop; flag[i] is loop-invariant here
                if j >= threshold:
                    flag[i] = 1  # conditional in-if store to the loop-invariant global
                acc += flag[i]  # must observe the store, not a stale cached load
            result[i] = acc

    flag = TensorType(dtype=qd.i32, shape=(n,))
    result = TensorType(dtype=qd.i32, shape=(n,))

    k(flag, result)
    expected = m - threshold  # flag == 1 for j in [threshold, m)
    for i in range(n):
        assert result[i] == expected, f"result[{i}] = {result[i]}, expected {expected}"


@pytest.mark.parametrize("use_ndarray", [False, True])
@test_utils.test()
def test_literal_index_load_not_cached_over_aliasing_store(use_ndarray: bool) -> None:
    """Regression for issue #810: a literal-index load (a[0]) that may alias a loop-variable-index
    store (a[i]) must not be served from a cached local slot while the store's write-back is deferred
    past the loop.  When the two indices coincide (i == 0) the literal-index read otherwise returns a
    stale pre-write-back value.

    Requires all four ingredients: a serialized outer loop, an inner loop, a store through the outer
    loop variable index, and a literal-index read of the same buffer guarded by a comparison of the
    loop variable to that literal.
    """
    n = 2

    TensorType = qd.ndarray if use_ndarray else qd.field

    AnnotationType = qd.types.ndarray() if use_ndarray else qd.template()

    @qd.kernel
    def k(a: AnnotationType, out: AnnotationType):
        qd.loop_config(serialize=True)
        for i_c in range(n):
            for i_iter in range(1):
                a[i_c] = 1
                if i_c == 0:
                    out[0] = a[0]
                    out[1] = a[i_c]

    a = TensorType(dtype=qd.i32, shape=(n,))
    out = TensorType(dtype=qd.i32, shape=(n,))

    k(a, out)
    assert out[0] == 1, f"stale literal-index read: out[0] = {out[0]}, expected 1"
    assert out[1] == 1, f"out[1] = {out[1]}, expected 1"


@pytest.mark.parametrize("use_ndarray", [False, True])
@test_utils.test()
def test_loop_index_load_not_cached_over_aliasing_store(use_ndarray: bool) -> None:
    """Regression for the same hazard as #810 but through two DISTINCT loop-variable indices instead of
    a literal.  A store through one loop index (a[i_c]) and a guarded read through another loop index
    (a[t]) may alias when i_c == t, so the buffer must not be cached.

    A structural const-vs-loop-index guard misses this because both accesses are loop-indexed; an
    alias-analysis guard catches it because alias_analysis(a[i_c], a[t]) is 'uncertain'.
    """
    n = 2

    TensorType = qd.ndarray if use_ndarray else qd.field

    AnnotationType = qd.types.ndarray() if use_ndarray else qd.template()

    @qd.kernel
    def k(a: AnnotationType, out: AnnotationType):
        qd.loop_config(serialize=True)
        for t in range(n):
            for i_c in range(n):
                for i_iter in range(1):
                    a[i_c] = 1
                    if i_c == t:
                        out[t] = a[t]

    a = TensorType(dtype=qd.i32, shape=(n,))
    out = TensorType(dtype=qd.i32, shape=(n,))

    k(a, out)
    assert out[0] == 1, f"stale loop-index read: out[0] = {out[0]}, expected 1"
    assert out[1] == 1, f"stale loop-index read: out[1] = {out[1]}, expected 1"


@pytest.mark.parametrize("use_ndarray", [False, True])
@test_utils.test()
def test_vector_element_aliasing_store_no_miscompile(use_ndarray: bool) -> None:
    """Guard test for the vector/matrix *element* variant of issue #810, raised in the PR #811 review.

    The concern: when the aliasing accesses are vector/matrix elements they reach the caching pass as
    MatrixPtrStmts over ExternalPtr/GlobalPtr origins (a[i_c][0] store vs guarded a[0][0] read), and
    alias_analysis() returns 'different' (not 'uncertain') for two MatrixPtrStmts whose origins are only
    'uncertain' aliases -- so a guard keyed on the loaded/stored MatrixPtr pointers could miss the
    hazard.

    Empirically (x64) this pattern is NOT miscompiled on either baseline main or the fix: across
    ndarray-vector, field-vector, matrix-field, and two-loop-index variants the matrix-element caching
    path never defers the write-back the way the scalar path does, so no stale read occurs regardless of
    the may-alias guard.  This test does not distinguish the fix; it guards against a future change that
    would start hoisting matrix-element accesses and reintroduce the hazard.
    """
    n = 2
    k = 2

    if use_ndarray:
        a = qd.Vector.ndarray(k, qd.i32, shape=(n,))
        out = qd.Vector.ndarray(k, qd.i32, shape=(n,))
        AnnotationType = qd.types.ndarray(ndim=1)
    else:
        a = qd.Vector.field(k, qd.i32, shape=(n,))
        out = qd.Vector.field(k, qd.i32, shape=(n,))
        AnnotationType = qd.template()

    @qd.kernel
    def kern(a: AnnotationType, out: AnnotationType):
        qd.loop_config(serialize=True)
        for i_c in range(n):
            for i_iter in range(1):
                a[i_c][0] = 1
                if i_c == 0:
                    out[0][0] = a[0][0]
                    out[1][0] = a[i_c][0]

    kern(a, out)
    res = out.to_numpy()
    assert res[0][0] == 1, f"vector-element stale read: out[0][0] = {res[0][0]}, expected 1"
    assert res[1][0] == 1, f"out[1][0] = {res[1][0]}, expected 1"


@pytest.mark.xfail(
    strict=True,
    reason="#812 review: cross-parameter ndarray aliasing is not yet handled. When the same ndarray is "
    "bound to two kernel parameters, the two ExternalPtrStmts have distinct ArgLoadStmt arg_ids, so "
    "alias_analysis() returns 'different' (not 'uncertain') and the may-alias caching guard never fires "
    "across the parameters. Deciding this needs launch-time aliasing knowledge (kernels are compiled "
    "per-signature, not per-pointer), so it is out of scope for the same-parameter #810 fix.",
)
@test_utils.test(arch=qd.cpu)
def test_cross_parameter_ndarray_aliasing_store_stale_read() -> None:
    """Known gap (issue #810 / PR #812 review): the #810 hazard split across two ndarray parameters that
    the caller binds to the SAME ndarray at launch.  The loop-variable-index store goes through one
    parameter (a[i_c]) and the guarded literal-index load through the other (b[0]); both are the same
    ndarray, so b[0] aliases a[i_c] when i_c == 0.

    Because the two parameters have distinct arg_ids, alias_analysis(a[i_c], b[0]) is 'different' rather
    than 'uncertain', the guard is skipped, and b[0] returns a stale pre-write-back value.  The
    field/template variant of this (same field bound to two params) IS fixed, because both handles
    resolve to the same SNode and alias_analysis() then reports 'uncertain'.

    Marked xfail(strict): if a later change closes the ndarray gap this will XPASS and flag that the
    guard and this test should be promoted to a real regression assertion.
    """
    n = 2

    @qd.kernel
    def k(a: qd.types.ndarray(), b: qd.types.ndarray(), out: qd.types.ndarray()):
        qd.loop_config(serialize=True)
        for i_c in range(n):
            for i_iter in range(1):
                a[i_c] = 1
                if i_c == 0:
                    out[0] = b[0]
                    out[1] = a[i_c]

    a = qd.ndarray(dtype=qd.i32, shape=(n,))
    out = qd.ndarray(dtype=qd.i32, shape=(n,))

    k(a, a, out)  # same ndarray bound to both parameters -> b aliases a at launch
    assert out[0] == 1, f"stale cross-parameter read: out[0] = {out[0]}, expected 1"
    assert out[1] == 1, f"out[1] = {out[1]}, expected 1"
