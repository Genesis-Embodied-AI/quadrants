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
def test_vector_element_literal_index_load_not_cached_over_aliasing_store(use_ndarray: bool) -> None:
    """Regression for issue #810 through vector/matrix *elements* (TensorType), the case raised in the
    PR #811 review.

    Here both aliasing accesses reach the caching pass as MatrixPtrStmts over ExternalPtr/GlobalPtr
    origins (a[i_c][0] store vs guarded a[0][0] read).  alias_analysis() returns 'different' (not
    'uncertain') for two MatrixPtrStmts whose origins are only 'uncertain' aliases, so a guard that
    queries alias_analysis on the loaded/stored MatrixPtr pointers directly misses the hazard.  The fix
    keys on the ExternalPtr/GlobalPtr origins (where alias_analysis is 'uncertain') and the cache pass
    resolves each MatrixPtr to that origin, so the element accesses stay out of the caching pass.
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
    assert res[0][0] == 1, f"stale literal-index vector-element read: out[0][0] = {res[0][0]}, expected 1"
    assert res[1][0] == 1, f"out[1][0] = {res[1][0]}, expected 1"
