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

    ``flag[i]`` is invariant w.r.t. the inner ``j`` loop, so its load is a candidate for
    cache_loop_invariant_global_vars.  It is written conditionally (``if j >= threshold``) inside that
    loop, and the read must observe the store.  Caching is only sound when the read and write pointers to
    the same address are the same statement; otherwise ``flag_access`` stamps the hoisted read
    ``activate=false``, the caching pass serves the pre-loop value, and the store is lost -- ``acc`` then
    sums the stale ``0`` instead of the stored ``1`` (this broke the rigid solver's convergence break-flag,
    an ~88% runtime regression / non-terminating loop).

    Whole-kernel CSE unifies those pointers on upstream; per-task CSE restores the same precondition via
    ``merge_global_ptrs`` for the field path (pre-offload ``GlobalPtrStmt``s) and ``cse_offloaded_tasks``
    for the ndarray path (``ExternalPtrStmt``s, which only exist post-offload).  Both are exercised here.
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
