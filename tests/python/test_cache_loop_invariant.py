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


@test_utils.test()
def test_conditional_store_to_loop_invariant_global() -> None:
    """Regression: a loop-invariant global written *conditionally* inside an ``if`` must not read stale.

    ``flag[i]`` is invariant w.r.t. the inner ``j`` loop, so its load is a candidate for
    cache_loop_invariant_global_vars.  It is written conditionally (``if j >= threshold``) inside that
    loop, and the read must observe the store.  Before the read and write ``GlobalPtrStmt``s to the same
    address were unified pre-offload (``merge_global_ptrs``, run once before the first ``flag_access``),
    per-task CSE left them split: ``flag_access`` stamped the hoisted read ``activate=false`` and the CSE
    eliminability rule refused to re-merge the later ``activate=true`` conditional write, so the cache served
    the pre-loop value.  That stale read broke the rigid solver's convergence break-flag (an ~88% runtime
    regression).  Here it manifests as ``acc`` summing the stale ``0`` instead of the stored ``1``.
    """
    n = 4
    m = 8
    threshold = 3

    @qd.kernel
    def k(flag: qd.template(), result: qd.template()):
        for i in range(n):  # offloaded task
            flag[i] = 0
            acc = 0
            for j in range(m):  # inner loop; flag[i] is loop-invariant here
                if j >= threshold:
                    flag[i] = 1  # conditional in-if store to the loop-invariant global
                acc += flag[i]  # must observe the store, not a stale cached load
            result[i] = acc

    flag = qd.field(dtype=qd.i32, shape=(n,))
    result = qd.field(dtype=qd.i32, shape=(n,))

    k(flag, result)
    expected = m - threshold  # flag == 1 for j in [threshold, m)
    for i in range(n):
        assert result[i] == expected, f"result[{i}] = {result[i]}, expected {expected}"
