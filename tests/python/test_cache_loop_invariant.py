import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cuda, qd.cpu])
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
