import sys
import threading
import time

import quadrants as ti
from quadrants.lang.ast import transform_tree as _original_transform_tree

from tests import test_utils

_kernel_module = sys.modules["quadrants.lang.kernel"]


def _slow_transform_tree(tree, ctx):
    """Widen the race window so a second thread can observe inside_kernel=True."""
    time.sleep(0.15)
    return _original_transform_tree(tree, ctx)


@test_utils.test()
def test_concurrent_kernel_materialization():
    """Two threads materializing different kernels must not interfere.

    Without the compilation lock in Kernel.materialize, thread B enters
    ASTGenerator.__call__ while thread A still has inside_kernel=True,
    triggering a spurious "nested kernels" error or an assertion failure
    on _compiling_callable.
    """

    @ti.kernel
    def kernel_a(x: ti.types.NDArray[ti.i32, 1]) -> None:
        for i in x:
            x[i] = x[i] + 1

    @ti.kernel
    def kernel_b(x: ti.types.NDArray[ti.i32, 1]) -> None:
        for i in x:
            x[i] = x[i] * 2

    errors: list[Exception] = []
    barrier = threading.Barrier(2)

    def run_kernel(kernel_fn, arr):
        try:
            barrier.wait(timeout=10)
            kernel_fn(arr)
        except Exception as e:
            errors.append(e)

    arr_a = ti.ndarray(ti.i32, (10,))
    arr_b = ti.ndarray(ti.i32, (10,))
    arr_a.fill(1)
    arr_b.fill(1)

    original_transform_tree = _kernel_module.transform_tree
    _kernel_module.transform_tree = _slow_transform_tree
    try:
        t1 = threading.Thread(target=run_kernel, args=(kernel_a, arr_a))
        t2 = threading.Thread(target=run_kernel, args=(kernel_b, arr_b))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
    finally:
        _kernel_module.transform_tree = original_transform_tree

    assert not errors, f"Concurrent kernel materialization failed: {errors}"
    assert (arr_a.to_numpy() == 2).all(), f"kernel_a wrong: {arr_a.to_numpy()}"
    assert (arr_b.to_numpy() == 2).all(), f"kernel_b wrong: {arr_b.to_numpy()}"
