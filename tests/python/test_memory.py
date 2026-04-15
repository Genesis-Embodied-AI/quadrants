import gc
import os

import psutil
import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@pytest.mark.run_in_serial
@test_utils.test(arch=qd.cuda)
def test_memory_allocate():
    HUGE_SIZE = 1024**2 * 128
    x = qd.field(qd.i32, shape=(HUGE_SIZE,))
    for i in range(10):
        x[i] = i


@pytest.mark.run_in_serial
@test_utils.test()
def test_ndarray_exceeds_default_device_memory_gb():
    # device_memory_GB defaults to 1, so allocate an ndarray whose byte size exceeds 1 GiB to exercise that
    # on-demand growth works without user-side pool tuning. On AMDGPU this path goes through hipMallocAsync since
    # ROCm 5.2+; CUDA uses cuMemAllocAsync; CPU/Vulkan/Metal are unaffected by device_memory_GB.
    n = (1 << 30) // 4 + 1
    arr = qd.ndarray(qd.i32, shape=(n,))
    assert arr[0] == 0
    assert arr[n // 2] == 0
    assert arr[n - 1] == 0


@test_utils.test(require=qd.extension.sparse)
def test_sparse_field_with_device_memory_pool():
    # With the device memory pool active on CUDA / AMDGPU, the eager preallocate_runtime_memory() in
    # materialize_runtime is skipped. Sparse (non-dense) SNode trees depend on a lazy fallback in
    # initialize_llvm_runtime_snodes to wire up the device-side bump allocator; exercise that path.
    x = qd.field(qd.i32)
    qd.root.pointer(qd.i, 32).dense(qd.i, 8).place(x)

    @qd.kernel
    def write():
        x[0] = 42
        x[255] = 7

    write()
    assert x[0] == 42
    assert x[255] == 7


@test_utils.test(arch=get_host_arch_list())
def test_oop_memory_leak():
    @qd.data_oriented
    class X:
        def __init__(self):
            self.py_l = [0] * 5242880  # a list containing 5M integers (5 * 2^20)

        @qd.kernel
        def run(self):
            for i in range(1):
                pass

    def get_process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / 1e6  # in MB

    # Init & Warm up
    for i in range(2):
        X().run()
        gc.collect()

    ref_mem = get_process_memory()
    for i in range(50):
        X().run()
        gc.collect()
        curr_mem = get_process_memory()
        assert curr_mem - ref_mem < 5  # shouldn't increase more than 5.0 MB each loop


@test_utils.test(arch=[qd.cuda])
def test_cuda_memory_reuse():
    def ad_sum_vector():
        N = 10

        @qd.kernel
        def compute_sum(a: qd.types.ndarray(), p: qd.types.ndarray()):
            for i in p:
                p[i] = a[i] * 2

        a = qd.ndarray(qd.math.vec2, shape=N, needs_grad=True)
        p = qd.ndarray(qd.math.vec2, shape=N, needs_grad=True)
        for i in range(N):
            a[i] = [3, 3]

        compute_sum(a, p)

        for i in range(N):
            assert p[i] == [a[i] * 2, a[i] * 3]
            p.grad[i] = [1, 1]

        compute_sum.grad(a, p)

        for i in range(N):
            for j in range(2):
                assert a.grad[i][j] == 2

    qd.init(arch=qd.cuda)
    ad_sum_vector()

    qd.init(arch=qd.cuda)
    ad_sum_vector()
