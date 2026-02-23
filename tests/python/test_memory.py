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
