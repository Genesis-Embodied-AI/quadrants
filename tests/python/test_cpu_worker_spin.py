import time

import numpy as np
import pytest

import quadrants as qd

from tests.helpers import test_utils


@pytest.mark.parametrize("budget", [0, 20, 200, 1000])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4)
def test_cpu_worker_spin_launches(budget):
    qd.init(arch=qd.cpu, cpu_max_num_threads=4, cpu_per_worker_min_block_dim=1, cpu_worker_spin_us=budget)
    assert qd.lang.impl.current_cfg().cpu_worker_spin_us == budget
    result = qd.ndarray(qd.i32, shape=137)

    @qd.kernel
    def fill(output: qd.Tensor, generation: qd.i32, threads: qd.Template):
        qd.loop_config(parallelize=threads)
        for i in range(137):
            output[i] = generation * 1000 + i

    for generation in range(40):
        threads = (1, 4, 2)[generation % 3]
        fill(result, generation, threads)
        np.testing.assert_array_equal(result.to_numpy(), generation * 1000 + np.arange(137))
        if generation % 7 == 0:
            time.sleep(0.002)
    qd.reset()


@test_utils.test(arch=qd.cpu, cpu_max_num_threads=2)
def test_cpu_worker_spin_environment(monkeypatch):
    monkeypatch.setenv("QD_CPU_WORKER_SPIN_US", "200")
    qd.init(arch=qd.cpu, cpu_max_num_threads=2)
    assert qd.lang.impl.current_cfg().cpu_worker_spin_us == 200
    qd.init(arch=qd.cpu, cpu_max_num_threads=2, cpu_worker_spin_us=0)
    assert qd.lang.impl.current_cfg().cpu_worker_spin_us == 0


@pytest.mark.parametrize("via_environment", [False, True])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=2)
def test_cpu_worker_spin_negative(via_environment, monkeypatch):
    options = {}
    if via_environment:
        monkeypatch.setenv("QD_CPU_WORKER_SPIN_US", "-1")
    else:
        options["cpu_worker_spin_us"] = -1
    with pytest.raises(RuntimeError, match="cpu_worker_spin_us must be >= 0"):
        qd.init(arch=qd.cpu, cpu_max_num_threads=2, **options)
