import time

import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test()
def test_parallel_range_for():
    n = 1024 * 1024
    val = qd.field(qd.i32, shape=(n))

    @qd.kernel
    def fill():
        qd.loop_config(parallelize=8, block_dim=8)
        for i in range(n):
            val[i] = i

    fill()
    # To speed up
    val_np = val.to_numpy()
    for i in range(n):
        assert val_np[i] == i


@test_utils.test()
def test_serial_for():
    @qd.kernel
    def foo() -> qd.i32:
        a = 0
        qd.loop_config(serialize=True)
        for i in range(100):
            a = a + 1
            if a == 50:
                break

        return a

    assert foo() == 50


@test_utils.test()
def test_loop_config_parallel_range_for():
    n = 1024 * 1024
    val = qd.field(qd.i32, shape=(n))

    @qd.kernel
    def fill():
        qd.loop_config(parallelize=8, block_dim=8)
        for i in range(n):
            val[i] = i

    fill()
    # To speed up
    val_np = val.to_numpy()
    for i in range(n):
        assert val_np[i] == i


@test_utils.test()
def test_loop_config_serial_for():
    @qd.kernel
    def foo() -> qd.i32:
        a = 0
        qd.loop_config(serialize=True)
        for i in range(100):
            a = a + 1
            if a == 50:
                break

        return a

    assert foo() == 50


def _finishing_order(n):
    order = qd.field(qd.i32, shape=n)
    ticket = qd.field(qd.i32, shape=())
    sink = qd.field(qd.f32, shape=n)

    @qd.kernel
    def run():
        ticket[None] = 0
        for i in range(n):
            # Earlier iterations do more work, so an iteration run by another thread finishes before them
            acc = 0.0
            for k in range((n - i) * 20000):
                acc += qd.sin(acc + k * 1e-6)
            sink[i] = acc
            order[qd.atomic_add(ticket[None], 1)] = i

    run()
    return order.to_numpy()


@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, cpu_min_range_for_block=1)
def test_cpu_min_range_for_block_splits_short_loop():
    order = _finishing_order(16)
    assert sorted(order) == list(range(16))
    assert order[0] != 0


@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4)
def test_cpu_short_loop_runs_in_one_block_by_default():
    order = _finishing_order(16)
    assert list(order) == list(range(16))


@test_utils.test(arch=qd.cpu, cpu_max_num_threads=8, cpu_min_range_for_block=1)
def test_cpu_thread_pool_successive_launches():
    n = 64
    val = qd.field(qd.i32, shape=n)

    @qd.kernel
    def fill(offset: qd.i32):
        for i in range(n):
            val[i] = i + offset

    # Back-to-back launches reach the workers while they spin, the pauses reach them asleep
    for launch in range(200):
        fill(launch)
        assert (val.to_numpy() == np.arange(n) + launch).all()
        if launch % 50 == 0:
            time.sleep(0.01)
