import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("block_dim", [32, 64])
@test_utils.test(make_cpu_multithreading_loop=False, cpu_max_num_threads=4)
def test_block_idx_portable(block_dim):
    block_indices = qd.ndarray(dtype=qd.i32, shape=(200, 2))

    @qd.func
    def get_block_idx():
        return qd.block_idx()

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=2)):
        qd.loop_config(block_dim=block_dim)
        for i in range(200):
            for j in range(2):
                out[i, j] = get_block_idx()

    k_record_blocks(block_indices)
    expected = np.repeat((np.arange(200, dtype=np.int32) // block_dim)[:, None], 2, axis=1)
    np.testing.assert_array_equal(block_indices.to_numpy(), expected)


@test_utils.test()
def test_block_idx_serial():
    @qd.kernel
    def k_block_idx() -> qd.i32:
        return qd.block_idx()

    assert k_block_idx() == 0


@test_utils.test(arch=[qd.cuda, qd.amdgpu], saturating_grid_dim=2)
def test_block_idx_grid_stride():
    block_indices = qd.ndarray(dtype=qd.i32, shape=513)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=32)
        for i in range(513):
            out[i] = qd.block_idx()

    k_record_blocks(block_indices)
    # Two hardware blocks revisit the loop, alternating groups of 32 iterations.
    np.testing.assert_array_equal(block_indices.to_numpy(), (np.arange(513, dtype=np.int32) // 32) % 2)


@pytest.mark.parametrize("make_cpu_multithreading_loop", [False, True])
@test_utils.test(arch=qd.cpu, offline_cache=False)
def test_block_idx_cpu_scheduling_modes(make_cpu_multithreading_loop):
    # Keep this independent of the existing cache-key bug when switching scheduling modes.
    qd.init(
        arch=qd.cpu,
        cpu_max_num_threads=4,
        make_cpu_multithreading_loop=make_cpu_multithreading_loop,
        offline_cache=False,
    )
    out = qd.ndarray(qd.i32, shape=4096)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1), begin: qd.i32, end: qd.i32) -> qd.i32:
        for i in range(begin, end):
            out[i - begin] = qd.block_idx()
        return qd.block_idx()

    # A nonzero starting index must not shift the block indices.
    assert k_record_blocks(out, -13, 4083) == 0
    # True creates four blocks of 1024 iterations. False creates 128 blocks of 32.
    width = 1024 if make_cpu_multithreading_loop else 32
    np.testing.assert_array_equal(out.to_numpy(), np.arange(4096, dtype=np.int32) // width)


@test_utils.test()
def test_block_idx_serialized_loop():
    out = qd.ndarray(qd.i32, shape=16)

    @qd.kernel
    def k_serial(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(serialize=True)
        for i in range(16):
            out[i] = qd.block_idx()

    out.fill(-1)
    k_serial(out)
    np.testing.assert_array_equal(out.to_numpy(), np.zeros(16, dtype=np.int32))


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu], offline_cache=False)
def test_block_idx_real_func_serial():
    @qd.real_func
    def get_block_idx() -> qd.i32:
        return qd.block_idx()

    @qd.real_func
    def nested_get_block_idx() -> qd.i32:
        return get_block_idx()

    @qd.kernel
    def k_record_serial(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        out[0] = get_block_idx()
        out[1] = nested_get_block_idx()

    out = qd.ndarray(qd.i32, shape=2)
    out.fill(-1)
    k_record_serial(out)
    np.testing.assert_array_equal(out.to_numpy(), np.zeros(2, dtype=np.int32))


@pytest.mark.parametrize("make_cpu_multithreading_loop", [False, True])
@test_utils.test(arch=qd.cpu, offline_cache=False)
def test_block_idx_real_func_cpu_scheduling_modes(make_cpu_multithreading_loop):
    qd.init(
        arch=qd.cpu,
        cpu_max_num_threads=4,
        make_cpu_multithreading_loop=make_cpu_multithreading_loop,
        offline_cache=False,
    )

    @qd.real_func
    def get_block_idx() -> qd.i32:
        return qd.block_idx()

    @qd.real_func
    def nested_get_block_idx() -> qd.i32:
        return get_block_idx()

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=2), begin: qd.i32, end: qd.i32) -> qd.i32:
        for i in range(begin, end):
            out[i - begin, 0] = get_block_idx()
            out[i - begin, 1] = nested_get_block_idx()
        return nested_get_block_idx()

    out = qd.ndarray(qd.i32, shape=(4096, 2))
    out.fill(-1)
    assert k_record_blocks(out, -13, 4083) == 0
    width = 1024 if make_cpu_multithreading_loop else 32
    expected = np.repeat((np.arange(4096, dtype=np.int32) // width)[:, None], 2, axis=1)
    np.testing.assert_array_equal(out.to_numpy(), expected)
