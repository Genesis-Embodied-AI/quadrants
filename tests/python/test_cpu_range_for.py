import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("cpu_per_worker_min_block_dim", [1, 2, 4, 16, 512, 2048])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("constant_bounds", [False, True])
@pytest.mark.parametrize(
    "begin,end,expected_total,expected_visits",
    [
        # expected_visits counts how often each value of i should occur.
        # For example, range(1, 4) should run with i = 1, 2, and 3, once each:
        # Value of i:       -2, -1, 0, 1, 2, 3, 4, 5
        # Expected count:    0,  0, 0, 1, 1, 1, 0, 0
        # Comparing these counts checks that parallel scheduling does not skip an iteration or run it twice.
        # The other parameterized values (block size, worker count, and constant bounds) should not affect these results.
        (-2, -2, 0, [0, 0, 0, 0, 0, 0, 0, 0]),
        (3, 1, 0, [0, 0, 0, 0, 0, 0, 0, 0]),
        (-2, -1, -2, [1, 0, 0, 0, 0, 0, 0, 0]),
        (-2, 3, 0, [1, 1, 1, 1, 1, 0, 0, 0]),
        (0, 4, 6, [0, 0, 1, 1, 1, 1, 0, 0]),
        (1, 6, 15, [0, 0, 0, 1, 1, 1, 1, 1]),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_bounds(
    cpu_per_worker_min_block_dim, threads, constant_bounds, begin, end, expected_total, expected_visits
):
    qd.init(arch=qd.cpu, cpu_max_num_threads=threads, cpu_per_worker_min_block_dim=cpu_per_worker_min_block_dim)
    out = qd.field(qd.i32, shape=8)

    @qd.kernel
    def k_dynamic_range(begin: qd.i32, end: qd.i32) -> qd.i32:
        total = 0
        for i in range(begin, end):
            out[i + 2] += 1
            total += i
        return total

    @qd.kernel
    def k_constant_range(begin: qd.template(), end: qd.template()) -> qd.i32:
        total = 0
        for i in range(begin, end):
            out[i + 2] += 1
            total += i
        return total

    kernel = k_constant_range if constant_bounds else k_dynamic_range
    out.fill(0)
    assert kernel(begin, end) == expected_total
    np.testing.assert_array_equal(out.to_numpy(), expected_visits)


@pytest.mark.parametrize("serialize", [False, True])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, cpu_per_worker_min_block_dim=1)
def test_cpu_range_for_block_serial_execution(serialize):
    @qd.kernel
    def k_serial() -> qd.i32:
        result = 0
        if qd.static(serialize):
            qd.loop_config(serialize=True)
        else:
            qd.loop_config(parallelize=1)
        for i in range(200):
            result = (result * 3 + i) % 10007
        return result

    expected = 0
    for i in range(200):
        expected = (expected * 3 + i) % 10007
    assert k_serial() == expected


@pytest.mark.parametrize("cpu_per_worker_min_block_dim", [0, -1, -512])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_invalid(cpu_per_worker_min_block_dim):
    with pytest.raises(
        RuntimeError, match=rf"cpu_per_worker_min_block_dim must be >= 1, but got {cpu_per_worker_min_block_dim}\."
    ):
        qd.init(arch=qd.cpu, cpu_per_worker_min_block_dim=cpu_per_worker_min_block_dim)


@pytest.mark.parametrize("cpu_per_worker_min_block_dim", [512, 1 << 30, (1 << 31) - 1])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_overflow(cpu_per_worker_min_block_dim):
    qd.init(arch=qd.cpu, cpu_max_num_threads=4, cpu_per_worker_min_block_dim=cpu_per_worker_min_block_dim)
    # Each nonempty test range has seven iterations. Reserve the final slot for invalid visits.
    INVALID_VISIT_INDEX = 15
    out = qd.field(qd.i32, shape=INVALID_VISIT_INDEX + 1)

    @qd.kernel
    def k_dynamic_range(begin: qd.i32, end: qd.i32):
        for i in range(begin, end):
            if begin <= i < end:
                out[i - begin] += 1
            else:
                qd.atomic_or(out[INVALID_VISIT_INDEX], 1)

    @qd.kernel
    def k_constant_range(begin: qd.template(), end: qd.template()):
        for i in range(begin, end):
            if begin <= i < end:
                out[i - begin] += 1
            else:
                qd.atomic_or(out[INVALID_VISIT_INDEX], 1)

    max_i32 = (1 << 31) - 1
    min_i32 = -(1 << 31)
    for begin, end in [(0, 7), (1, 1), (max_i32 - 7, max_i32), (min_i32, min_i32 + 7), (max_i32, min_i32)]:
        expected = np.zeros(INVALID_VISIT_INDEX + 1, dtype=np.int32)
        expected[: max(0, end - begin)] = 1
        for kernel in (k_dynamic_range, k_constant_range):
            out.fill(0)
            kernel(begin, end)
            np.testing.assert_array_equal(out.to_numpy(), expected)


@pytest.mark.parametrize(
    "cpu_per_worker_min_block_dim, expected_sizes",
    [
        (1, [50, 50, 50, 50]),
        (16, [50, 50, 50, 50]),
        (64, [64, 64, 64, 8]),
        (512, [200, 0, 0, 0]),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_sizes(cpu_per_worker_min_block_dim, expected_sizes):
    qd.init(
        arch=qd.cpu,
        cpu_max_num_threads=4,
        cpu_per_worker_min_block_dim=cpu_per_worker_min_block_dim,
        cpu_work_scheduling=qd.CPUWorkScheduling.PER_WORKER,
    )
    block_indices = qd.ndarray(dtype=qd.i32, shape=200)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(200):
            out[i] = qd.block_idx()

    k_record_blocks(block_indices)
    # Check which block executed every iteration, including exact block sizes.
    expected = np.repeat(np.arange(4, dtype=np.int32), expected_sizes)
    np.testing.assert_array_equal(block_indices.to_numpy(), expected)


@pytest.mark.parametrize("block_dim", [1, 16, 32, 64, 512])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, cpu_work_scheduling=qd.CPUWorkScheduling.FIXED_SIZE)
def test_cpu_range_for_fixed_block_sizes(block_dim):
    block_indices = qd.ndarray(dtype=qd.i32, shape=200)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=block_dim)
        for i in range(200):
            out[i] = qd.block_idx()

    k_record_blocks(block_indices)
    np.testing.assert_array_equal(block_indices.to_numpy(), np.arange(200, dtype=np.int32) // block_dim)


@pytest.mark.parametrize("cpu_work_scheduling", [qd.CPUWorkScheduling.FIXED_SIZE, qd.CPUWorkScheduling.PER_WORKER])
@test_utils.test(arch=qd.cpu)
def test_cpu_block_idx_serial_and_nested(cpu_work_scheduling):
    qd.init(
        arch=qd.cpu,
        cpu_max_num_threads=4,
        cpu_per_worker_min_block_dim=1,
        cpu_work_scheduling=cpu_work_scheduling,
    )
    block_indices = qd.ndarray(dtype=qd.i32, shape=(200, 3))

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=2)) -> qd.i32:
        for i in range(200):
            for j in range(3):
                out[i, j] = qd.block_idx()
        return qd.block_idx()

    assert k_record_blocks(block_indices) == 0
    width = 50 if cpu_work_scheduling == qd.CPUWorkScheduling.PER_WORKER else 32
    expected = np.repeat((np.arange(200, dtype=np.int32) // width)[:, None], 3, axis=1)
    np.testing.assert_array_equal(block_indices.to_numpy(), expected)

    @qd.kernel
    def k_serial() -> qd.i32:
        result = 0
        qd.loop_config(serialize=True)
        for i in range(10):
            result += qd.block_idx()
        return result

    assert k_serial() == 0
