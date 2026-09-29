import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("cpu_min_block_size", [1, 16, 512, 2048])
@pytest.mark.parametrize("threads", [1, 4])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_bounds(cpu_min_block_size, threads):
    qd.init(arch=qd.cpu, cpu_max_num_threads=threads, cpu_min_block_size=cpu_min_block_size)
    out = qd.field(qd.i32, shape=1100)

    @qd.kernel
    def k_dynamic_range(begin: qd.i32, end: qd.i32) -> qd.i32:
        total = 0
        for i in range(begin, end):
            out[i + 20] += 1
            total += i
        return total

    for begin, end in [(-7, -7), (9, 3), (-7, -6), (-7, 6), (0, 200), (0, 512), (3, 516), (3, 1028)]:
        out.fill(0)
        assert k_dynamic_range(begin, end) == sum(range(begin, end))
        expected = np.zeros(1100, dtype=np.int32)
        if end > begin:
            expected[begin + 20 : end + 20] = 1
        np.testing.assert_array_equal(out.to_numpy(), expected)

    @qd.kernel
    def k_constant_range() -> qd.i32:
        total = 0
        for i in range(200):
            out[i + 20] += 1
            total += i
        return total

    out.fill(0)
    assert k_constant_range() == sum(range(200))
    expected = np.zeros(1100, dtype=np.int32)
    expected[20:220] = 1
    np.testing.assert_array_equal(out.to_numpy(), expected)


@pytest.mark.parametrize("serialize", [False, True])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, cpu_min_block_size=1)
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


@pytest.mark.parametrize("cpu_min_block_size", [0, -1, -512])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_invalid(cpu_min_block_size):
    with pytest.raises(RuntimeError, match=rf"cpu_min_block_size must be >= 1, but got {cpu_min_block_size}\."):
        qd.init(arch=qd.cpu, cpu_min_block_size=cpu_min_block_size)


@pytest.mark.parametrize("cpu_min_block_size", [512, 1 << 30, (1 << 31) - 1])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_overflow(cpu_min_block_size):
    qd.init(arch=qd.cpu, cpu_max_num_threads=4, cpu_min_block_size=cpu_min_block_size)
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
    "cpu_min_block_size, expected_sizes",
    [
        (1, [50, 50, 50, 50]),
        (16, [50, 50, 50, 50]),
        (64, [64, 64, 64, 8]),
        (512, [200, 0, 0, 0]),
    ],
)
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_sizes(cpu_min_block_size, expected_sizes):
    qd.init(
        arch=qd.cpu, cpu_max_num_threads=4, cpu_min_block_size=cpu_min_block_size, make_cpu_multithreading_loop=True
    )
    block_ids = qd.ndarray(dtype=qd.i32, shape=200)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        for i in range(200):
            out[i] = qd.cpu_block_id()

    k_record_blocks(block_ids)
    # Check which block executed every iteration, including exact block sizes.
    expected = np.repeat(np.arange(4, dtype=np.int32), expected_sizes)
    np.testing.assert_array_equal(block_ids.to_numpy(), expected)


@pytest.mark.parametrize("block_dim", [1, 16, 32, 64, 512])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, make_cpu_multithreading_loop=False)
def test_cpu_range_for_fixed_block_sizes(block_dim):
    block_ids = qd.ndarray(dtype=qd.i32, shape=200)

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=block_dim)
        for i in range(200):
            out[i] = qd.cpu_block_id()

    k_record_blocks(block_ids)
    np.testing.assert_array_equal(block_ids.to_numpy(), np.arange(200, dtype=np.int32) // block_dim)


@pytest.mark.parametrize("make_cpu_multithreading_loop", [False, True])
@test_utils.test(arch=qd.cpu)
def test_cpu_block_id_serial_and_nested(make_cpu_multithreading_loop):
    qd.init(
        arch=qd.cpu,
        cpu_max_num_threads=4,
        cpu_min_block_size=1,
        make_cpu_multithreading_loop=make_cpu_multithreading_loop,
    )
    block_ids = qd.ndarray(dtype=qd.i32, shape=(200, 3))

    @qd.kernel
    def k_record_blocks(out: qd.types.ndarray(dtype=qd.i32, ndim=2)) -> qd.i32:
        for i in range(200):
            for j in range(3):
                out[i, j] = qd.cpu_block_id()
        return qd.cpu_block_id()

    assert k_record_blocks(block_ids) == -1
    width = 50 if make_cpu_multithreading_loop else 32
    expected = np.repeat((np.arange(200, dtype=np.int32) // width)[:, None], 3, axis=1)
    np.testing.assert_array_equal(block_ids.to_numpy(), expected)

    @qd.kernel
    def k_serial() -> qd.i32:
        result = 0
        qd.loop_config(serialize=True)
        for i in range(10):
            result += qd.cpu_block_id()
        return result

    assert k_serial() == -10


@test_utils.test(arch=[qd.cuda, qd.amdgpu, qd.vulkan, qd.metal])
def test_cpu_block_id_rejects_gpu():
    @qd.kernel
    def k_block_id() -> qd.i32:
        return qd.cpu_block_id()

    with pytest.raises(ValueError, match=r"qd.cpu_block_id\(\) is only supported on CPU"):
        k_block_id()
