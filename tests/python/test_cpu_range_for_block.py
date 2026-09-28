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


@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_offline_cache(tmp_path):
    @qd.kernel
    def total(n: qd.i32) -> qd.i32:
        result = 0
        for i in range(n):
            result += i
        return result

    def run(cpu_min_block_size):
        qd.init(
            arch=qd.cpu,
            cpu_max_num_threads=4,
            cpu_min_block_size=cpu_min_block_size,
            offline_cache=True,
            offline_cache_file_path=str(tmp_path),
            offline_cache_cleaning_policy="never",
        )
        assert total(200) == 19900
        qd.reset()
        return {path.name for path in tmp_path.rglob("*.qdc")}

    default_files = run(512)
    assert default_files
    small_files = run(1)
    assert default_files < small_files
    assert run(512) == small_files
    assert run(1) == small_files


@pytest.mark.parametrize("cpu_min_block_size", [512, 1 << 30, (1 << 31) - 1])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_overflow(cpu_min_block_size):
    qd.init(arch=qd.cpu, cpu_max_num_threads=4, cpu_min_block_size=cpu_min_block_size, debug=True)
    out = qd.field(qd.i32, shape=16)

    @qd.kernel
    def dynamic(begin: qd.i32, end: qd.i32):
        for i in range(begin, end):
            # Fail promptly if an unused chunk wraps around to a negative start.
            assert i >= begin and i < end
            out[i - begin] += 1

    @qd.kernel
    def constant(begin: qd.template(), end: qd.template()):
        for i in range(begin, end):
            assert i >= begin and i < end
            out[i - begin] += 1

    max_i32 = (1 << 31) - 1
    min_i32 = -(1 << 31)
    for begin, end in [(0, 7), (1, 1), (max_i32 - 7, max_i32), (min_i32, min_i32 + 7), (max_i32, min_i32)]:
        expected = np.zeros(16, dtype=np.int32)
        expected[: max(0, end - begin)] = 1
        for kernel in (dynamic, constant):
            out.fill(0)
            kernel(begin, end)
            np.testing.assert_array_equal(out.to_numpy(), expected)
