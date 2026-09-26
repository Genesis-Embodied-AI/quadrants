import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@pytest.mark.parametrize("minimum", [1, 16, 512, 2048])
@pytest.mark.parametrize("threads", [1, 4])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_bounds(minimum, threads):
    qd.init(arch=qd.cpu, cpu_max_num_threads=threads, cpu_min_range_for_block=minimum)
    out = qd.field(qd.i32, shape=1100)

    @qd.kernel
    def dynamic(begin: qd.i32, end: qd.i32) -> qd.i32:
        total = 0
        for i in range(begin, end):
            out[i + 20] += 1
            total += i
        return total

    @qd.kernel
    def constant() -> qd.i32:
        total = 0
        for i in range(200):
            out[i + 20] += 1
            total += i
        return total

    for begin, end in [(-7, -7), (9, 3), (-7, -6), (-7, 6), (0, 200), (0, 512), (3, 516), (3, 1028)]:
        out.fill(0)
        assert dynamic(begin, end) == sum(range(begin, end))
        expected = np.zeros(1100, dtype=np.int32)
        if end > begin:
            expected[begin + 20 : end + 20] = 1
        np.testing.assert_array_equal(out.to_numpy(), expected)

    out.fill(0)
    assert constant() == sum(range(200))
    expected = np.zeros(1100, dtype=np.int32)
    expected[20:220] = 1
    np.testing.assert_array_equal(out.to_numpy(), expected)


@pytest.mark.parametrize("serialize", [False, True])
@test_utils.test(arch=qd.cpu, cpu_max_num_threads=4, cpu_min_range_for_block=1)
def test_cpu_range_for_block_serial_execution(serialize):
    @qd.kernel
    def recurrence() -> qd.i32:
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
    assert recurrence() == expected


@pytest.mark.parametrize("minimum", [0, -1])
@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_invalid(minimum):
    with pytest.raises(RuntimeError, match="cpu_min_range_for_block must be positive"):
        qd.init(arch=qd.cpu, cpu_min_range_for_block=minimum)


@test_utils.test(arch=qd.cpu)
def test_cpu_range_for_block_offline_cache(tmp_path):
    @qd.kernel
    def total(n: qd.i32) -> qd.i32:
        result = 0
        for i in range(n):
            result += i
        return result

    def run(minimum):
        qd.init(
            arch=qd.cpu,
            cpu_max_num_threads=4,
            cpu_min_range_for_block=minimum,
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
