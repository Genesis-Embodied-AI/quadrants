import re
from pathlib import Path

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


@pytest.mark.parametrize("cpu_min_block_size", [1, 512])
def test_cpu_range_for_llvm_dump(cpu_min_block_size, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.init(
        arch=qd.cpu,
        cpu_max_num_threads=4,
        cpu_min_block_size=cpu_min_block_size,
        make_cpu_multithreading_loop=True,
        offline_cache=False,
        debug_dump_path=str(tmp_path),
    )

    try:
        @qd.kernel(fastcache=False)
        def k_record_values(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
            for i in range(200):
                out[i] = i

        out = qd.ndarray(dtype=qd.i32, shape=200)
        k_record_values(out)
        qd.sync()

        # These files contain LLVM IR before CPU optimization. Other .ll files
        # contain Quadrants IR. Restrict the search to this kernel's LLVM dumps.
        llvm_files = list(tmp_path.glob("k_record_values*_llvm.ll"))
        assert llvm_files, f"No LLVM dump for k_record_values in {tmp_path}"
        llvm_ir = "\n".join(path.read_text() for path in llvm_files)

        # Match call instructions, not declarations of the runtime function.
        dispatch_calls = re.findall(r"\bcall void @cpu_parallel_range_for\(([^\n)]*)\)", llvm_ir)
        assert len(dispatch_calls) == 1, dispatch_calls

        # The five i32 arguments are workers, begin, end, step, and block_dim.
        # Remaining arguments are context/function pointers and an i64 size.
        dispatch_values = [int(value) for value in re.findall(r"\bi32\s+(-?\d+)\b", dispatch_calls[0])]
        assert len(dispatch_values) == 5, dispatch_calls[0]
        workers, begin, end, step, block_dim = dispatch_values

        assert workers == 4
        assert (begin, end, step, block_dim) == (0, 4, 1, 1)
        task_count = (end - begin + block_dim - 1) // block_dim
        assert task_count == 4

        # Both minimum settings create four tasks. To prove the setting reaches
        # generated code, also inspect the two block-boundary helper calls.
        boundary_calls = re.findall(r"\bcall i32 @get_cpu_block_start_index\(([^\n)]*)\)", llvm_ir)
        assert len(boundary_calls) == 2, boundary_calls

        for call in boundary_calls:
            arguments = [argument.strip() for argument in call.split(",")]
            assert len(arguments) == 5, call

            # Original range, worker count, and configured minimum.
            # The final argument is a computed block index, not a constant.
            assert arguments[:4] == [
                "i32 0",
                "i32 200",
                "i32 4",
                f"i32 {cpu_min_block_size}",
            ], call

        np.testing.assert_array_equal(out.to_numpy(), np.arange(200, dtype=np.int32))
    finally:
        qd.reset()
