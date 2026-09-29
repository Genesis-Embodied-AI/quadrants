import re
from pathlib import Path

import numpy as np
import pytest

import quadrants as qd


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
