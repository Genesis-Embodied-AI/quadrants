import re

import numpy as np

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu])
def test_ndarray_external_ptr_uses_i64_linear_index():
    @qd.kernel
    def read(arr: qd.types.NDArray[qd.i32, 2], i: qd.i32, j: qd.i32) -> qd.i32:
        return arr[i, j]

    np_arr = np.arange(16, dtype=np.int32).reshape(4, 4)
    assert read(np_arr, 1, 2) == np_arr[1, 2]

    compiled = read._primal._last_compiled_kernel_data
    assert compiled is not None

    llvm_ir = compiled._debug_dump_to_string()

    assert re.search(r"sext i32 .* to i64", llvm_ir)
    assert "mul nsw i64" in llvm_ir
    assert re.search(r"getelementptr i32, ptr .* i64 ", llvm_ir)
