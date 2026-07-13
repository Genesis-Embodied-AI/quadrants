import re

import numpy as np
import psutil
import pytest

import quadrants as qd

from tests import test_utils

# ExternalPtrStmt flattens an N-D ndarray access into a single linear element
# index: for a 2-D array of shape (D0, D1) the codegen emits
#   linear = i * D1 + j
# (see TaskCodeGenLLVM::visit(ExternalPtrStmt) in codegen/llvm/codegen_llvm.cpp).
# Before this fix that accumulation was done in i32 and only sign-extended to
# i64 for the final GEP, so any array with more than 2**31 elements overflowed
# *before* the extend and produced a wrong (often negative) address.
#
# The smallest 2-D shape that pushes the last valid linear index past INT32_MAX:
#   shape = (2, 2**30 + 1)  ->  last index [1, 2**30]  ->  linear = 2**31 + 1
_D1 = 2**30 + 1
_I = 1
_J = 2**30
_TRUE_LINEAR = _I * _D1 + _J  # == 2**31 + 1, exceeds np.iinfo(np.int32).max


@test_utils.test(arch=[qd.cpu])
def test_i32_linear_index_overflows_but_i64_is_correct():
    # Reproduces the exact arithmetic ExternalPtrStmt performs. The i32 kernel
    # mirrors the pre-fix codegen and must wrap to a wrong value; the i64 kernel
    # mirrors the fix and must stay correct.
    @qd.kernel
    def linear_i32(d1: qd.i32, i: qd.i32, j: qd.i32) -> qd.i32:
        return i * d1 + j

    @qd.kernel
    def linear_i64(d1: qd.i64, i: qd.i64, j: qd.i64) -> qd.i64:
        return i * d1 + j

    assert _TRUE_LINEAR > np.iinfo(np.int32).max

    # Pre-fix behavior: i32 accumulation overflows and wraps to a negative offset.
    overflowed = linear_i32(_D1, _I, _J)
    assert overflowed != _TRUE_LINEAR
    assert overflowed < 0

    # Post-fix behavior: i64 accumulation yields the true linear index.
    assert linear_i64(_D1, _I, _J) == _TRUE_LINEAR


# ~2 GB for the int8 backing array plus headroom for the device-side copy.
_REQUIRED_BYTES = 5 * 1024**3


@pytest.mark.skipif(
    psutil.virtual_memory().available < _REQUIRED_BYTES,
    reason="needs >5 GB RAM to allocate an ndarray with more than 2**31 elements",
)
@test_utils.test(arch=[qd.cpu])
def test_ndarray_read_past_int32_index_boundary():
    # End-to-end regression guard: on the pre-fix codegen the i32 linear index
    # for [1, 2**30] wraps negative and this read returns garbage / segfaults.
    @qd.kernel
    def read(arr: qd.types.NDArray[qd.i8, 2], i: qd.i32, j: qd.i32) -> qd.i8:
        return arr[i, j]

    np_arr = np.zeros((2, _D1), dtype=np.int8)
    sentinel = np.int8(7)
    np_arr[_I, _J] = sentinel

    assert read(np_arr, _I, _J) == sentinel


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
