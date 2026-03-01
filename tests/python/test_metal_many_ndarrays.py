"""Test that Metal can handle dataclasses with many ndarray fields.

Metal has a 31 buffer binding limit per compute kernel. With physical storage
buffer support, ndarray addresses are packed into the args buffer as u64
pointers rather than each consuming a separate buffer binding. This test
verifies we can exceed 31 ndarray fields in a single dataclass on Metal.
"""

import dataclasses

import numpy as np

import quadrants as qd

from tests import test_utils


@dataclasses.dataclass(frozen=True)
class ThirtyFiveArrays:
    a0: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a1: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a2: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a3: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a4: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a5: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a6: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a7: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a8: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a9: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a10: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a11: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a12: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a13: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a14: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a15: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a16: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a17: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a18: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a19: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a20: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a21: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a22: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a23: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a24: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a25: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a26: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a27: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a28: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a29: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a30: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a31: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a32: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a33: qd.types.ndarray(dtype=qd.f32, ndim=1)
    a34: qd.types.ndarray(dtype=qd.f32, ndim=1)


@qd.kernel
def sum_35_arrays(s: ThirtyFiveArrays, n: qd.i32):
    for i in range(n):
        total = qd.f32(0.0)
        total += s.a0[i]
        total += s.a1[i]
        total += s.a2[i]
        total += s.a3[i]
        total += s.a4[i]
        total += s.a5[i]
        total += s.a6[i]
        total += s.a7[i]
        total += s.a8[i]
        total += s.a9[i]
        total += s.a10[i]
        total += s.a11[i]
        total += s.a12[i]
        total += s.a13[i]
        total += s.a14[i]
        total += s.a15[i]
        total += s.a16[i]
        total += s.a17[i]
        total += s.a18[i]
        total += s.a19[i]
        total += s.a20[i]
        total += s.a21[i]
        total += s.a22[i]
        total += s.a23[i]
        total += s.a24[i]
        total += s.a25[i]
        total += s.a26[i]
        total += s.a27[i]
        total += s.a28[i]
        total += s.a29[i]
        total += s.a30[i]
        total += s.a31[i]
        total += s.a32[i]
        total += s.a33[i]
        total += s.a34[i]
        s.a0[i] = total


@qd.kernel
def write_and_read_back(s: ThirtyFiveArrays, n: qd.i32):
    for i in range(n):
        s.a0[i] = 1.0
        s.a17[i] = 2.0
        s.a34[i] = s.a0[i] + s.a17[i]


@test_utils.test(arch=qd.metal)
def test_dataclass_35_ndarrays_sum():
    n = 16
    arrays = {f"a{i}": qd.ndarray(dtype=qd.f32, shape=(n,)) for i in range(35)}
    for i in range(35):
        arrays[f"a{i}"].from_numpy(np.full(n, float(i), dtype=np.float32))

    s = ThirtyFiveArrays(**arrays)
    sum_35_arrays(s, n)
    qd.sync()

    result = s.a0.to_numpy()
    expected = sum(range(35))  # 0+1+...+34 = 595
    np.testing.assert_allclose(result, expected, rtol=1e-5)


@test_utils.test(arch=qd.metal)
def test_dataclass_35_ndarrays_write_read():
    n = 16
    arrays = {f"a{i}": qd.ndarray(dtype=qd.f32, shape=(n,)) for i in range(35)}
    s = ThirtyFiveArrays(**arrays)
    write_and_read_back(s, n)
    qd.sync()

    np.testing.assert_allclose(s.a0.to_numpy(), 1.0)
    np.testing.assert_allclose(s.a17.to_numpy(), 2.0)
    np.testing.assert_allclose(s.a34.to_numpy(), 3.0)
