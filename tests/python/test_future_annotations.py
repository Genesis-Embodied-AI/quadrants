"""Test that kernels work with `from __future__ import annotations` (PEP 563)."""

from __future__ import annotations

import quadrants as qd

from tests import test_utils


@qd.kernel
def add_kernel(a: qd.types.NDArray[qd.i32, 1], b: qd.types.NDArray[qd.i32, 1]) -> None:
    for i in a:
        a[i] = a[i] + b[i]


@test_utils.test()
def test_future_annotations_kernel():
    a = qd.ndarray(qd.i32, (4,))
    b = qd.ndarray(qd.i32, (4,))
    for i in range(4):
        a[i] = i
        b[i] = 10
    add_kernel(a, b)
    for i in range(4):
        assert a[i] == i + 10
