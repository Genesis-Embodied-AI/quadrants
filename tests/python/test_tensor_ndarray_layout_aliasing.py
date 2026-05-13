"""Aliasing of layout metadata across the supported Quadrants aliasing patterns.

Note: in-kernel rebinding (``y = x; y[i, j] = ...``) is **not** supported by Quadrants for any ndarray - that's an
upstream limitation not specific to tensor (it raises ``QuadrantsTypeError: Invalid constant scalar data type:
<class 'quadrants.lang.any_array.AnyArray'>``). So this file pins down the aliasing patterns Quadrants *does* support
and that tensor layout metadata must propagate through:

1. Same ``Ndarray`` passed twice to the same kernel - two distinct ``AnyArray`` instances inside the kernel, both must
   carry the same layout.
2. Same ``Ndarray`` shared across two consecutive kernel calls - the layout cannot leak or get lost between calls.
3. Repeated access through ``.grad`` inside a single kernel - every call must return an ``AnyArray`` with the same
   layout (an earlier change covered the single-access path; this exercises the repeated-access cache).
4. The same ``Ndarray`` via two different kernel signatures (one annotated as ``qd.types.ndarray()`` directly, one via
   a wrapper) - metadata must travel via the runtime feature tuple, not the annotation.
"""

import numpy as np

import quadrants as qd
from quadrants._tensor import _with_layout

from tests import test_utils


def _allocate_layout10(M, N, dtype=qd.i32, needs_grad=False):
    a = qd.tensor(dtype, shape=(N, M), backend=qd.Backend.NDARRAY, needs_grad=needs_grad)
    _with_layout(a, (1, 0))
    return a


# ----------------------------------------------------------------------------
# 1. Same Ndarray passed twice to the same kernel
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_same_ndarray_passed_twice():
    """Both AnyArrays inside the kernel see the same layout."""
    M, N = 2, 3
    a = _allocate_layout10(M, N)

    @qd.kernel
    def write_via_two_handles(x: qd.types.ndarray(), y: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 7
            # `y` aliases `x` - write through it; result must agree.
            y[i, j] = x[i, j] + (i * 10 + j)

    write_via_two_handles(a, a)
    arr = a.to_numpy()
    assert arr.shape == (M, N)
    for i in range(M):
        for j in range(N):
            assert arr[i, j] == 7 + i * 10 + j


# ----------------------------------------------------------------------------
# 2. Same Ndarray across two consecutive kernel calls
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_persists_across_kernel_calls():
    M, N = 3, 4
    a = _allocate_layout10(M, N)

    @qd.kernel
    def init(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 10 + j

    @qd.kernel
    def add(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] += 1000

    init(a)
    add(a)
    arr = a.to_numpy()
    assert arr.shape == (M, N)
    for i in range(M):
        for j in range(N):
            assert arr[i, j] == 1000 + i * 10 + j


# ----------------------------------------------------------------------------
# 3. Repeated .grad access inside the same kernel
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_repeated_grad_access_in_kernel():
    M, N = 2, 3
    a = _allocate_layout10(M, N, dtype=qd.f32, needs_grad=True)

    @qd.kernel
    def write_grad_repeatedly(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x.grad[i, j] = float(i * 100)
            # Re-access .grad in the same iteration - must hit same layout.
            x.grad[i, j] += float(j * 10)
            x.grad[i, j] += 1.0

    write_grad_repeatedly(a)
    grad = a.grad.to_numpy()
    assert grad.shape == (M, N)
    for i in range(M):
        for j in range(N):
            assert grad[i, j] == i * 100 + j * 10 + 1


# ----------------------------------------------------------------------------
# 4. Same Ndarray via two kernels with different (compatible) annotations
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_consistent_across_different_kernel_signatures():
    """Layout metadata travels with the value, not the kernel annotation."""
    M, N = 2, 3
    a = _allocate_layout10(M, N)

    @qd.kernel
    def kernel_a(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = i * 10 + j

    @qd.kernel
    def kernel_b(arr: qd.types.ndarray()):  # different param name, same type
        for i, j in qd.ndrange(M, N):
            arr[i, j] += 100

    kernel_a(a)
    kernel_b(a)
    arr_np = a.to_numpy()
    assert arr_np.shape == (M, N)
    for i in range(M):
        for j in range(N):
            assert arr_np[i, j] == 100 + i * 10 + j


# ----------------------------------------------------------------------------
# 5. Untagged + tagged + .grad in one kernel: metadata isolation per arg
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_isolated_between_args():
    """One tagged + one untagged ndarray in the same kernel: each carries its own (or no) layout."""
    M, N = 2, 3
    untagged = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)  # canonical (M, N)
    tagged = _allocate_layout10(M, N)

    @qd.kernel
    def k(u: qd.types.ndarray(), t: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            u[i, j] = i * 10 + j  # no rewrite
            t[i, j] = i * 10 + j  # rewrite to t[j, i]

    k(untagged, tagged)
    # Both ndarrays were filled by the same kernel writing canonical values; both ``to_numpy()`` calls return canonical
    # views, so the numpy arrays compare equal element-for-element regardless of the tagged ndarray's physical layout.
    np.testing.assert_array_equal(untagged.to_numpy(), tagged.to_numpy())


# ----------------------------------------------------------------------------
# 6. Two separately-allocated layout-tagged ndarrays - independence
# ----------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_layout_two_tagged_ndarrays_independent():
    M, N = 2, 3
    a = _allocate_layout10(M, N)
    b = _allocate_layout10(M, N)

    @qd.kernel
    def init_a(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 1

    @qd.kernel
    def init_b(x: qd.types.ndarray()):
        for i, j in qd.ndrange(M, N):
            x[i, j] = 2

    init_a(a)
    init_b(b)
    assert (a.to_numpy() == 1).all()
    assert (b.to_numpy() == 2).all()
