"""Phase 2 tests: Tensor as a kernel argument via the dataclass bridge.

Covers:
- A kernel can take a Tensor arg and access ``t.underlying[...]`` and
  ``qd.static(t.layout)``.
- Cache key includes layout: tensors with different layouts compile to
  separate kernel instances; tensors with the same (layout, dtype, ndim)
  share a cache entry.
"""

import numpy as np

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@qd.kernel
def _layout_aware_write(t: qd.Tensor, n_log0: qd.i32, n_log1: qd.i32):
    """Write logical i*10+j into the right physical slot for either layout."""
    for i, j in qd.ndrange(n_log0, n_log1):
        if qd.static(t.layout == (1, 0)):
            t.underlying[j, i] = qd.f32(i * 10 + j)
        else:
            t.underlying[i, j] = qd.f32(i * 10 + j)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_as_kernel_arg_identity_layout():
    t = qd.tensor(qd.f32, shape=(3, 4))
    _layout_aware_write(t, t.shape[0], t.shape[1])
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(t.to_numpy(), expected)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_as_kernel_arg_transposed_layout():
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    _layout_aware_write(t, t.shape[0], t.shape[1])
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(t.to_numpy(), expected)
    # Physical storage is the transpose.
    np.testing.assert_array_equal(t.underlying.to_numpy(), expected.T)


@qd.kernel
def _read_logical_shape(t: qd.Tensor, out: qd.types.ndarray(qd.i32, 1)):
    out[0] = t.underlying.shape[0]
    out[1] = t.underlying.shape[1]


@test_utils.test(arch=get_host_arch_list())
def test_tensor_underlying_shape_visible_inside_kernel():
    """The kernel sees physical shape via t.underlying.shape."""
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    out = qd.ndarray(qd.i32, shape=(2,))
    _read_logical_shape(t, out)
    # Physical shape = (4, 3) when layout=(1, 0) on logical (3, 4).
    assert out[0] == 4
    assert out[1] == 3


@qd.kernel
def _identity_fill(t: qd.Tensor, n_log0: qd.i32, n_log1: qd.i32):
    """Identity-only kernel; no qd.static branching on layout."""
    for i, j in qd.ndrange(n_log0, n_log1):
        t.underlying[i, j] = qd.f32(i * 100 + j)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_kernel_arg_cache_key_distinguishes_layouts():
    """Two tensors with different layouts must produce different cache entries."""
    t_id = qd.tensor(qd.f32, shape=(3, 4))
    t_tr = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))

    # Force compile for each.
    _identity_fill(t_id, 3, 4)
    _identity_fill(t_tr, 3, 4)

    # The same kernel object must now have at least 2 distinct mapping entries.
    n_compiled = len(_identity_fill._primal.mapper.mapping)
    assert n_compiled >= 2, f"expected at least 2 cache entries (one per layout), got {n_compiled}"


@test_utils.test(arch=get_host_arch_list())
def test_tensor_kernel_arg_cache_key_reuses_for_same_layout_and_shape():
    """Two tensors with same dtype/shape/layout must share a cache entry."""

    # Use a fresh kernel to avoid pollution from other tests in the same session.
    @qd.kernel
    def _local(t: qd.Tensor, n_log0: qd.i32, n_log1: qd.i32):
        for i, j in qd.ndrange(n_log0, n_log1):
            t.underlying[i, j] = qd.f32(0.0)

    t_a = qd.tensor(qd.f32, shape=(3, 4))
    t_b = qd.tensor(qd.f32, shape=(3, 4))

    _local(t_a, 3, 4)
    n_after_first = len(_local._primal.mapper.mapping)

    _local(t_b, 3, 4)
    n_after_second = len(_local._primal.mapper.mapping)

    assert n_after_first == n_after_second == 1, (
        f"expected 1 cache entry shared between identical-layout tensors, " f"got {n_after_first} -> {n_after_second}"
    )


@qd.kernel
def _layout_aware_increment(t: qd.Tensor, n_log0: qd.i32, n_log1: qd.i32):
    for i, j in qd.ndrange(n_log0, n_log1):
        if qd.static(t.layout == (1, 0)):
            t.underlying[j, i] += qd.f32(1.0)
        else:
            t.underlying[i, j] += qd.f32(1.0)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_layout_static_branch_works_both_directions():
    """qd.static(t.layout == ...) selects the right physical access at trace time."""
    for layout in [(0, 1), (1, 0)]:
        t = qd.tensor(qd.f32, shape=(2, 3), layout=layout)
        t.from_numpy(np.zeros((2, 3), dtype=np.float32))
        _layout_aware_increment(t, 2, 3)
        np.testing.assert_array_equal(t.to_numpy(), np.ones((2, 3), dtype=np.float32))


@test_utils.test(arch=get_host_arch_list())
def test_tensor_qd_types_alias_exists():
    """qd.types.tensor is exported as an alias to the Tensor class."""
    assert qd.types.tensor is qd.Tensor


@test_utils.test(arch=get_host_arch_list())
def test_tensor_passes_to_kernel_after_from_numpy():
    """Round-trip: from_numpy in python scope, kernel reads via underlying, to_numpy verifies."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)

    @qd.kernel
    def double_in_place(t: qd.Tensor, n_log0: qd.i32, n_log1: qd.i32):
        for i, j in qd.ndrange(n_log0, n_log1):
            if qd.static(t.layout == (1, 0)):
                t.underlying[j, i] *= qd.f32(2.0)
            else:
                t.underlying[i, j] *= qd.f32(2.0)

    for layout in [(0, 1), (1, 0)]:
        t = qd.tensor(qd.f32, shape=(3, 4), layout=layout)
        t.from_numpy(src)
        double_in_place(t, 3, 4)
        np.testing.assert_array_equal(t.to_numpy(), src * 2.0)
