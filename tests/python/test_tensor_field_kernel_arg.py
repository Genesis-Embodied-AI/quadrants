"""Phase 2b tests: FieldTensor as a kernel argument via the dataclass bridge.

Mirrors ``test_tensor_kernel_arg.py`` but exercises the field storage
backend. The dataclass bridge dispatches on the parameter annotation, so
kernels here pin the type as ``qd.FieldTensor`` (not ``qd.Tensor``).

Covers:
- A kernel can take a FieldTensor arg and access ``t.underlying[...]`` and
  ``qd.static(t.layout)``.
- Cache key includes layout: tensors with different layouts compile to
  separate kernel instances; tensors with the same layout share an entry.
- ``qd.types.field_tensor`` is the type-annotation alias for FieldTensor.
- Round-trip via ``from_numpy`` then kernel-side write.
"""

import numpy as np

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

# ---------------------------------------------------------------------------
# Basic kernel-arg path: FieldTensor is bridged as a template-bound field
# inside a dataclass.
# ---------------------------------------------------------------------------


@qd.kernel
def _layout_aware_write_field(t: qd.FieldTensor, n_log0: qd.i32, n_log1: qd.i32):
    """Write logical i*10+j to the right physical slot for either layout."""
    for i, j in qd.ndrange(n_log0, n_log1):
        if qd.static(t.layout == (1, 0)):
            t.underlying[j, i] = qd.f32(i * 10 + j)
        else:
            t.underlying[i, j] = qd.f32(i * 10 + j)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_as_kernel_arg_identity_layout():
    t = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD)
    _layout_aware_write_field(t, t.shape[0], t.shape[1])
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(t.to_numpy(), expected)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_as_kernel_arg_transposed_layout():
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0), backend=qd.Backend.FIELD)
    _layout_aware_write_field(t, t.shape[0], t.shape[1])
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(t.to_numpy(), expected)
    np.testing.assert_array_equal(t.underlying.to_numpy(), expected.T)


@qd.kernel
def _read_logical_shape_field(t: qd.FieldTensor, out: qd.types.ndarray(qd.i32, 1)):
    out[0] = t.underlying.shape[0]
    out[1] = t.underlying.shape[1]


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_underlying_shape_visible_inside_kernel():
    """Kernel sees physical shape via t.underlying.shape on a field-backed tensor."""
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0), backend=qd.Backend.FIELD)
    out = qd.ndarray(qd.i32, shape=(2,))
    _read_logical_shape_field(t, out)
    assert out[0] == 4
    assert out[1] == 3


# ---------------------------------------------------------------------------
# fastcache: layout participates in the cache key.
# ---------------------------------------------------------------------------


@qd.kernel
def _identity_fill_field(t: qd.FieldTensor, n_log0: qd.i32, n_log1: qd.i32):
    """Identity-only kernel; no qd.static branching on layout."""
    for i, j in qd.ndrange(n_log0, n_log1):
        t.underlying[i, j] = qd.f32(i * 100 + j)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_kernel_arg_cache_key_distinguishes_layouts():
    """Two field-backed tensors with different layouts must produce different cache entries."""
    t_id = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD)
    t_tr = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0), backend=qd.Backend.FIELD)

    _identity_fill_field(t_id, 3, 4)
    _identity_fill_field(t_tr, 3, 4)

    n_compiled = len(_identity_fill_field._primal.mapper.mapping)
    assert n_compiled >= 2, f"expected at least 2 cache entries (one per layout), got {n_compiled}"


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_kernel_arg_cache_key_reuses_for_same_layout():
    """Two field-backed tensors with the same layout share a cache entry.

    Note: fields are bound by-reference via ``qd.template`` and the template
    path keys on object identity for templates *plus* on layout (the second
    field). So repeated calls with the *same* tensor must reuse, but distinct
    tensors with identical (dtype, shape, layout) may or may not — we only
    assert the within-instance reuse, which is what matters for hot loops.
    """

    @qd.kernel
    def _local_field(t: qd.FieldTensor, n_log0: qd.i32, n_log1: qd.i32):
        for i, j in qd.ndrange(n_log0, n_log1):
            t.underlying[i, j] = qd.f32(0.0)

    t = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD)

    _local_field(t, 3, 4)
    n_after_first = len(_local_field._primal.mapper.mapping)
    _local_field(t, 3, 4)
    n_after_second = len(_local_field._primal.mapper.mapping)

    assert n_after_first == n_after_second == 1, (
        f"expected 1 cache entry on repeated call with same field-backed tensor, "
        f"got {n_after_first} -> {n_after_second}"
    )


# ---------------------------------------------------------------------------
# qd.static layout branching, augmented assignment, round-trip.
# ---------------------------------------------------------------------------


@qd.kernel
def _layout_aware_increment_field(t: qd.FieldTensor, n_log0: qd.i32, n_log1: qd.i32):
    for i, j in qd.ndrange(n_log0, n_log1):
        if qd.static(t.layout == (1, 0)):
            t.underlying[j, i] += qd.f32(1.0)
        else:
            t.underlying[i, j] += qd.f32(1.0)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_layout_static_branch_works_both_directions():
    """qd.static(t.layout == ...) selects the right physical access at trace time on field backend."""
    for layout in [(0, 1), (1, 0)]:
        t = qd.tensor(qd.f32, shape=(2, 3), layout=layout, backend=qd.Backend.FIELD)
        t.from_numpy(np.zeros((2, 3), dtype=np.float32))
        _layout_aware_increment_field(t, 2, 3)
        np.testing.assert_array_equal(t.to_numpy(), np.ones((2, 3), dtype=np.float32))


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_passes_to_kernel_after_from_numpy():
    """Round-trip: from_numpy in python scope, kernel doubles via underlying, to_numpy verifies."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)

    @qd.kernel
    def double_in_place_field(t: qd.FieldTensor, n_log0: qd.i32, n_log1: qd.i32):
        for i, j in qd.ndrange(n_log0, n_log1):
            if qd.static(t.layout == (1, 0)):
                t.underlying[j, i] *= qd.f32(2.0)
            else:
                t.underlying[i, j] *= qd.f32(2.0)

    for layout in [(0, 1), (1, 0)]:
        t = qd.tensor(qd.f32, shape=(3, 4), layout=layout, backend=qd.Backend.FIELD)
        t.from_numpy(src)
        double_in_place_field(t, 3, 4)
        np.testing.assert_array_equal(t.to_numpy(), src * 2.0)


# ---------------------------------------------------------------------------
# Type-annotation alias.
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_qd_types_field_tensor_alias_exists():
    """qd.types.field_tensor is exported as an alias to the FieldTensor class."""
    assert qd.types.field_tensor is qd.FieldTensor


# ---------------------------------------------------------------------------
# Cross-backend separation: NdarrayTensor and FieldTensor compile to
# different cache entries when used in matching kernels.
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_field_and_ndarray_kernels_are_separate_signatures():
    """Two kernels with identical bodies but different backend annotations
    produce independent compilations — proving the dataclass bridge picks up
    the per-variant ``underlying`` annotation type correctly.
    """

    @qd.kernel
    def fill_nd(t: qd.NdarrayTensor, n0: qd.i32, n1: qd.i32):
        for i, j in qd.ndrange(n0, n1):
            t.underlying[i, j] = qd.f32(1.0)

    @qd.kernel
    def fill_fd(t: qd.FieldTensor, n0: qd.i32, n1: qd.i32):
        for i, j in qd.ndrange(n0, n1):
            t.underlying[i, j] = qd.f32(1.0)

    t_nd = qd.tensor(qd.f32, shape=(2, 3))
    t_fd = qd.tensor(qd.f32, shape=(2, 3), backend=qd.Backend.FIELD)
    fill_nd(t_nd, 2, 3)
    fill_fd(t_fd, 2, 3)
    np.testing.assert_array_equal(t_nd.to_numpy(), np.ones((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(t_fd.to_numpy(), np.ones((2, 3), dtype=np.float32))
