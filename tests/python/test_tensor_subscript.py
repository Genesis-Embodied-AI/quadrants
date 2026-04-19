"""Phase 3 tests: bare ``t[i, j]`` subscript inside a kernel.

Phase 3 ships an AST pre-pass that rewrites tensor subscripts so users can
write logical indexing directly inside kernels:

    @qd.kernel
    def k(t: qd.Tensor, n0, n1):
        for i, j in qd.ndrange(n0, n1):
            t[i, j] = ...   # was: t.underlying[j, i] for layout=(1, 0)

Coverage:
- Read / write / augmented assignment.
- 1D / 2D / 3D layouts; identity and non-identity permutations.
- Both backends (NdarrayTensor + FieldTensor).
- ``qd.func`` callee with a tensor parameter.
- The pass leaves non-tensor subscripts (e.g. ``ndarray[i, j]``,
  ``vec[k]``) untouched — including when both a tensor and a non-tensor
  parameter sit in the same kernel.
"""

import numpy as np

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


# ---------------------------------------------------------------------------
# Read / write — both layouts, both backends.
# ---------------------------------------------------------------------------


@qd.kernel
def _write_logical(t: qd.Tensor, n0: qd.i32, n1: qd.i32):
    """Write logical i*10+j via bare subscript syntax."""
    for i, j in qd.ndrange(n0, n1):
        t[i, j] = qd.f32(i * 10 + j)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_write_identity_layout():
    t = qd.tensor(qd.f32, shape=(3, 4))
    _write_logical(t, 3, 4)
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(t.to_numpy(), expected)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_write_transposed_layout():
    """Bare t[i, j] write on a transposed tensor lands at the right physical slot."""
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    _write_logical(t, 3, 4)
    expected = np.fromfunction(lambda i, j: i * 10 + j, (3, 4), dtype=np.float32)
    # Logical view round-trips.
    np.testing.assert_array_equal(t.to_numpy(), expected)
    # Physical storage is the transpose.
    np.testing.assert_array_equal(t.underlying.to_numpy(), expected.T)


@qd.kernel
def _read_sum(t: qd.Tensor, n0: qd.i32, n1: qd.i32, out: qd.types.ndarray(qd.f32, 1)):
    """Read every cell via bare subscript and accumulate."""
    s = qd.f32(0.0)
    for i, j in qd.ndrange(n0, n1):
        s += t[i, j]
    out[0] = s


@test_utils.test(arch=get_host_arch_list())
def test_subscript_read_identity_layout():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = qd.tensor(qd.f32, shape=(3, 4))
    t.from_numpy(src)
    out = qd.ndarray(qd.f32, shape=(1,))
    _read_sum(t, 3, 4, out)
    assert out[0] == src.sum()


@test_utils.test(arch=get_host_arch_list())
def test_subscript_read_transposed_layout():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    t.from_numpy(src)
    out = qd.ndarray(qd.f32, shape=(1,))
    _read_sum(t, 3, 4, out)
    assert out[0] == src.sum()


# ---------------------------------------------------------------------------
# Augmented assignment.
# ---------------------------------------------------------------------------


@qd.kernel
def _increment(t: qd.Tensor, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        t[i, j] += qd.f32(1.0)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_augassign_both_layouts():
    for layout in [(0, 1), (1, 0)]:
        t = qd.tensor(qd.f32, shape=(2, 3), layout=layout)
        t.from_numpy(np.zeros((2, 3), dtype=np.float32))
        _increment(t, 2, 3)
        _increment(t, 2, 3)
        np.testing.assert_array_equal(t.to_numpy(), 2.0 * np.ones((2, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# 1D and 3D ranks.
# ---------------------------------------------------------------------------


@qd.kernel
def _fill_1d(t: qd.Tensor, n: qd.i32):
    for i in range(n):
        t[i] = qd.f32(i * 7)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_1d_identity():
    t = qd.tensor(qd.f32, shape=(5,))
    _fill_1d(t, 5)
    np.testing.assert_array_equal(t.to_numpy(), np.array([0, 7, 14, 21, 28], dtype=np.float32))


@qd.kernel
def _fill_3d(t: qd.Tensor, n0: qd.i32, n1: qd.i32, n2: qd.i32):
    for i, j, k in qd.ndrange(n0, n1, n2):
        t[i, j, k] = qd.f32(i * 100 + j * 10 + k)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_3d_permuted_layout():
    """Layout (2, 0, 1) on a 3D tensor: physical[layout[k]] = idx[k]."""
    t = qd.tensor(qd.f32, shape=(2, 3, 4), layout=(2, 0, 1))
    _fill_3d(t, 2, 3, 4)
    expected = np.fromfunction(lambda i, j, k: i * 100 + j * 10 + k, (2, 3, 4), dtype=np.float32)
    np.testing.assert_array_equal(t.to_numpy(), expected)
    # Physical shape: (3, 4, 2) — confirms the rewrite is using layout, not identity.
    assert t.underlying.to_numpy().shape == (3, 4, 2)


# ---------------------------------------------------------------------------
# Field backend with bare subscript.
# ---------------------------------------------------------------------------


@qd.kernel
def _write_logical_field(t: qd.FieldTensor, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        t[i, j] = qd.f32(i * 100 + j)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_field_backend_transposed():
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0), backend=qd.Backend.FIELD)
    _write_logical_field(t, 3, 4)
    expected = np.fromfunction(lambda i, j: i * 100 + j, (3, 4), dtype=np.float32)
    np.testing.assert_array_equal(t.to_numpy(), expected)
    np.testing.assert_array_equal(t.underlying.to_numpy(), expected.T)


# ---------------------------------------------------------------------------
# Coexistence: tensor subscript + ndarray subscript in the same kernel.
# Confirms the rewrite only touches tensor-typed parameters.
# ---------------------------------------------------------------------------


@qd.kernel
def _copy_tensor_to_ndarray(
    t: qd.Tensor,
    out: qd.types.ndarray(qd.f32, 2),
    n0: qd.i32,
    n1: qd.i32,
):
    for i, j in qd.ndrange(n0, n1):
        out[i, j] = t[i, j]


@test_utils.test(arch=get_host_arch_list())
def test_subscript_does_not_touch_ndarray_param():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    t.from_numpy(src)
    out = qd.ndarray(qd.f32, shape=(3, 4))
    _copy_tensor_to_ndarray(t, out, 3, 4)
    np.testing.assert_array_equal(out.to_numpy(), src)


# ---------------------------------------------------------------------------
# qd.func callee with a tensor parameter.
# ---------------------------------------------------------------------------


@qd.func
def _tensor_sum(t: qd.Tensor, n0: qd.i32, n1: qd.i32) -> qd.f32:
    s = qd.f32(0.0)
    for i, j in qd.ndrange(n0, n1):
        s += t[i, j]
    return s


@qd.kernel
def _call_tensor_sum(t: qd.Tensor, n0: qd.i32, n1: qd.i32, out: qd.types.ndarray(qd.f32, 1)):
    out[0] = _tensor_sum(t, n0, n1)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_inside_qd_func_callee():
    """The same AST pass runs inside qd.func bodies; bare subscript works there too."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    t.from_numpy(src)
    out = qd.ndarray(qd.f32, shape=(1,))
    _call_tensor_sum(t, 3, 4, out)
    assert out[0] == src.sum()


# ---------------------------------------------------------------------------
# fastcache: bare subscript still produces per-layout cache entries.
# ---------------------------------------------------------------------------


@qd.kernel
def _bare_fill(t: qd.Tensor, n0: qd.i32, n1: qd.i32):
    for i, j in qd.ndrange(n0, n1):
        t[i, j] = qd.f32(0.0)


@test_utils.test(arch=get_host_arch_list())
def test_subscript_layouts_compile_to_distinct_cache_entries():
    t_id = qd.tensor(qd.f32, shape=(3, 4))
    t_tr = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    _bare_fill(t_id, 3, 4)
    _bare_fill(t_tr, 3, 4)
    n_compiled = len(_bare_fill._primal.mapper.mapping)
    assert n_compiled >= 2, f"expected at least 2 cache entries (one per layout), got {n_compiled}"
