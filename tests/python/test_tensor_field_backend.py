"""Tests for qd.tensor with backend=Backend.FIELD — Phase 1b.

These tests cover the python-scope surface of :class:`qd.FieldTensor`,
mirroring ``test_tensor.py`` but selecting the field storage backend.
The kernel-arg binding path for FieldTensor is exercised in Phase 2b.
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

# ---------------------------------------------------------------------------
# Backend enum + factory dispatch.
# ---------------------------------------------------------------------------


def test_backend_enum_members():
    """Backend exposes both NDARRAY and FIELD; values are stable strings."""
    assert qd.Backend.NDARRAY.value == "ndarray"
    assert qd.Backend.FIELD.value == "field"
    assert {b.value for b in qd.Backend} == {"ndarray", "field"}


@test_utils.test(arch=get_host_arch_list())
def test_factory_default_is_ndarray():
    """Default backend kwarg yields an NdarrayTensor (backwards compatible)."""
    t = qd.tensor(qd.f32, shape=(3, 4))
    assert isinstance(t, qd.NdarrayTensor)
    assert t.backend is qd.Backend.NDARRAY


@test_utils.test(arch=get_host_arch_list())
def test_factory_field_backend_yields_FieldTensor():
    t = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD)
    assert isinstance(t, qd.FieldTensor)
    assert t.backend is qd.Backend.FIELD


@test_utils.test(arch=get_host_arch_list())
def test_factory_string_backend_coercion():
    """Tolerate string literal 'field' / 'ndarray' for ergonomics."""
    t_field = qd.tensor(qd.f32, shape=(2, 3), backend="field")
    t_nd = qd.tensor(qd.f32, shape=(2, 3), backend="ndarray")
    assert isinstance(t_field, qd.FieldTensor)
    assert isinstance(t_nd, qd.NdarrayTensor)


def test_factory_rejects_bad_backend_type():
    qd.init(arch=qd.cpu)
    with pytest.raises(TypeError, match="backend must be a Backend enum"):
        qd.tensor(qd.f32, shape=(2,), backend=42)


def test_factory_rejects_unknown_backend_string():
    qd.init(arch=qd.cpu)
    with pytest.raises(ValueError, match="unknown backend"):
        qd.tensor(qd.f32, shape=(2,), backend="nope")


# ---------------------------------------------------------------------------
# FieldTensor python-scope: same API as NdarrayTensor, different storage.
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_default_layout_is_identity():
    t = qd.tensor(qd.f32, shape=(4, 6), backend=qd.Backend.FIELD)
    assert t.shape == (4, 6)
    assert t.physical_shape == (4, 6)
    assert t.layout == (0, 1)
    assert t.ndim == 2


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_transposed_layout():
    """layout=(1, 0) flips the physical shape but preserves logical shape."""
    t = qd.tensor(qd.f32, shape=(4, 6), layout=(1, 0), backend=qd.Backend.FIELD)
    assert t.shape == (4, 6)
    assert t.physical_shape == (6, 4)
    assert t.layout == (1, 0)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_3d_layout():
    t = qd.tensor(qd.f32, shape=(2, 3, 5), layout=(2, 0, 1), backend=qd.Backend.FIELD)
    assert t.shape == (2, 3, 5)
    # physical[layout[i]] = logical[i] -> physical[2]=2, physical[0]=3, physical[1]=5
    assert t.physical_shape == (3, 5, 2)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_dtype():
    t = qd.tensor(qd.f32, shape=(3,), backend=qd.Backend.FIELD)
    assert t.dtype == qd.f32


# ---------------------------------------------------------------------------
# Logical subscript translates to physical via field.__getitem__/__setitem__.
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_subscript_identity_layout():
    t = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD)
    t[1, 2] = 7.5
    assert t[1, 2] == 7.5
    # No layout permutation: underlying field has the same physical layout.
    assert t.underlying[1, 2] == 7.5


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_subscript_transposed_writes_to_swapped_physical():
    """Writing logical[i, j] on a (1, 0) FieldTensor lands at physical[j, i]."""
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0), backend=qd.Backend.FIELD)
    t[1, 2] = 7.5
    # Physical shape is (4, 3); the write should be at physical[2, 1].
    assert t.underlying[2, 1] == 7.5
    # And reading back through the logical view returns the same value.
    assert t[1, 2] == 7.5


# ---------------------------------------------------------------------------
# from_numpy / to_numpy round-trip (logical layout).
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_from_numpy_to_numpy_identity():
    t = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD)
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_from_numpy_to_numpy_transposed():
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0), backend=qd.Backend.FIELD)
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t.from_numpy(src)
    # Logical view round-trips.
    np.testing.assert_array_equal(t.to_numpy(), src)
    # Underlying physical storage really is transposed.
    np.testing.assert_array_equal(t.underlying.to_numpy(), src.T)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_from_numpy_3d_permuted():
    """A 3D permutation: logical (2, 3, 5), layout (2, 0, 1)."""
    t = qd.tensor(qd.f32, shape=(2, 3, 5), layout=(2, 0, 1), backend=qd.Backend.FIELD)
    src = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)
    # Physical shape: (3, 5, 2).
    assert t.underlying.to_numpy().shape == (3, 5, 2)


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_from_numpy_wrong_logical_shape_raises():
    t = qd.tensor(qd.f32, shape=(3, 4), backend=qd.Backend.FIELD)
    with pytest.raises(ValueError, match="expects logical shape"):
        # Passing physical (transposed) shape should fail because we expect logical.
        t.from_numpy(np.zeros((4, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# Misc: fill, repr, isinstance behaviour, separation from NdarrayTensor.
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_fill_layout_independent():
    t = qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0), backend=qd.Backend.FIELD)
    t.fill(2.0)
    np.testing.assert_array_equal(t.to_numpy(), 2.0 * np.ones((2, 3), dtype=np.float32))


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_repr_includes_backend_marker():
    t = qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0), backend=qd.Backend.FIELD)
    text = repr(t)
    assert "FieldTensor" in text
    assert "shape=(2, 3)" in text
    assert "layout=(1, 0)" in text
    assert "backend=field" in text


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_is_not_an_ndarray_tensor():
    """Backend variants are distinct types so kernel-arg dispatch can branch on them."""
    t_nd = qd.tensor(qd.f32, shape=(2, 3))
    t_fd = qd.tensor(qd.f32, shape=(2, 3), backend=qd.Backend.FIELD)
    assert isinstance(t_nd, qd.NdarrayTensor)
    assert not isinstance(t_nd, qd.FieldTensor)
    assert isinstance(t_fd, qd.FieldTensor)
    assert not isinstance(t_fd, qd.NdarrayTensor)
    # qd.Tensor is the backwards-compat alias for the ndarray variant.
    assert qd.Tensor is qd.NdarrayTensor


@test_utils.test(arch=get_host_arch_list())
def test_field_tensor_invalid_layout_rejected():
    """Layout validation must reject non-permutations on the field backend too."""
    qd.init(arch=qd.cpu)
    with pytest.raises(ValueError, match="permutation of range"):
        qd.tensor(qd.f32, shape=(3, 4), layout=(0, 0), backend=qd.Backend.FIELD)
