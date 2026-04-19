"""Tests for qd.tensor / qd.Tensor — Phase 1: Python-side wrapper.

These tests cover construction, validation, properties, python-scope
__getitem__/__setitem__, and to_numpy/from_numpy round-trips. Kernel-scope
subscript translation is added in Phase 3 and tested separately.
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils


@test_utils.test(arch=get_host_arch_list())
def test_tensor_default_layout_is_identity():
    t = qd.tensor(qd.f32, shape=(4, 6))
    assert t.shape == (4, 6)
    assert t.physical_shape == (4, 6)
    assert t.layout == (0, 1)
    assert t.ndim == 2


@test_utils.test(arch=get_host_arch_list())
def test_tensor_explicit_identity_layout():
    t = qd.tensor(qd.f32, shape=(4, 6), layout=(0, 1))
    assert t.layout == (0, 1)
    assert t.physical_shape == (4, 6)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_transposed_layout_swaps_physical_shape():
    t = qd.tensor(qd.f32, shape=(4, 6), layout=(1, 0))
    assert t.shape == (4, 6)
    assert t.physical_shape == (6, 4)
    assert t.layout == (1, 0)
    assert tuple(t.underlying.shape) == (6, 4)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_3d_arbitrary_permutation():
    t = qd.tensor(qd.f32, shape=(2, 3, 5), layout=(2, 0, 1))
    assert t.shape == (2, 3, 5)
    # logical 0 -> physical 2, logical 1 -> physical 0, logical 2 -> physical 1
    # so physical_shape[2]=2, physical_shape[0]=3, physical_shape[1]=5
    assert t.physical_shape == (3, 5, 2)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_int_shape_promoted_to_tuple():
    t = qd.tensor(qd.i32, shape=8)
    assert t.shape == (8,)
    assert t.layout == (0,)
    assert t.physical_shape == (8,)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_invalid_layout_wrong_length():
    with pytest.raises(ValueError, match="layout must have length"):
        qd.tensor(qd.f32, shape=(4, 6), layout=(0, 1, 2))


@test_utils.test(arch=get_host_arch_list())
def test_tensor_invalid_layout_not_a_permutation():
    with pytest.raises(ValueError, match="must be a permutation"):
        qd.tensor(qd.f32, shape=(4, 6), layout=(0, 0))


@test_utils.test(arch=get_host_arch_list())
def test_tensor_invalid_layout_out_of_range():
    with pytest.raises(ValueError, match="must be a permutation"):
        qd.tensor(qd.f32, shape=(4, 6), layout=(0, 5))


@test_utils.test(arch=get_host_arch_list())
def test_tensor_dtype_property():
    t = qd.tensor(qd.f32, shape=(3,))
    assert t.dtype == qd.f32

    t_i = qd.tensor(qd.i32, shape=(3,))
    assert t_i.dtype == qd.i32


@test_utils.test(arch=get_host_arch_list())
def test_tensor_python_scope_setitem_getitem_identity():
    t = qd.tensor(qd.f32, shape=(3, 4))
    t[1, 2] = 7.5
    assert t[1, 2] == pytest.approx(7.5)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_python_scope_setitem_getitem_transposed():
    """Logical indexing should be transparent regardless of layout."""
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    t[1, 2] = 7.5
    assert t[1, 2] == pytest.approx(7.5)
    # Verify the physical storage actually has the value at the transposed slot.
    # logical (1, 2) -> physical (2, 1) when layout=(1, 0).
    assert t.underlying[2, 1] == pytest.approx(7.5)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_python_scope_wrong_ndim_index_raises():
    t = qd.tensor(qd.f32, shape=(3, 4))
    with pytest.raises(IndexError, match="2d tensor indexed with 1d key"):
        _ = t[1]


@test_utils.test(arch=get_host_arch_list())
def test_tensor_python_scope_slice_raises_clear_error():
    t = qd.tensor(qd.f32, shape=(3, 4))
    with pytest.raises(TypeError, match="Slicing is not yet implemented"):
        _ = t[1, slice(0, 2)]


@test_utils.test(arch=get_host_arch_list())
def test_tensor_to_numpy_identity():
    t = qd.tensor(qd.f32, shape=(2, 3))
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    for i in range(2):
        for j in range(3):
            t[i, j] = float(expected[i, j])
    np.testing.assert_array_equal(t.to_numpy(), expected)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_to_numpy_transposed_returns_logical_view():
    t = qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0))
    expected = np.arange(6, dtype=np.float32).reshape(2, 3)
    for i in range(2):
        for j in range(3):
            t[i, j] = float(expected[i, j])
    np.testing.assert_array_equal(t.to_numpy(), expected)
    # Underlying physical storage should be the transpose.
    np.testing.assert_array_equal(t.underlying.to_numpy(), expected.T)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_from_numpy_identity_roundtrip():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = qd.tensor(qd.f32, shape=(3, 4))
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_from_numpy_transposed_roundtrip():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = qd.tensor(qd.f32, shape=(3, 4), layout=(1, 0))
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)
    # Physical storage should be the transposed view.
    np.testing.assert_array_equal(t.underlying.to_numpy(), src.T)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_from_numpy_wrong_shape_raises():
    t = qd.tensor(qd.f32, shape=(3, 4))
    with pytest.raises(ValueError, match="expects logical shape"):
        t.from_numpy(np.zeros((4, 3), dtype=np.float32))


@test_utils.test(arch=get_host_arch_list())
def test_tensor_from_numpy_non_ndarray_raises():
    t = qd.tensor(qd.f32, shape=(3,))
    with pytest.raises(TypeError, match="expected numpy.ndarray"):
        t.from_numpy([1.0, 2.0, 3.0])


@test_utils.test(arch=get_host_arch_list())
def test_tensor_3d_transpose_roundtrip():
    src = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    # Move logical axes (0,1,2) -> physical (2,0,1).
    t = qd.tensor(qd.f32, shape=(2, 3, 5), layout=(2, 0, 1))
    t.from_numpy(src)
    np.testing.assert_array_equal(t.to_numpy(), src)


@test_utils.test(arch=get_host_arch_list())
def test_tensor_repr_contains_key_fields():
    t = qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0))
    r = repr(t)
    assert "shape=(2, 3)" in r
    assert "layout=(1, 0)" in r
    assert "physical_shape=(3, 2)" in r


@test_utils.test(arch=get_host_arch_list())
def test_tensor_fill():
    t = qd.tensor(qd.f32, shape=(2, 3), layout=(1, 0))
    t.fill(2.5)
    np.testing.assert_array_equal(t.to_numpy(), np.full((2, 3), 2.5, dtype=np.float32))
