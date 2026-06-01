"""Tests for PR #560 review fixes.

Covers:
1. AnyArray.shape returns canonical shape inside kernels on layout-tagged ndarrays
2. Fastcache hash includes _qd_layout so different layouts produce different cache keys
3. Field offset permutation with order=
4. Matrix.field offset permutation with order=
5. Docstring default backend (indirectly tested via existing tests)
6. _validate_kwargs dead-end hint for Vector/Matrix factories
7. VectorNdarray.to_numpy(dtype=) and MatrixNdarray.to_numpy(dtype=)
"""

import numpy as np
import pytest

import quadrants as qd
from quadrants._tensor import _with_layout

from tests import test_utils

# ---------------------------------------------------------------------------
# 1. AnyArray.shape returns canonical shape inside kernels
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_anyarray_shape_canonical_in_kernel_rank2():
    """x.shape[i] inside a kernel on a layout-tagged NDARRAY must return canonical sizes, not physical ones. Otherwise
    for non-square shapes the idiomatic ``for i in range(x.shape[0])`` loop produces OOB writes.
    """
    N, B = 3, 5

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = i * 100 + j

    a = qd.tensor(qd.i32, shape=(B, N), backend=qd.Backend.NDARRAY)
    _with_layout(a, (1, 0))

    fill(a)
    got = a.to_numpy()

    expected = np.zeros((N, B), dtype=np.int32)
    for i in range(N):
        for j in range(B):
            expected[i, j] = i * 100 + j
    np.testing.assert_array_equal(got, expected)


@test_utils.test(arch=qd.cpu)
def test_anyarray_shape_canonical_identity_layout():
    """Identity layout should not affect .shape semantics."""
    M, N = 4, 6

    @qd.kernel
    def fill(x: qd.types.ndarray()):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = i * 10 + j

    a = qd.tensor(qd.i32, shape=(M, N), backend=qd.Backend.NDARRAY)
    _with_layout(a, (0, 1))

    fill(a)
    got = a.to_numpy()

    expected = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            expected[i, j] = i * 10 + j
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# 2. Fastcache hash includes layout
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_fastcache_layout_distinct_keys():
    """Two ndarrays of the same dtype/rank but different layouts must NOT share a fastcache entry, otherwise the second
    call reuses IR compiled for the first layout and silently mis-indexes.
    """
    from quadrants.lang._fast_caching.args_hasher import stringify_obj_type

    a_plain = qd.ndarray(qd.f32, shape=(4, 3))
    a_layout = qd.ndarray(qd.f32, shape=(3, 4))
    a_layout._qd_layout = (1, 0)

    key_plain = stringify_obj_type(False, ("x",), a_plain, None)
    key_layout = stringify_obj_type(False, ("x",), a_layout, None)

    assert key_plain is not None
    assert key_layout is not None
    assert key_plain != key_layout, (
        f"Plain and layout-tagged ndarrays must produce different cache keys, " f"got {key_plain!r} for both"
    )


@test_utils.test(arch=qd.cpu)
def test_fastcache_vector_ndarray_layout_distinct():
    from quadrants.lang._fast_caching.args_hasher import stringify_obj_type

    v_plain = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    v_layout = qd.Vector.ndarray(3, qd.f32, shape=(4,))
    v_layout._qd_layout = (0,)

    key_plain = stringify_obj_type(False, ("x",), v_plain, None)
    key_layout = stringify_obj_type(False, ("x",), v_layout, None)

    assert key_plain is not None
    assert key_layout is not None
    # Identity layout tagged is technically distinct from None
    assert key_plain != key_layout


@test_utils.test(arch=qd.cpu)
def test_fastcache_matrix_ndarray_layout_distinct():
    from quadrants.lang._fast_caching.args_hasher import stringify_obj_type

    m_plain = qd.Matrix.ndarray(2, 2, qd.f32, shape=(4,))
    m_layout = qd.Matrix.ndarray(2, 2, qd.f32, shape=(4,))
    m_layout._qd_layout = (0,)

    key_plain = stringify_obj_type(False, ("x",), m_plain, None)
    key_layout = stringify_obj_type(False, ("x",), m_layout, None)

    assert key_plain is not None
    assert key_layout is not None
    assert key_plain != key_layout


# ---------------------------------------------------------------------------
# 3 & 4. Field offset permutation with order=
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_scalar_field_offset_with_order():
    """qd.field with order= and non-zero offset must permute the offset so that canonical indexing lands at the correct
    physical cell.
    """
    x = qd.field(qd.i32, shape=(4, 6), order="ji", offset=(1, 2))

    @qd.kernel
    def write():
        x[1, 2] = 42

    write()
    assert x[1, 2] == 42


@test_utils.test(arch=qd.cpu)
def test_matrix_field_offset_with_order():
    """Matrix.field with order= and non-zero offset must permute the offset so canonical indexing is correct."""
    m = qd.Matrix.field(2, 2, qd.i32, shape=(4, 6), order="ji", offset=(1, 2))

    @qd.kernel
    def write():
        m[1, 2] = qd.Matrix([[10, 20], [30, 40]])

    write()
    got = m[1, 2]
    assert got[0, 0] == 10
    assert got[0, 1] == 20
    assert got[1, 0] == 30
    assert got[1, 1] == 40


# ---------------------------------------------------------------------------
# 6. _validate_kwargs dead-end hint
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_vector_tensor_order_hint_no_dead_end():
    """The error message for order= on Vector.tensor must NOT suggest layout= since Vector.tensor doesn't accept
    layout= either.
    """
    with pytest.raises(TypeError, match="does not accept order=") as exc_info:
        qd.Vector.tensor(3, qd.f32, shape=(4,), order="ji")
    msg = str(exc_info.value)
    assert (
        "pass layout=(...) instead" not in msg
    ), f"Dead-end hint: message suggests layout= which is also rejected: {msg}"


@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_order_hint_no_dead_end():
    with pytest.raises(TypeError, match="does not accept order=") as exc_info:
        qd.Matrix.tensor(2, 3, qd.f32, shape=(4,), order="ji")
    msg = str(exc_info.value)
    assert "pass layout=(...) instead" not in msg


@test_utils.test(arch=qd.cpu)
def test_scalar_tensor_order_hint_still_suggests_layout():
    """The scalar qd.tensor() factory DOES accept layout=, so the hint should still point to it."""
    with pytest.raises(TypeError, match="pass layout="):
        qd.tensor(qd.f32, shape=(4,), order="ji")


# ---------------------------------------------------------------------------
# 7. VectorNdarray.to_numpy(dtype=) and MatrixNdarray.to_numpy(dtype=)
# ---------------------------------------------------------------------------


@test_utils.test(arch=qd.cpu)
def test_vector_ndarray_to_numpy_dtype():
    v = qd.Vector.ndarray(3, qd.f32, shape=(2,))
    arr = v.to_numpy(dtype=np.float64)
    assert arr.dtype == np.float64
    assert arr.shape == (2, 3)


@test_utils.test(arch=qd.cpu)
def test_matrix_ndarray_to_numpy_dtype():
    m = qd.Matrix.ndarray(2, 2, qd.f32, shape=(3,))
    arr = m.to_numpy(dtype=np.float64)
    assert arr.dtype == np.float64
    assert arr.shape == (3, 2, 2)


@test_utils.test(arch=qd.cpu)
def test_vector_tensor_to_numpy_dtype_ndarray_backend():
    """Tensor wrapper around VectorNdarray must forward dtype= correctly."""
    v = qd.Vector.tensor(3, qd.f32, shape=(2,), backend=qd.Backend.NDARRAY)
    arr = v.to_numpy(dtype=np.float64)
    assert arr.dtype == np.float64


@test_utils.test(arch=qd.cpu)
def test_matrix_tensor_to_numpy_dtype_ndarray_backend():
    m = qd.Matrix.tensor(2, 2, qd.f32, shape=(3,), backend=qd.Backend.NDARRAY)
    arr = m.to_numpy(dtype=np.float64)
    assert arr.dtype == np.float64


@test_utils.test(arch=qd.cpu)
def test_vector_ndarray_to_numpy_no_dtype_unchanged():
    """to_numpy() without dtype should return native dtype."""
    v = qd.Vector.ndarray(3, qd.f32, shape=(2,))
    arr = v.to_numpy()
    assert arr.dtype == np.float32


@test_utils.test(arch=qd.cpu)
def test_matrix_ndarray_to_numpy_no_dtype_unchanged():
    m = qd.Matrix.ndarray(2, 2, qd.f32, shape=(3,))
    arr = m.to_numpy()
    assert arr.dtype == np.float32
