"""Tests for BufferView: slice syntax, kernel annotation, @qd.func support, and debug-mode OOB with callstack."""

import platform

import numpy as np
import pytest

import quadrants as qd
from quadrants import BufferView
from quadrants.lang.exception import QuadrantsAssertionError, QuadrantsRuntimeTypeError
from quadrants.lang.misc import get_host_arch_list

from tests import test_utils

# Skip debug-mode assertion tests on ARM64 Linux where assertions are unsupported.
_u = platform.uname()
_no_assert = _u.system == "linux" and _u.machine in ("arm64", "aarch64")

N = 32


# ---------------------------------------------------------------------------
# Group A - Slice syntax (host-level)
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_slice_basic():
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[:16]
    assert isinstance(view, BufferView)
    assert view.offset == 0
    assert view.size == 16
    assert view.shape == (16,)
    assert view.get_ndarray() is data


@test_utils.test(arch=get_host_arch_list())
def test_slice_with_start():
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[8:24]
    assert view.offset == 8
    assert view.size == 16
    assert view.shape == (16,)


@test_utils.test(arch=get_host_arch_list())
def test_slice_full():
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[:]
    assert view.offset == 0
    assert view.size == N
    assert view.shape == (N,)


@test_utils.test(arch=get_host_arch_list())
def test_slice_to_end():
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[8:]
    assert view.offset == 8
    assert view.size == N - 8


@test_utils.test(arch=get_host_arch_list())
def test_slice_negative_start():
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[-8:]
    assert view.offset == N - 8
    assert view.size == 8


@test_utils.test(arch=get_host_arch_list())
def test_slice_step_error():
    data = qd.ndarray(qd.f32, shape=(N,))
    with pytest.raises(ValueError, match="step=1"):
        _ = data[::2]


@test_utils.test(arch=get_host_arch_list())
def test_slice_2d_error():
    data = qd.ndarray(qd.f32, shape=(4, 4))
    with pytest.raises(TypeError, match="1D"):
        _ = data[:2]


@test_utils.test(arch=get_host_arch_list())
def test_explicit_constructor():
    data = qd.ndarray(qd.f32, shape=(N,))
    view = BufferView(data, 4, 8)
    assert view.offset == 4
    assert view.size == 8
    assert view.shape == (8,)
    assert view.get_ndarray() is data


@test_utils.test(arch=get_host_arch_list())
def test_vector_ndarray_slice():
    v_arr = qd.Vector.ndarray(3, qd.f32, (N,))
    view = v_arr[:16]
    assert isinstance(view, BufferView)
    assert view.offset == 0
    assert view.size == 16
    assert view.get_ndarray() is v_arr


@test_utils.test(arch=get_host_arch_list())
def test_matrix_ndarray_slice():
    """MatrixNdarray[:N] produces a BufferView over matrix elements."""
    m_arr = qd.Matrix.ndarray(3, 3, qd.f32, (N,))
    view = m_arr[:16]
    assert isinstance(view, BufferView)
    assert view.offset == 0
    assert view.size == 16
    assert view.get_ndarray() is m_arr


@test_utils.test(arch=get_host_arch_list())
def test_matrix_ndarray_kernel():
    """BufferView over MatrixNdarray can be passed to a kernel and written."""
    m_arr = qd.Matrix.ndarray(3, 3, qd.f32, (N,))
    m_arr.from_numpy(np.zeros((N, 3, 3), dtype=np.float32))

    @qd.kernel
    def fill_diag(v: BufferView):
        for i in range(v.size):
            for k in range(3):
                v[i][k, k] = 1.0

    fill_diag(m_arr[8:16])
    result = m_arr.to_numpy()
    for i in range(N):
        if 8 <= i < 16:
            assert np.allclose(result[i], np.eye(3, dtype=np.float32))
        else:
            assert np.allclose(result[i], np.zeros((3, 3), dtype=np.float32))


@test_utils.test(arch=get_host_arch_list())
def test_vector_ndarray_kernel():
    """BufferView over VectorNdarray can be passed to a kernel and written."""
    v_arr = qd.Vector.ndarray(3, qd.f32, (N,))
    v_arr.from_numpy(np.zeros((N, 3), dtype=np.float32))

    @qd.kernel
    def fill_ones(v: BufferView):
        for i in range(v.size):
            for k in range(3):
                v[i][k] = 1.0

    fill_ones(v_arr[4:12])
    result = v_arr.to_numpy()
    for i in range(N):
        if 4 <= i < 12:
            assert np.allclose(result[i], np.ones(3, dtype=np.float32))
        else:
            assert np.allclose(result[i], np.zeros(3, dtype=np.float32))


@test_utils.test(arch=get_host_arch_list())
def test_host_subview():
    """subview() creates a narrower view with accumulated offset."""
    data = qd.ndarray(qd.f32, shape=(N,))
    a = data[8:24]  # offset=8, size=16
    b = a.subview(4, 8)  # offset=12, size=8
    assert b.offset == 12
    assert b.size == 8
    assert b.get_ndarray() is data


@test_utils.test(arch=get_host_arch_list())
def test_host_subview_error():
    """subview() raises ValueError when sub-range exceeds parent."""
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[:16]
    with pytest.raises(ValueError, match="subview out of range"):
        view.subview(8, 16)  # 8 + 16 > 16


@test_utils.test(arch=get_host_arch_list())
def test_host_view_slice():
    """BufferView slice creates a subview with correct offset accumulation."""
    data = qd.ndarray(qd.f32, shape=(N,))
    a = data[8:24]  # offset=8, size=16
    b = a[4:12]  # offset=12, size=8
    c = b[:4]  # offset=12, size=4
    assert b.offset == 12
    assert b.size == 8
    assert c.offset == 12
    assert c.size == 4
    assert c.get_ndarray() is data


@test_utils.test(arch=get_host_arch_list())
def test_host_view_slice_step_error():
    """BufferView slice with step != 1 raises ValueError."""
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[:16]
    with pytest.raises(ValueError, match="step=1"):
        _ = view[::2]


@test_utils.test(arch=get_host_arch_list())
def test_host_view_int_index_error():
    """BufferView integer indexing on host raises TypeError (use slice instead)."""
    data = qd.ndarray(qd.f32, shape=(N,))
    view = data[:16]
    with pytest.raises(TypeError, match="slice"):
        _ = view[0]


# ---------------------------------------------------------------------------
# Group B - Kernel functional tests
# ---------------------------------------------------------------------------


@test_utils.test(arch=get_host_arch_list())
def test_bracket_annotation():
    """BufferView[qd.f32] annotation auto-decomposes and recomposes correctly."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.kernel
    def fill(v: BufferView[qd.f32]):
        for i in range(v.size):
            v[i] = 1.0

    fill(data[:16])
    result = data.to_numpy()
    assert np.all(result[:16] == 1.0)
    assert np.all(result[16:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_no_dtype_annotation():
    """BufferView without dtype (v: BufferView) infers dtype from the passed argument."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.kernel
    def fill(v: BufferView):
        for i in range(v.size):
            v[i] = 3.0

    fill(data[:16])
    result = data.to_numpy()
    assert np.all(result[:16] == 3.0)
    assert np.all(result[16:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_write_via_view():
    """Kernel write through an offset view only touches the view's range."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.kernel
    def fill_view(v: BufferView[qd.f32]):
        for i in range(v.size):
            v[i] = 7.0

    fill_view(data[8:16])  # offset=8, size=8
    result = data.to_numpy()
    assert np.all(result[:8] == 0.0)
    assert np.all(result[8:16] == 7.0)
    assert np.all(result[16:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_size_in_kernel():
    """v.size inside a kernel gives the correct iteration bound."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))
    counter = qd.ndarray(qd.i32, shape=(1,))
    counter.from_numpy(np.zeros(1, dtype=np.int32))

    @qd.kernel
    def count_iters(v: BufferView[qd.f32], c: qd.types.ndarray(qd.i32)):
        s = 0
        for i in range(v.size):
            s += 1
        c[0] = s

    count_iters(data[4:20], counter)
    assert counter.to_numpy()[0] == 16


@test_utils.test(arch=get_host_arch_list())
def test_multiple_views():
    """Two BufferView arguments with different offsets are independent."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.kernel
    def fill_two(a: BufferView[qd.f32], b: BufferView[qd.f32]):
        for i in range(a.size):
            a[i] = 1.0
        for i in range(b.size):
            b[i] = 2.0

    fill_two(data[:8], data[16:24])
    result = data.to_numpy()
    assert np.all(result[:8] == 1.0)
    assert np.all(result[8:16] == 0.0)
    assert np.all(result[16:24] == 2.0)
    assert np.all(result[24:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_func_annotation():
    """@qd.func with BufferView[dtype] annotation works when called from a kernel."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.func
    def set_val(v: BufferView[qd.f32], idx: qd.i32, val: qd.f32):
        v[idx] = val

    @qd.kernel
    def run(v: BufferView[qd.f32]):
        for i in range(v.size):
            set_val(v, i, 5.0)

    run(data[8:16])
    result = data.to_numpy()
    assert np.all(result[:8] == 0.0)
    assert np.all(result[8:16] == 5.0)
    assert np.all(result[16:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_kernel_subview():
    """subview() inside a kernel writes to the correct sub-range of the backing ndarray."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        sub = v.subview(4, 4)
        for i in range(sub.size):
            sub[i] = 9.0

    k(data[8:24])  # view: offset=8, size=16; subview: offset=12, size=4
    result = data.to_numpy()
    assert np.all(result[:12] == 0.0)
    assert np.all(result[12:16] == 9.0)
    assert np.all(result[16:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_kernel_construct_from_ndarray():
    """BufferView constructed inside a kernel from ndarray + offset + size."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.kernel
    def k(arr: qd.types.ndarray(qd.f32, ndim=1), off: qd.i32, sz: qd.i32):
        view = BufferView(arr, off, sz)
        for i in range(view.size):
            view[i] = 4.0

    k(data, 8, 8)  # write 4.0 to data[8:16]
    result = data.to_numpy()
    assert np.all(result[:8] == 0.0)
    assert np.all(result[8:16] == 4.0)
    assert np.all(result[16:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_kernel_slice():
    """v[4:8] slice syntax inside a kernel creates a subview and writes correctly."""
    data = qd.ndarray(qd.f32, shape=(N,))
    data.from_numpy(np.zeros(N, dtype=np.float32))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        sub = v[4:8]
        for i in range(sub.size):
            sub[i] = 6.0

    k(data[8:24])  # view: offset=8, size=16; slice [4:8] -> subview offset=12, size=4
    result = data.to_numpy()
    assert np.all(result[:12] == 0.0)
    assert np.all(result[12:16] == 6.0)
    assert np.all(result[16:] == 0.0)


@test_utils.test(arch=get_host_arch_list())
def test_constructor_negative_offset():
    """BufferView with negative offset raises ValueError."""
    data = qd.ndarray(qd.f32, shape=(N,))
    with pytest.raises(ValueError, match="out of range"):
        BufferView(data, -1, 8)


@test_utils.test(arch=get_host_arch_list())
def test_constructor_negative_size():
    """BufferView with negative size raises ValueError."""
    data = qd.ndarray(qd.f32, shape=(N,))
    with pytest.raises(ValueError, match="out of range"):
        BufferView(data, 0, -1)


@test_utils.test(arch=get_host_arch_list())
def test_constructor_exceeds_length():
    """BufferView with offset+size > ndarray length raises ValueError."""
    data = qd.ndarray(qd.f32, shape=(N,))
    with pytest.raises(ValueError, match="out of range"):
        BufferView(data, 16, 32)  # 16 + 32 > 32


@test_utils.test(arch=get_host_arch_list())
def test_kernel_slice_step_error():
    """v[::2] inside a kernel raises QuadrantsSyntaxError."""
    from quadrants.lang.exception import QuadrantsSyntaxError as QSE

    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        sub = v[::2]
        for i in range(sub.size):
            sub[i] = 1.0

    with pytest.raises(QSE, match="explicit step"):
        k(data[:16])


@test_utils.test(arch=get_host_arch_list())
def test_wrong_type_error():
    """Passing a plain ndarray where BufferView[dtype] is expected raises an error."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        for i in range(v.size):
            v[i] = 0.0

    with pytest.raises(QuadrantsRuntimeTypeError):
        k(data)  # plain ndarray, not a BufferView


# ---------------------------------------------------------------------------
# Group C - Debug mode: auto-error + complete callstack
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_no_assert, reason="assert not supported on linux arm64/aarch64")
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_debug_oob_upper():
    """Debug mode: index == size raises QuadrantsAssertionError."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        for i in range(1):
            v[v.size + i] = 1.0  # v.size + 0 == size → OOB

    with pytest.raises(QuadrantsAssertionError, match=r"BufferView Out Of Range"):
        k(data[:16])


@pytest.mark.skipif(_no_assert, reason="assert not supported on linux arm64/aarch64")
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_debug_oob_lower():
    """Debug mode: negative index (passed as qd.i32) raises QuadrantsAssertionError."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32], bad_idx: qd.i32):
        for i in range(1):
            v[bad_idx + i] = 1.0  # bad_idx = -1 → OOB (< 0)

    with pytest.raises(QuadrantsAssertionError, match=r"BufferView Out Of Range"):
        k(data[:16], -1)  # pass -1 as qd.i32 to avoid constant-negative-index restriction


@pytest.mark.skipif(_no_assert, reason="assert not supported on linux arm64/aarch64")
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_debug_oob_offset_reported():
    """Debug mode error message reports the VIEW's offset and size, not the ndarray's."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        for i in range(1):
            v[v.size + i] = 1.0  # OOB: size=16, index=16

    with pytest.raises(QuadrantsAssertionError) as exc_info:
        k(data[8:24])  # offset=8, size=16

    msg = str(exc_info.value)
    assert "BufferView Out Of Range" in msg
    assert "tid=" in msg
    assert "offset=" in msg
    assert "size=" in msg
    assert "Callstack:" in msg
    # Offset and size must reflect the VIEW, not the backing ndarray.
    assert "offset=8" in msg
    assert "size=16" in msg


@pytest.mark.skipif(_no_assert, reason="assert not supported on linux arm64/aarch64")
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_debug_callstack_nested():
    """Debug mode: OOB via @qd.func shows both kernel and func frames in callstack."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.func
    def writer(v: BufferView[qd.f32], idx: qd.i32):
        v[idx] = 99.0  # OOB happens here when idx >= size

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        for i in range(1):
            writer(v, v.size + i)  # passes size+0 = out-of-range index

    with pytest.raises(QuadrantsAssertionError) as exc_info:
        k(data[:16])

    msg = str(exc_info.value)
    assert "Callstack:" in msg
    # Both frames must appear: kernel frame and func frame.
    assert "k" in msg
    assert "writer" in msg


@pytest.mark.skipif(_no_assert, reason="assert not supported on linux arm64/aarch64")
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_debug_subview_oob():
    """Debug mode: subview exceeding parent size raises QuadrantsAssertionError."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        for i in range(1):
            sub = v.subview(v.size + i, 1)  # offset == parent size -> OOB
            sub[0] = 1.0

    with pytest.raises(QuadrantsAssertionError, match=r"subview Out Of Range"):
        k(data[:16])


@pytest.mark.skipif(_no_assert, reason="assert not supported on linux arm64/aarch64")
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_debug_kernel_construct_oob():
    """Debug mode: kernel-constructed BufferView with offset+size > ndarray length raises assertion."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(arr: qd.types.ndarray(qd.f32, ndim=1)):
        for i in range(1):
            view = BufferView(arr, 28 + i, 8)  # 28 + 8 = 36 > 32
            view[0] = 1.0

    with pytest.raises(QuadrantsAssertionError, match=r"BufferView construction out of range"):
        k(data)


@pytest.mark.skipif(_no_assert, reason="assert not supported on linux arm64/aarch64")
@test_utils.test(require=qd.extension.assertion, debug=True, gdb_trigger=False)
def test_debug_no_false_positive():
    """Debug mode: accessing the last valid index (size-1) must NOT raise."""
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        for i in range(1):
            v[v.size - 1 + i] = 1.0  # last valid index - should succeed

    k(data[:16])  # must complete without exception


@test_utils.test(arch=get_host_arch_list())
def test_no_debug_no_assertion():
    """Without debug mode, an out-of-range index does NOT raise QuadrantsAssertionError."""
    # The view is data[:16]; accessing index 16 writes to data[16], within the backing ndarray (N=32), no IR-level OOB.
    data = qd.ndarray(qd.f32, shape=(N,))

    @qd.kernel
    def k(v: BufferView[qd.f32]):
        v[v.size] = 1.0  # OOB relative to view, but no assertions in non-debug mode

    k(data[:16])  # must complete without QuadrantsAssertionError
