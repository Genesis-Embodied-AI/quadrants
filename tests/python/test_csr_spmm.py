"""Contract and numerical qualification for the experimental float32 CSR primitive."""

import numpy as np
import pytest

import quadrants as qd
from quadrants.examples.csr_spmm import (
    _check_buffer_shape,
    _checked_i32,
    _error_budget,
    _reference_dense_f64,
    _reference_ordered_f32,
    _upload,
    _validate_csr,
)

from tests import test_utils


@qd.kernel
def _multiply(
    row_ptr: qd.types.NDArray,
    col_idx: qd.types.NDArray,
    values: qd.types.NDArray,
    rhs: qd.types.NDArray,
    out: qd.types.NDArray,
):
    # Generic outer arguments intentionally exercise the inner function's checks.
    qd.algorithms.csr_spmm(row_ptr, col_idx, values, rhs, out)


def _host_case():
    return [
        np.array([0, 3, 3, 4], dtype=np.int32),
        np.array([1, 0, 1, 2, -(2**31)], dtype=np.int32),
        np.array([0.5, -1, 0.25, 2, np.nan], dtype=np.float32),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        np.full((3, 2), np.nan, dtype=np.float32),
    ]


@pytest.mark.parametrize("indirect", [False, True])
@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_rejects_nested_runtime_calls(indirect):
    buffers = [_upload(array) for array in _host_case()]
    buffers[-1].fill(-777)

    @qd.func
    def helper(
        row_ptr: qd.types.NDArray,
        col_idx: qd.types.NDArray,
        values: qd.types.NDArray,
        rhs: qd.types.NDArray,
        out: qd.types.NDArray,
    ):
        qd.algorithms.csr_spmm(row_ptr, col_idx, values, rhs, out)

    @qd.kernel
    def nested(
        row_ptr: qd.types.NDArray,
        col_idx: qd.types.NDArray,
        values: qd.types.NDArray,
        rhs: qd.types.NDArray,
        out: qd.types.NDArray,
        count: qd.i32,
    ):
        for _ in range(count):
            if qd.static(indirect):
                helper(row_ptr, col_idx, values, rhs, out)
            else:
                qd.algorithms.csr_spmm(row_ptr, col_idx, values, rhs, out)

    with pytest.raises(qd.QuadrantsSyntaxError, match="requires_top_level"):
        nested(*buffers, 1)
    np.testing.assert_array_equal(buffers[-1].to_numpy(), np.full((3, 2), -777, dtype=np.float32))


def _execute(host, exact=False, difficult=False):
    _validate_csr(*host)
    before = [array.copy() for array in host[:-1]]
    buffers = [_upload(array) for array in host]
    _multiply(*buffers)
    result = np.asarray(buffers[-1].to_numpy(), dtype=np.float32)
    ordered = _reference_ordered_f32(*host[:-1])
    dense = _reference_dense_f64(*host[:-1])
    if exact:
        np.testing.assert_array_equal(result, ordered)
        np.testing.assert_array_equal(result, dense)
    elif difficult:
        budget = _error_budget(*host[:-1])
        assert np.all(np.isfinite(result))
        assert np.all(np.abs(result.astype(np.float64) - dense) <= budget)
        assert np.all(np.abs(ordered.astype(np.float64) - dense) <= budget)
    else:
        np.testing.assert_allclose(result, ordered, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(result, dense, rtol=2e-5, atol=2e-6)
    for device, expected in zip(buffers[:-1], before):
        np.testing.assert_array_equal(device.to_numpy(), expected)
    return result


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_duplicates_empty_rows_and_poison_tail():
    host = _host_case()
    actual = _execute(host, exact=True)
    np.testing.assert_array_equal(actual, [[1.25, 1], [0, 0], [10, 12]])
    # Keep allocation sizes fixed while changing topology and logical edge count.
    buffers = [_upload(array) for array in host]
    for pointers, columns, coefficients in (
        ([0, 0, 0, 0], [-(2**31)] * 5, [np.nan] * 5),
        (
            [0, 1, 2, 2],
            [2, 0, -(2**31), -(2**31), -(2**31)],
            [2, -1, np.nan, np.nan, np.nan],
        ),
        ([0, 2, 2, 4], [2, 2, 0, 1, -(2**31)], [0.5, 0.25, -1, 2, np.nan]),
    ):
        host[0][:] = pointers
        host[1][:] = columns
        host[2][:] = coefficients
        for device, data in zip(buffers[:3], host[:3]):
            device.from_numpy(data)
        buffers[-1].fill(-777)
        _multiply(*buffers)
        np.testing.assert_array_equal(buffers[-1].to_numpy(), _reference_dense_f64(*host[:-1]))
    empty = [
        np.zeros(4, dtype=np.int32),
        np.array([-1], dtype=np.int32),
        np.array([np.nan], dtype=np.float32),
        host[3],
        np.full((3, 2), -777, dtype=np.float32),
    ]
    np.testing.assert_array_equal(_execute(empty, exact=True), np.zeros((3, 2)))


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_rectangular_widths_and_changed_bindings():
    # One reusable wrapper covers partial launch grids and changes every runtime shape/binding.
    for width in (1, 3, 17, 32, 33, 64, 65):
        host = [
            np.array([0, 0, 3, 3, 4, 4], dtype=np.int32),
            np.array([3, 1, 3, 0, -(2**31), 2**31 - 1], dtype=np.int32),
            np.array([0.5, -1, 0.25, 2, np.nan, np.nan], dtype=np.float32),
            (np.arange(4 * width, dtype=np.float32).reshape(4, width) - 16) / 8,
            np.full((5, width), -777, dtype=np.float32),
        ]
        _execute(host, exact=True)
        # Replacement arrays of the same shape must not retain previous RHS/weight/output bindings.
        replacement = [array.copy() for array in host]
        replacement[2][:4] *= -0.5
        replacement[3] += 0.25
        replacement[4].fill(123)
        _execute(replacement, exact=True)
    _execute(
        [
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([-0.5], dtype=np.float32),
            np.array([[2]], dtype=np.float32),
            np.array([[123]], dtype=np.float32),
        ],
        exact=True,
    )
    # More sources than receivers complements the preceding M > N fixture.
    _execute(
        [
            np.array([0, 2, 3], dtype=np.int32),
            np.array([6, 0, 4], dtype=np.int32),
            np.array([0.5, -1, 2], dtype=np.float32),
            np.arange(21, dtype=np.float32).reshape(7, 3) / 8,
            np.full((2, 3), 123, dtype=np.float32),
        ],
        exact=True,
    )


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_updated_values_and_rhs_in_existing_buffers():
    host = _host_case()
    buffers = [_upload(array) for array in host]
    for step in range(3):
        host[2][:4] = np.array([0.5, -1, 0.25, 2], dtype=np.float32) * (step + 1)
        host[3][:] = np.arange(6, dtype=np.float32).reshape(3, 2) / 8 + step
        for device, data in zip(buffers[2:4], host[2:4]):
            device.from_numpy(data)
        buffers[-1].fill(123 + step)
        _multiply(*buffers)
        np.testing.assert_array_equal(buffers[-1].to_numpy(), _reference_dense_f64(*host[:-1]))
        for device, expected in zip(buffers[:-1], host[:-1]):
            np.testing.assert_array_equal(device.to_numpy(), expected)


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_producer_two_calls_and_consumer():
    # Cross workgroups and read distant producers, so launch ordering is exercised beyond one workgroup.
    rows, channels = 257, 33
    row_ptr = np.arange(rows + 1, dtype=np.int32) * 2
    row_ids = np.arange(rows, dtype=np.int32)
    col_idx = np.stack(((row_ids + 130) % rows, (row_ids + 1) % rows), axis=1).reshape(-1)
    values = np.tile(np.array([0.5, -0.25], dtype=np.float32), rows)
    rhs = np.zeros((rows, channels), dtype=np.float32)
    output = np.full_like(rhs, -777)
    rp, ci, v, x, y = [_upload(array) for array in (row_ptr, col_idx, values, rhs, output)]
    intermediate = _upload(output)

    @qd.kernel
    def pipeline(
        rp: qd.types.NDArray,
        ci: qd.types.NDArray,
        v: qd.types.NDArray,
        x: qd.types.NDArray,
        intermediate: qd.types.NDArray,
        y: qd.types.NDArray,
    ):
        for i, j in x:
            x[i, j] = qd.cast(i + 1, qd.f32) * 0.25 + qd.cast(j, qd.f32) * 0.125
        qd.algorithms.csr_spmm(rp, ci, v, x, intermediate)
        qd.algorithms.csr_spmm(rp, ci, v, intermediate, y)
        for i, j in y:
            y[i, j] = y[i, j] * 0.5 + 1

    pipeline(rp, ci, v, x, intermediate, y)
    produced = (np.arange(rows, dtype=np.float32)[:, None] + 1) / 4
    produced = produced + np.arange(channels, dtype=np.float32)[None, :] / 8
    first = _reference_dense_f64(row_ptr, col_idx, values, produced)
    second = _reference_dense_f64(row_ptr, col_idx, values, first.astype(np.float32))
    np.testing.assert_array_equal(x.to_numpy(), produced)
    np.testing.assert_array_equal(intermediate.to_numpy(), first)
    np.testing.assert_array_equal(y.to_numpy(), second * 0.5 + 1)


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_ping_pong_recurrence():
    row_ptr = np.array([0, 2, 3, 4], dtype=np.int32)
    col_idx = np.array([2, 0, 1, 0], dtype=np.int32)
    values = np.array([0.5, -0.25, 0.5, 0.25], dtype=np.float32)
    expected = np.arange(9, dtype=np.float32).reshape(3, 3) / 8
    rp, ci, v, previous, following = [
        _upload(array)
        for array in (
            row_ptr,
            col_idx,
            values,
            expected,
            np.full_like(expected, -777),
        )
    ]
    for _ in range(4):
        retained = previous.to_numpy()
        _multiply(rp, ci, v, previous, following)
        expected = _reference_dense_f64(row_ptr, col_idx, values, expected).astype(np.float32)
        np.testing.assert_array_equal(following.to_numpy(), expected)
        np.testing.assert_array_equal(previous.to_numpy(), retained)
        previous, following = following, previous


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_shared_column_and_independent_ragged_batches():
    row_ptr = np.array([0, 2, 3], dtype=np.int32)
    col_idx = np.array([2, 0, 1], dtype=np.int32)
    values = np.array([0.5, -0.25, 2], dtype=np.float32)
    trials = [np.arange(6, dtype=np.float32).reshape(3, 2) / 8 + trial for trial in range(3)]
    packed_rhs = np.concatenate(trials, axis=1)
    actual = _execute(
        [row_ptr, col_idx, values, packed_rhs, np.full((2, 6), -777, dtype=np.float32)],
        exact=True,
    )
    expected = np.concatenate([_reference_dense_f64(row_ptr, col_idx, values, x) for x in trials], axis=1)
    np.testing.assert_array_equal(actual, expected)

    # Different shapes and coefficients use a block diagonal graph, never just feature-column packing.
    graphs = [
        (row_ptr, col_idx, values, np.arange(9, dtype=np.float32).reshape(3, 3) / 8),
        (
            np.array([0, 2], dtype=np.int32),
            np.array([1, 0], dtype=np.int32),
            np.array([-1, 0.5], dtype=np.float32),
            np.arange(6, dtype=np.float32).reshape(2, 3) / 4,
        ),
        (
            np.array([0, 0, 1, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([0.25], dtype=np.float32),
            np.array([[2, 4, 6]], dtype=np.float32),
        ),
    ]
    pointers, columns, coefficients, sources, results = [0], [], [], [], []
    source_offset = edge_offset = 0
    for rp, ci, v, x in graphs:
        pointers.extend((rp[1:] + edge_offset).tolist())
        columns.extend((ci + source_offset).tolist())
        coefficients.extend(v.tolist())
        sources.append(x)
        results.append(_reference_dense_f64(rp, ci, v, x))
        source_offset += x.shape[0]
        edge_offset += len(ci)
    host = [
        np.array(pointers, dtype=np.int32),
        np.array(columns + [-999], dtype=np.int32),
        np.array(coefficients + [np.nan], dtype=np.float32),
        np.concatenate(sources, axis=0),
        np.full((len(pointers) - 1, 3), -777, dtype=np.float32),
    ]
    np.testing.assert_array_equal(_execute(host, exact=True), np.concatenate(results, axis=0))


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_difficult_cancellation_and_long_rows():
    rng = np.random.default_rng(7)
    degree, n, k = 2048, 131, 17
    # Signed inputs with exact dyadic scale avoid exceptional arithmetic while producing cancellation.
    columns = rng.integers(0, n, size=degree, dtype=np.int32)
    values = rng.uniform(-1, 1, degree).astype(np.float32)
    rhs = rng.uniform(-1, 1, (n, k)).astype(np.float32)
    host = [
        np.array([0, 0, degree, degree], dtype=np.int32),
        columns,
        values,
        rhs,
        np.full((3, k), np.nan, dtype=np.float32),
    ]
    _execute(host, difficult=True)
    # Hand-computable cancellation with duplicate, unsorted edges.
    host = [
        np.array([0, 5], dtype=np.int32),
        np.array([1, 0, 1, 0, 1], dtype=np.int32),
        np.array([16, 0.5, -16, -0.25, 0], dtype=np.float32),
        np.array([[4, -8, 0.5], [2, 4, -1]], dtype=np.float32),
        np.full((1, 3), np.nan, dtype=np.float32),
    ]
    np.testing.assert_array_equal(_execute(host, exact=True), [[1, -2, 0.125]])


@test_utils.test(arch=qd.cpu, default_fp=qd.f64)
def test_csr_spmm_float32_accumulator_with_float64_default():
    assert qd.cfg is not None
    assert qd.cfg.default_fp == qd.f64
    host = [
        np.array([0, 3], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        np.ones(3, dtype=np.float32),
        np.array([[2**24], [1], [-(2**24)]], dtype=np.float32),
        np.full((1, 1), np.nan, dtype=np.float32),
    ]
    buffers = [_upload(array) for array in host]
    _multiply(*buffers)
    # A default-f64 accumulator would retain the middle unit and return 1.
    np.testing.assert_array_equal(buffers[-1].to_numpy(), [[0]])


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_rejects_wrong_inner_ranks():
    names = ("row_ptr", "col_idx", "values", "rhs", "out")
    for index, name in enumerate(names):
        host = _host_case()
        host[index] = host[index].reshape((1, -1)) if index < 3 else host[index].reshape(-1)
        buffers = [_upload(array) for array in host]
        with pytest.raises(AssertionError, match=f"{name} must have rank"):
            _multiply(*buffers)


@test_utils.test(arch=[qd.cpu, qd.metal])
def test_csr_spmm_rejects_wrong_inner_dtypes():
    for index in range(5):
        buffers = [_upload(array) for array in _host_case()]
        wrong_dtype = qd.f32 if index < 2 else qd.i32
        buffers[index] = qd.ndarray(wrong_dtype, shape=buffers[index].shape)
        with pytest.raises(qd.QuadrantsCompilationError, match="Expect element type"):
            _multiply(*buffers)
    for index in range(5):
        buffers = [_upload(array) for array in _host_case()]
        dtype = qd.i32 if index < 2 else qd.f32
        buffers[index] = qd.Vector.ndarray(2, dtype, shape=buffers[index].shape)
        with pytest.raises(qd.QuadrantsCompilationError, match="Expect element type"):
            _multiply(*buffers)


def test_csr_spmm_host_byte_limits_and_checked_conversion():
    limit = (2**31 - 1) // 4
    assert _check_buffer_shape("test", (limit,)) == limit * 4
    assert _check_buffer_shape("test", (np.int64(1024), np.int64(1024))) == 4 * 1024**2
    for shape in (
        (limit + 1,),
        (65536, 65536),
        (0,),
        (-1,),
        (2**31,),
        (),
        (1.5,),
        (True,),
    ):
        with pytest.raises(ValueError):
            _check_buffer_shape("test", shape)
    for dtype in (np.int64, np.uint64):
        np.testing.assert_array_equal(
            _checked_i32(np.array([0, 2**31 - 1], dtype=dtype), "indices"),
            np.array([0, 2**31 - 1], dtype=np.int32),
        )
        with pytest.raises(ValueError, match="before conversion"):
            _checked_i32(np.array([2**31], dtype=dtype), "indices")
    with pytest.raises(ValueError, match="before conversion"):
        _checked_i32(np.array([-(2**31) - 1], dtype=np.int64), "indices")
    with pytest.raises(TypeError, match="integer data"):
        _checked_i32([0.0, 1.0], "indices")


def test_csr_spmm_host_validation():
    assert _validate_csr(*_host_case()) == (3, 3, 2, 4)
    invalid = (
        (0, np.array([1, 3, 3, 4], dtype=np.int32), "start at zero"),
        (0, np.array([0, 3, 2, 4], dtype=np.int32), "nondecreasing"),
        (0, np.array([0, 3, 3, 6], dtype=np.int32), "capacity"),
        (0, np.array([0, 4], dtype=np.int32), "out rows"),
        (1, np.array([1, 0, 3, 2, -1], dtype=np.int32), "source indices"),
        (1, np.array([1, -1, 1, 2, -1], dtype=np.int32), "source indices"),
        (1, np.array([1], dtype=np.int32), "capacities"),
        (2, np.ones(5, dtype=np.float64), "float32"),
        (2, np.array([np.nan, 1, 1, 1, 1], dtype=np.float32), "finite"),
        (3, np.ones((3, 3), dtype=np.float32), "feature widths"),
        (3, np.ones((0, 2), dtype=np.float32), "positive"),
        (3, np.ones((3, 4), dtype=np.float32)[:, ::2], "C-contiguous"),
        (4, np.ones(6, dtype=np.float32), "rank 2"),
    )
    for index, replacement, message in invalid:
        host = _host_case()
        host[index] = replacement
        with pytest.raises((ValueError, TypeError), match=message):
            _validate_csr(*host)
    host = _host_case()
    host[-1] = host[-2]
    with pytest.raises(ValueError, match="overlap rhs"):
        _validate_csr(*host)
    host = _host_case()
    host[-1].flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        _validate_csr(*host)
