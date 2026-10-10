"""Numerical and composition tests for buffer-backed CSR multiplication."""

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@qd.kernel
def _apply(matrix: qd.linalg.CSRMatrix, rhs: qd.types.NDArray, out: qd.types.NDArray):
    (matrix @ rhs).write_to(out)


def _multiply(row_ptr, col_idx, values, rhs, out):
    matrix = qd.linalg.SparseMatrix.from_csr(row_ptr, col_idx, values, shape=(out.shape[0], rhs.shape[0]))
    _apply(matrix, rhs, out)


def _upload(array):
    result = qd.ndarray(qd.i32 if array.dtype == np.int32 else qd.f32, shape=array.shape)
    result.from_numpy(array)
    return result


def _reference_ordered_f32(row_ptr, col_idx, values, rhs):
    result = np.zeros((len(row_ptr) - 1, rhs.shape[1]), dtype=np.float32)
    for row in range(result.shape[0]):
        for column in range(result.shape[1]):
            for edge in range(int(row_ptr[row]), int(row_ptr[row + 1])):
                result[row, column] = np.float32(
                    result[row, column] + np.float32(values[edge] * rhs[col_idx[edge], column])
                )
    return result


def _reference_dense_f64(row_ptr, col_idx, values, rhs):
    dense = np.zeros((len(row_ptr) - 1, rhs.shape[0]), dtype=np.float64)
    for row in range(dense.shape[0]):
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        np.add.at(dense[row], col_idx[start:end], values[start:end].astype(np.float64))
    return dense @ rhs.astype(np.float64)


def _error_budget(row_ptr, col_idx, values, rhs):
    result = np.empty((len(row_ptr) - 1, rhs.shape[1]), dtype=np.float64)
    for row in range(result.shape[0]):
        start, end = int(row_ptr[row]), int(row_ptr[row + 1])
        nu = 2 * (end - start) * 2.0**-24
        magnitude = np.zeros(rhs.shape[1], dtype=np.float64)
        for edge in range(start, end):
            magnitude += np.abs(np.float64(values[edge]) * rhs[col_idx[edge]].astype(np.float64))
        result[row] = 4 * nu / (1 - nu) * magnitude + 2e-6
    return result


def _host_case():
    return [
        np.array([0, 3, 3, 4], dtype=np.int32),
        np.array([1, 0, 1, 2, -(2**31)], dtype=np.int32),
        np.array([0.5, -1, 0.25, 2, np.nan], dtype=np.float32),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
        np.full((3, 2), np.nan, dtype=np.float32),
    ]


@pytest.mark.parametrize("indirect", [False, True])
@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
def test_csr_spmm_rejects_nested_runtime_calls(indirect):
    buffers = [_upload(array) for array in _host_case()]
    buffers[-1].fill(-777)

    @qd.func
    def helper(matrix: qd.linalg.CSRMatrix, rhs: qd.types.NDArray, out: qd.types.NDArray):
        (matrix @ rhs).write_to(out)

    @qd.kernel
    def nested(
        matrix: qd.linalg.CSRMatrix,
        rhs: qd.types.NDArray,
        out: qd.types.NDArray,
        count: qd.i32,
    ):
        for _ in range(count):
            if qd.static(indirect):
                helper(matrix, rhs, out)
            else:
                (matrix @ rhs).write_to(out)

    with pytest.raises(qd.QuadrantsSyntaxError, match="requires_top_level"):
        nested(qd.linalg.SparseMatrix.from_csr(*buffers[:3], shape=(3, 3)), *buffers[3:], 1)
    np.testing.assert_array_equal(buffers[-1].to_numpy(), np.full((3, 2), -777, dtype=np.float32))


def _execute(host, exact=False, difficult=False):
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


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
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


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
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


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
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


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
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
        matrix: qd.linalg.CSRMatrix,
        x: qd.types.NDArray,
        intermediate: qd.types.NDArray,
        y: qd.types.NDArray,
    ):
        for i, j in x:
            x[i, j] = qd.cast(i + 1, qd.f32) * 0.25 + qd.cast(j, qd.f32) * 0.125
        product = matrix @ x
        product.write_to(intermediate)
        (matrix @ intermediate).write_to(y)
        for i, j in y:
            y[i, j] = y[i, j] * 0.5 + 1

    pipeline(qd.linalg.SparseMatrix.from_csr(rp, ci, v, shape=(rows, rows)), x, intermediate, y)
    produced = (np.arange(rows, dtype=np.float32)[:, None] + 1) / 4
    produced = produced + np.arange(channels, dtype=np.float32)[None, :] / 8
    first = _reference_dense_f64(row_ptr, col_idx, values, produced)
    second = _reference_dense_f64(row_ptr, col_idx, values, first.astype(np.float32))
    np.testing.assert_array_equal(x.to_numpy(), produced)
    np.testing.assert_array_equal(intermediate.to_numpy(), first)
    np.testing.assert_array_equal(y.to_numpy(), second * 0.5 + 1)


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
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


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
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


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
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


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
def test_csr_host_product_borrows_buffers_and_reads_current_contents():
    host = _host_case()
    rp, ci, values, rhs, out = [_upload(array) for array in host]
    matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, values, shape=(3, 3))
    assert matrix.shape == (3, 3)
    assert (matrix.row_ptr, matrix.col_idx, matrix.values) == (rp, ci, values)
    with pytest.raises(FrozenInstanceError):
        matrix.rows = 4
    product = matrix @ rhs
    assert product.write_to(out) is None
    np.testing.assert_array_equal(out.to_numpy(), _reference_dense_f64(*host[:-1]))
    host[2][:4] *= 2
    host[3] += 0.5
    values.from_numpy(host[2])
    rhs.from_numpy(host[3])
    product.write_to(out)
    np.testing.assert_array_equal(out.to_numpy(), _reference_dense_f64(*host[:-1]))
    with pytest.raises(ValueError):
        product.write_to(rhs)


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
def test_csr_kernel_product_evaluates_rhs_once_and_reads_current_contents():
    host = _host_case()
    rp, ci, values, rhs, out = [_upload(array) for array in host]
    matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, values, shape=(3, 3))
    count = qd.ndarray(qd.i32, shape=1)
    count.fill(0)

    @qd.func
    def counted_rhs(rhs: qd.types.NDArray, count: qd.types.NDArray):
        count[0] += 1
        return rhs

    @qd.kernel
    def reuse_product(
        matrix: qd.linalg.CSRMatrix, rhs: qd.types.NDArray, out: qd.types.NDArray, count: qd.types.NDArray
    ):
        product = matrix @ counted_rhs(rhs, count)
        product.write_to(out)
        for i, j in rhs:
            rhs[i, j] += 1.0
        product.write_to(out)

    reuse_product(matrix, rhs, out, count)
    np.testing.assert_array_equal(count.to_numpy(), [1])
    host[3] += 1
    np.testing.assert_array_equal(out.to_numpy(), _reference_dense_f64(*host[:-1]))


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
def test_csr_metadata_validation():
    buffers = [_upload(array) for array in _host_case()]
    for shape in ((0, 3), (3, -1), (True, 3), (3.0, 3), (3,), (3, 3, 3), (2**31, 3)):
        with pytest.raises((ValueError, TypeError)):
            qd.linalg.SparseMatrix.from_csr(*buffers[:3], shape=shape)
    for index in range(3):
        invalid = list(buffers[:3])
        invalid[index] = qd.ndarray(qd.f32 if index < 2 else qd.i32, shape=invalid[index].shape)
        with pytest.raises(TypeError):
            qd.linalg.SparseMatrix.from_csr(*invalid, shape=(3, 3))
        invalid[index] = qd.ndarray(qd.i32 if index < 2 else qd.f32, shape=(1, 5))
        with pytest.raises((TypeError, ValueError)):
            qd.linalg.SparseMatrix.from_csr(*invalid, shape=(3, 3))
    with pytest.raises(ValueError):
        qd.linalg.SparseMatrix.from_csr(buffers[0], buffers[1], _upload(np.ones(4, np.float32)), shape=(3, 3))
    with pytest.raises(ValueError):
        qd.linalg.SparseMatrix.from_csr(*buffers[:3], shape=(2, 3))
    matrix = qd.linalg.SparseMatrix.from_csr(*buffers[:3], shape=(3, 3))
    for rhs, out in (
        (qd.ndarray(qd.f32, shape=(2, 2)), buffers[-1]),
        (buffers[-2], qd.ndarray(qd.f32, shape=(2, 2))),
        (buffers[-2], qd.ndarray(qd.f32, shape=(3, 3))),
        (qd.ndarray(qd.i32, shape=(3, 2)), buffers[-1]),
        (buffers[-2], qd.ndarray(qd.f32, shape=6)),
    ):
        with pytest.raises((TypeError, ValueError)):
            (matrix @ rhs).write_to(out)


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda], debug=False)
def test_csr_kernel_invalid_shapes_do_not_access_storage():
    host = _host_case()
    rp, ci, values, rhs, out = [_upload(array) for array in host]
    matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, values, shape=(3, 3))
    # The stored column 2 would read out of bounds from this incompatible RHS.
    short_rhs = qd.ndarray(qd.f32, shape=(2, 2))
    short_rhs.fill(1)
    for source in (short_rhs, rhs, short_rhs, rhs):
        out.fill(-777)
        _apply(matrix, source, out)
        expected = np.full((3, 2), -777, np.float32) if source is short_rhs else _reference_dense_f64(*host[:-1])
        np.testing.assert_array_equal(out.to_numpy(), expected)
    for shape in ((2, 2), (3, 1)):
        invalid_out = qd.ndarray(qd.f32, shape=shape)
        invalid_out.fill(-777)
        _apply(matrix, rhs, invalid_out)
        np.testing.assert_array_equal(invalid_out.to_numpy(), np.full(shape, -777, np.float32))


@test_utils.test(arch=[qd.cpu, qd.cuda], require=qd.extension.assertion, debug=True)
def test_csr_kernel_invalid_shapes_report_debug_error():
    rp, ci, values, _, out = [_upload(array) for array in _host_case()]
    matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, values, shape=(3, 3))
    out.fill(-777)
    with pytest.raises(qd.QuadrantsAssertionError):
        _apply(matrix, qd.ndarray(qd.f32, shape=(2, 2)), out)
    np.testing.assert_array_equal(out.to_numpy(), np.full((3, 2), -777, np.float32))


@test_utils.test(arch=[qd.cpu, qd.metal, qd.cuda])
def test_csr_nested_typed_binding_and_specialization():
    @dataclass(frozen=True)
    class Pair:
        matrix: qd.linalg.CSRMatrix
        unused: qd.types.NDArray[qd.f32, 1]

    @qd.func
    def helper(matrix: qd.linalg.CSRMatrix, rhs: qd.types.NDArray, out: qd.types.NDArray):
        product = matrix @ rhs
        product.write_to(out)

    @qd.kernel
    def nested(pair: Pair, rhs: qd.types.NDArray, out: qd.types.NDArray):
        helper(pair.matrix, rhs, out)

    for width in (3, 17):
        host = _host_case()
        host[3] = np.arange(3 * width, dtype=np.float32).reshape(3, width) / 8
        host[4] = np.full((3, width), -777, np.float32)
        rp, ci, values, rhs, out = [_upload(array) for array in host]
        matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, values, shape=(3, 3))
        nested(Pair(matrix, qd.ndarray(qd.f32, shape=1)), rhs, out)
        np.testing.assert_array_equal(out.to_numpy(), _reference_dense_f64(*host[:-1]))
    assert len(nested._primal.mapper.mapping) == 1


@test_utils.test(arch=qd.cpu)
def test_csr_native_interoperability_errors():
    rp, ci, values, _, _ = [_upload(array) for array in _host_case()]
    matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, values, shape=(3, 3))
    native = qd.linalg.SparseMatrix(3, 3)
    for operation in (lambda: native + matrix, lambda: native - matrix, lambda: native * matrix):
        with pytest.raises(qd.QuadrantsRuntimeError):
            operation()
    with pytest.raises(qd.QuadrantsRuntimeError):
        qd.linalg.SparseCG(matrix, np.ones(3, np.float32))
    with pytest.raises(qd.QuadrantsRuntimeError):
        qd.linalg.SparseSolver().compute(matrix)


@test_utils.test(arch=qd.cpu)
def test_csr_rejects_reverse_mode():
    rp, ci, values, rhs, out = [_upload(array) for array in _host_case()]
    matrix = qd.linalg.SparseMatrix.from_csr(rp, ci, values, shape=(3, 3))
    with pytest.raises(qd.QuadrantsCompilationError, match="forward"):
        _apply.grad(matrix, rhs, out)


def test_csr_public_exports():
    assert "CSRMatrix" in qd.linalg.__all__
    assert "csr_spmm" not in qd.algorithms.__all__
