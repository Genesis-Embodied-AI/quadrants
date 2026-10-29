import numpy as np
import pytest

import quadrants as qd
from quadrants.profiler.kernel_profiler import get_default_kernel_profiler

from tests.helpers import test_utils


def _tasks(name):
    qd.sync()
    profiler = get_default_kernel_profiler()
    profiler._update_records()
    return [record.name for record in profiler._traced_records if name in record.name]


@pytest.mark.parametrize("opt_level", [0, 1])
@test_utils.test(arch=qd.cuda, kernel_profiler=True, debug=True, offline_cache=False)
def test_cuda_scalar_range_bounds(opt_level):
    qd.lang.impl.current_cfg().opt_level = opt_level
    out = qd.field(qd.i32, shape=128)

    @qd.kernel
    def scalar_bounds(begin: int, chunks: int):
        for i in range(begin, 4 * chunks):
            out[i + 16] = i // chunks + i % chunks

    for begin, chunks in [(-7, 5), (3, 9), (7, 1), (0, 0)]:
        out.fill(-123)
        qd.profiler.clear_kernel_profiler_info()
        scalar_bounds(begin, chunks)
        tasks = _tasks("scalar_bounds")
        assert len(tasks) == 1
        assert tasks[0].endswith("range_for")
        expected = np.full(128, -123, dtype=np.int32)
        for i in range(begin, 4 * chunks):
            expected[i + 16] = i // chunks + i % chunks
        np.testing.assert_array_equal(out.to_numpy(), expected)


@test_utils.test(arch=qd.cuda, kernel_profiler=True, debug=True, offline_cache=False)
def test_cuda_ndarray_range_bounds():
    @qd.kernel
    def shape_bounds(out: qd.types.ndarray(dtype=qd.i32, ndim=2), offset: int):
        for i, j in qd.ndrange(out.shape[0], out.shape[1]):
            out[i, j] = i * out.shape[1] + j + offset

    for shape in [(1, 1), (7, 13), (23, 17)]:
        out = qd.ndarray(qd.i32, shape=shape)
        qd.profiler.clear_kernel_profiler_info()
        shape_bounds(out, 3)
        tasks = _tasks("shape_bounds")
        assert len(tasks) == 1
        assert tasks[0].endswith("range_for")
        np.testing.assert_array_equal(out.to_numpy(), np.arange(np.prod(shape)).reshape(shape) + 3)


@test_utils.test(arch=qd.cuda, kernel_profiler=True, debug=True, offline_cache=False)
def test_cuda_range_keeps_device_dependencies():
    count = qd.field(qd.i32, shape=())
    out = qd.field(qd.i32, shape=32)

    @qd.kernel
    def device_bound(n: int):
        count[None] = n + 1
        for i in range(count[None]):
            out[i] = n
        count[None] += 1

    for n in [3, 7]:
        out.fill(0)
        qd.profiler.clear_kernel_profiler_info()
        device_bound(n)
        tasks = _tasks("device_bound")
        assert any(name.endswith("serial") for name in tasks)
        assert count[None] == n + 2
        np.testing.assert_array_equal(out.to_numpy()[: n + 1], n)
        np.testing.assert_array_equal(out.to_numpy()[n + 1 :], 0)


@test_utils.test(arch=qd.cuda, debug=True, offline_cache=False)
def test_cuda_rematerialized_bounds_graph_replay():
    out = qd.ndarray(qd.i32, shape=64)

    @qd.kernel(graph=True)
    def graph_bounds(out: qd.types.ndarray(dtype=qd.i32, ndim=1), begin: int, end: int):
        for i in range(begin, end):
            out[i] = end

    for begin, end in [(0, 3), (5, 47), (0, 0), (7, 3)]:
        out.fill(0)
        graph_bounds(out, begin, end)
        expected = np.zeros(64, dtype=np.int32)
        expected[begin:end] = end
        np.testing.assert_array_equal(out.to_numpy(), expected)


@test_utils.test(arch=qd.cuda, require=qd.extension.adstack, debug=True, offline_cache=False)
def test_cuda_dynamic_range_autodiff():
    x = qd.field(qd.f32, shape=16, needs_grad=True)
    y = qd.field(qd.f32, shape=16, needs_grad=True)
    x.fill(2)

    @qd.kernel
    def differentiate(n: int):
        for i in range(n):
            value = x[i]
            for j in range(i % 3 + 1):
                value *= x[i]
            y[i] = value

    for n in [3, 13]:
        x.grad.fill(0)
        y.grad.fill(1)
        differentiate(n)
        differentiate.grad(n)
        expected = np.zeros(16, dtype=np.float32)
        for i in range(n):
            exponent = i % 3 + 2
            expected[i] = exponent * 2 ** (exponent - 1)
        np.testing.assert_allclose(x.grad.to_numpy(), expected)
