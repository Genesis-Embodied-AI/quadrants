"""Regression coverage for the AMDGPU launcher's skip-redundant-RuntimeContext-HtoD optimization.

See ``quadrants/runtime/amdgpu/kernel_launcher.cpp``. On the default-stream path, repeated same-handle launches
reuse a cached ``RuntimeContext`` and skip the per-launch host->device copy; results must be identical whether or
not a launch skipped the upload.

The subtle property under test: per-launch-varying data (ndarray/scalar args) reaches the device via the
separately-uploaded ``arg_buffer``, not the cached ``RuntimeContext``, which holds only the stable ``arg_buffer``
device address and so keeps hitting the cache. The genuine re-upload path -- a ``checkpoint_*_ptr`` mutation, which
lives inside ``RuntimeContext`` -- is covered by ``tests/python/test_checkpoint.py``.

On non-AMDGPU backends the same kernels run through the generic launcher as a cross-backend baseline.
"""

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_repeated_launch_argless_result_less():
    # Arg-less, result-less kernel: arg_buffer==nullptr normalization keeps the context stable, so identical
    # launches hit the cache after the first.
    x = qd.field(qd.i32, shape=())
    x[None] = 0

    @qd.kernel
    def inc():
        x[None] += 1

    for _ in range(64):
        inc()
    assert x[None] == 64


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_repeated_launch_reduction_result_buffer():
    # Reduction kernel launched repeatedly: result_buffer is pinned to the persistent device buffer, so the struct
    # stays stable even though the kernel writes results back. The scalar arg changes each iteration but rides
    # arg_buffer, so the cache keeps hitting while the reduction must still track the freshly filled data.
    n = 1024
    f = qd.field(qd.f32, shape=n)

    @qd.kernel
    def fill(c: qd.f32):
        for i in f:
            f[i] = c

    @qd.kernel
    def total() -> qd.f32:
        s = 0.0
        for i in f:
            s += f[i]
        return s

    for k in range(1, 17):
        fill(float(k))
        assert total() == pytest.approx(float(n * k))


@test_utils.test(arch=[qd.cpu, qd.cuda, qd.amdgpu])
def test_repeated_launch_changing_ndarray_arg_buffer_split():
    # Same compiled handle against different ndarrays: the changing data pointers ride arg_buffer, while
    # RuntimeContext holds only the stable arg_buffer address and keeps hitting the cache. The results must still
    # reflect the correct per-launch buffer.
    n = 256
    arrs = [qd.ndarray(qd.f32, shape=n) for _ in range(6)]
    for j, a in enumerate(arrs):
        a.from_numpy(np.full(n, float(j), dtype=np.float32))

    @qd.kernel
    def add_one(a: qd.types.ndarray(dtype=qd.f32, ndim=1)):
        for i in range(a.shape[0]):
            a[i] += 1.0

    for a in arrs:
        add_one(a)

    for j, a in enumerate(arrs):
        assert np.allclose(a.to_numpy(), float(j) + 1.0)
