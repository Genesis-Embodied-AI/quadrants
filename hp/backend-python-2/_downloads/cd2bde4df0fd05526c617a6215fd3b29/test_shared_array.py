import numpy as np
import pytest

import quadrants as qd
from quadrants.math import vec4

from tests import test_utils


@test_utils.test(arch=[qd.cuda], print_full_traceback=False)
def test_large_shared_array():
    # Skip the GPUs prior to Ampere which doesn't have large dynamical shared memory.
    if qd.lang.impl.get_cuda_compute_capability() < 86:
        pytest.skip("Skip the GPUs prior to Ampere")

    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim
    v_arr = np.random.randn(N).astype(np.float32)
    d_arr = np.random.randn(N).astype(np.float32)
    a_arr = np.zeros(N).astype(np.float32)
    reference = np.zeros(N).astype(np.float32)

    @qd.kernel
    def calc(
        v: qd.types.ndarray(ndim=1),
        d: qd.types.ndarray(ndim=1),
        a: qd.types.ndarray(ndim=1),
    ):
        for i in range(N):
            acc = 0.0
            v_val = v[i]
            for j in range(N):
                acc += v_val * d[j]
            a[i] = acc

    @qd.kernel
    def calc_shared_array(
        v: qd.types.ndarray(ndim=1),
        d: qd.types.ndarray(ndim=1),
        a: qd.types.ndarray(ndim=1),
    ):
        qd.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim):
            tid = i % block_dim
            pad = qd.simt.block.SharedArray((65536 // 4,), qd.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad[tid] = d[k * block_dim + tid]
                qd.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad[j]
                qd.simt.block.sync()
            a[i] = acc

    calc(v_arr, d_arr, reference)
    calc_shared_array(v_arr, d_arr, a_arr)
    assert np.allclose(reference, a_arr)


@test_utils.test(arch=[qd.cuda, qd.vulkan, qd.amdgpu])
def test_multiple_shared_array():
    assert qd.cfg is not None
    if qd.cfg.arch == qd.amdgpu:
        pytest.xfail("failing on amd currently")
    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim * 4
    v_arr = np.random.randn(N).astype(np.float32)
    d_arr = np.random.randn(N).astype(np.float32)
    a_arr = np.zeros(N).astype(np.float32)
    reference = np.zeros(N).astype(np.float32)

    @qd.kernel
    def calc(
        v: qd.types.ndarray(ndim=1),
        d: qd.types.ndarray(ndim=1),
        a: qd.types.ndarray(ndim=1),
    ):
        for i in range(N):
            acc = 0.0
            v_val = v[i]
            for j in range(N):
                acc += v_val * d[j]
            a[i] = acc

    @qd.kernel
    def calc_shared_array(
        v: qd.types.ndarray(ndim=1),
        d: qd.types.ndarray(ndim=1),
        a: qd.types.ndarray(ndim=1),
    ):
        qd.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim * 4):
            tid = i % block_dim
            pad0 = qd.simt.block.SharedArray((block_dim,), qd.f32)
            pad1 = qd.simt.block.SharedArray((block_dim,), qd.f32)
            pad2 = qd.simt.block.SharedArray((block_dim,), qd.f32)
            pad3 = qd.simt.block.SharedArray((block_dim,), qd.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad0[tid] = d[k * block_dim * 4 + tid]
                pad1[tid] = d[k * block_dim * 4 + block_dim + tid]
                pad2[tid] = d[k * block_dim * 4 + 2 * block_dim + tid]
                pad3[tid] = d[k * block_dim * 4 + 3 * block_dim + tid]
                qd.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad0[j]
                    acc += v_val * pad1[j]
                    acc += v_val * pad2[j]
                    acc += v_val * pad3[j]
                qd.simt.block.sync()
            a[i] = acc

    calc(v_arr, d_arr, reference)
    calc_shared_array(v_arr, d_arr, a_arr)
    assert np.allclose(reference, a_arr, rtol=1e-4)


@test_utils.test(arch=[qd.cuda, qd.vulkan, qd.amdgpu])
def test_shared_array_atomics():
    N = 256
    block_dim = 32

    @qd.kernel
    def atomic_test(out: qd.types.ndarray()):
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            val = tid
            sharr = qd.simt.block.SharedArray((block_dim,), qd.i32)
            sharr[tid] = val
            qd.simt.block.sync()
            sharr[0] += val
            qd.simt.block.sync()
            out[i] = sharr[tid]

    arr = qd.ndarray(qd.i32, (N))
    atomic_test(arr)
    qd.sync()
    sum = block_dim * (block_dim - 1) // 2
    assert arr[0] == sum
    assert arr[32] == sum
    assert arr[128] == sum
    assert arr[224] == sum


@test_utils.test(arch=[qd.cuda])
def test_shared_array_tensor_type():
    data_type = vec4
    block_dim = 16
    N = 64

    y = qd.Vector.field(4, dtype=qd.f32, shape=(block_dim))

    @qd.kernel
    def test():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            val = qd.Vector([1.0, 2.0, 3.0, 4.0])

            shared_mem = qd.simt.block.SharedArray((block_dim), data_type)
            shared_mem[tid] = val
            qd.simt.block.sync()

            y[tid] += shared_mem[tid]

    test()
    assert (y.to_numpy()[0] == [4.0, 8.0, 12.0, 16.0]).all()


@test_utils.test(arch=[qd.cuda], debug=True)
def test_shared_array_matrix():
    @qd.kernel
    def foo():
        for x in range(10):
            shared = qd.simt.block.SharedArray((10,), dtype=qd.math.vec3)
            shared[x] = qd.Vector([x + 1, x + 2, x + 3])
            assert shared[x].z == x + 3
            assert (shared[x] == qd.Vector([x + 1, x + 2, x + 3])).all()

            print(shared[x].z)
            print(shared[x])

    foo()
