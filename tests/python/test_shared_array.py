import numpy as np
import pytest

import quadrants as qd
from quadrants.math import vec4

from tests import test_utils


@pytest.mark.parametrize(
    "shape_offset,dtype2",
    [
        (0, qd.i8),  # same shape, same dtype — triggers the singleton bug
        (0, qd.f32),  # same shape, different dtype
        (1, qd.i8),  # different shape, same dtype
    ],
)
@test_utils.test(arch=[qd.cuda, qd.metal])
def test_shared_array_not_accumulated_across_offloads(shape_offset, dtype2):
    # Execute 2 successive offloaded tasks both allocating more than half of
    # the maximum shared memory available on the device to make sure shared
    # memory is properly deallocated and per-task address offset is correctly
    # reset.
    # Note that in practice, there is a different code path for "statically"
    # allocated shared memory (which has fixed size 48KB on CUDA) and
    # "dynamically" allocated shared memory. Some CUDA GPUs support larger
    # shared memory allocation via dynamic allocation. In such a case, if
    # some task-level GPU kernel requests more than 48KB, the entire shared
    # array is dynamically allocated. This requires creating a new LLVM
    # array type of size 0 with the same dtype as the original tensor (the
    # actual size is passed at kernel launch time). Creation of this special
    # type was previously corrupting the original cached tensor type when
    # several tasks using shared arrays with the same shape and dtype were
    # involved. The cached tensor type is a singleton keyed by (shape, dtype)
    # in TypeFactory, so the corruption only occurs when multiple offloaded
    # tasks allocate shared arrays with identical shape and dtype. If either
    # differs, each task gets a distinct tensor type instance and is
    # unaffected.

    block_dim = 32
    if qd.cfg.arch == qd.cuda:
        max_shared_bytes = qd.lang.impl.get_max_shared_memory_bytes()
    else:
        # Metal guarantees 32KB of threadgroup memory
        max_shared_bytes = 32 * 1024
    # 75% of max shared memory in bytes
    shared_array_bytes = int(0.75 * max_shared_bytes)
    shared_array_size = shared_array_bytes // qd._lib.core.data_type_size(qd.i8)
    shared_array_size_2 = shared_array_bytes // qd._lib.core.data_type_size(dtype2) + shape_offset

    @qd.kernel
    def kern(out: qd.types.ndarray):
        qd.loop_config(block_dim=block_dim)
        for tid in range(block_dim):
            buf = qd.simt.block.SharedArray((shared_array_size,), qd.i8)
            i = tid
            while i < shared_array_size:
                buf[i] = qd.cast(i % 127, qd.i8)
                i += block_dim
            qd.simt.block.sync()
            out[tid] = qd.cast(buf[tid], qd.i32)

        qd.loop_config(block_dim=block_dim)
        for tid in range(block_dim):
            buf = qd.simt.block.SharedArray((shared_array_size_2,), dtype2)
            i = tid
            while i < shared_array_size_2:
                buf[i] = qd.cast((i % 127) * 2, dtype2)
                i += block_dim
            qd.simt.block.sync()
            out[tid] = out[tid] + qd.cast(buf[tid], qd.i32)

    out = qd.ndarray(dtype=qd.i32, shape=(block_dim,))
    kern(out)

    expected = 3 * np.arange(block_dim, dtype=np.int32)
    assert np.array_equal(out.to_numpy(), expected)


@pytest.mark.parametrize("gpu_graph", [False, True])
@test_utils.test(arch=[qd.cuda], print_full_traceback=False)
def test_large_shared_array(gpu_graph):
    if qd.lang.impl.get_max_shared_memory_bytes() < 65536:
        pytest.skip("Device does not support large dynamic shared memory")

    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim
    v_np = np.random.randn(N).astype(np.float32)
    d_np = np.random.randn(N).astype(np.float32)

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

    @qd.kernel(gpu_graph=gpu_graph)
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

    # gpu_graph requires device-resident ndarrays
    v_arr = qd.ndarray(dtype=qd.f32, shape=(N,))
    d_arr = qd.ndarray(dtype=qd.f32, shape=(N,))
    v_arr.from_numpy(v_np)
    d_arr.from_numpy(d_np)

    reference = qd.ndarray(dtype=qd.f32, shape=(N,))
    a_arr = qd.ndarray(dtype=qd.f32, shape=(N,))
    calc(v_arr, d_arr, reference)
    calc_shared_array(v_arr, d_arr, a_arr)
    assert np.allclose(reference.to_numpy(), a_arr.to_numpy())


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
