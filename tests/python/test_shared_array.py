import math

import numpy as np
import pytest

import quadrants as qd
from quadrants.math import vec4

from tests import test_utils


@pytest.mark.parametrize(
    "num_dim, first_shape_delta_size, dtype1, dtype2",
    [
        (1, 0, qd.i8, qd.i8),  # same shape, same dtype — triggers the singleton bug
        (2, 0, qd.i8, qd.i8),  # same shape, same dtype — triggers the singleton bug
        (1, 0, qd.f32, qd.f32),  # same shape, same dtype — triggers the singleton bug
        (2, 0, qd.u32, qd.u32),  # same shape, same dtype — triggers the singleton bug
        (1, 0, qd.i8, qd.f32),  # same shape, different dtype
        (1, 1, qd.i8, qd.i8),  # different shape, same dtype
        (2, 0, qd.u32, qd.u32),  # different shape, same dtype
        (1, 1, qd.i8, qd.f32),  # different shape, different dtype
    ],
)
@test_utils.test(arch=[qd.cuda, qd.metal])
def test_shared_array_not_accumulated_across_offloads(num_dim, first_shape_delta_size, dtype1, dtype2):
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
    # involved.
    # The cached tensor type is a singleton keyed by (shape, dtype) in
    # TypeFactory, so the corruption only occurs when multiple offloaded tasks
    # allocate shared arrays with identical shape and dtype. If either differs,
    # each task gets a distinct tensor type instance and is unaffected.
    # The corruption would affect ALL tensors of the exact same type, not just
    # shared memory, because this information is not part of the type but stored
    # in 'stmt->is_shared'. However, it is only an issue when shared memories
    # from different offloaded tasks are sharing the same tensor type because at
    # this point in the codegen path, the IR structure (index calculations,
    # strides, offsets) has already been baked into the IR statements. The only
    # place that reads the type during codegen is the AllocaStmt visitor itself,
    # which uses 'get_num_elements()' to decide between the static/dynamic path
    # and to compute the LLVM type size.

    block_dim = 32
    max_shared_bytes = qd.lang.impl.get_max_shared_memory_bytes(is_lowerbound_ok=True)
    # 75% of max shared memory in bytes, converted to element counts
    shared_array_bytes = int(0.75 * max_shared_bytes)
    num_elems_1 = shared_array_bytes // qd._lib.core.data_type_size(dtype1) + first_shape_delta_size
    num_elems_2 = shared_array_bytes // qd._lib.core.data_type_size(dtype2)

    # Build 1D or 2D shape tuples with the same total number of elements.
    # For 2D, split into (block_dim, num_elems // block_dim).
    if num_dim == 1:
        shape_1 = (num_elems_1,)
        shape_2 = (num_elems_2,)
    else:
        shape_1 = (block_dim, num_elems_1 // block_dim)
        shape_2 = (block_dim, num_elems_2 // block_dim)
        num_elems_1 = math.prod(shape_1)
        num_elems_2 = math.prod(shape_2)

    # Each offloaded task cooperatively fills a large shared array with an
    # LCG sequence, syncs, then each thread sums a contiguous chunk written
    # by other threads. This forces the entire shared array to be materialized
    # — the compiler cannot short-circuit it.
    chunk_size_1 = num_elems_1 // block_dim
    chunk_size_2 = num_elems_2 // block_dim
    cols_1 = shape_1[-1]
    cols_2 = shape_2[-1]

    @qd.kernel
    def kern(out: qd.types.ndarray):
        qd.loop_config(block_dim=block_dim)
        for tid in range(block_dim):
            buf = qd.simt.block.SharedArray(shape_1, dtype1)
            i = tid
            while i < num_elems_1:
                if qd.static(num_dim == 2):
                    buf[i // cols_1, i % cols_1] = qd.cast((i * 1103515245 + 12345) % 128, dtype1)
                else:
                    buf[i] = qd.cast((i * 1103515245 + 12345) % 128, dtype1)
                i += block_dim
            qd.simt.block.sync()
            acc = 0
            j = tid * chunk_size_1
            end = j + chunk_size_1
            while j < end:
                if qd.static(num_dim == 2):
                    acc += qd.cast(buf[j // cols_1, j % cols_1], qd.i32)
                else:
                    acc += qd.cast(buf[j], qd.i32)
                j += 1
            out[tid] = acc

        qd.loop_config(block_dim=block_dim)
        for tid in range(block_dim):
            buf = qd.simt.block.SharedArray(shape_2, dtype2)
            i = tid
            while i < num_elems_2:
                if qd.static(num_dim == 2):
                    buf[i // cols_2, i % cols_2] = qd.cast((i * 1103515245 + 12345) % 128, dtype2)
                else:
                    buf[i] = qd.cast((i * 1103515245 + 12345) % 128, dtype2)
                i += block_dim
            qd.simt.block.sync()
            acc = 0
            j = tid * chunk_size_2
            end = j + chunk_size_2
            while j < end:
                if qd.static(num_dim == 2):
                    acc += qd.cast(buf[j // cols_2, j % cols_2], qd.i32)
                else:
                    acc += qd.cast(buf[j], qd.i32)
                j += 1
            out[tid] = out[tid] + acc

    out = qd.ndarray(dtype=qd.i32, shape=(block_dim,))
    kern(out)

    # Compute expected values on host
    vals1 = np.array([(i * 1103515245 + 12345) % 128 for i in range(num_elems_1)], dtype=np.int32)
    vals2 = np.array([(i * 1103515245 + 12345) % 128 for i in range(num_elems_2)], dtype=np.int32)
    expected = np.array(
        [
            vals1[t * chunk_size_1 : (t + 1) * chunk_size_1].sum()
            + vals2[t * chunk_size_2 : (t + 1) * chunk_size_2].sum()
            for t in range(block_dim)
        ],
        dtype=np.int32,
    )
    assert np.array_equal(out.to_numpy(), expected)


@pytest.mark.parametrize("graph", [False, True])
@test_utils.test(arch=[qd.cuda], print_full_traceback=False)
def test_large_shared_array(graph):
    # Any shared memory larger than 48kB requires so-called "dynamic
    # allocation", which is a special feature that requires toggling some opt-in
    # flag in gpu kernel context and is currently only supported on CUDA. In
    # practice, all GPUs supporting this feature have a max shared memory size
    # of 64kB or more, so hardcoding this value in the unit test guarantees to
    # exercise this feature, while being safe and consistent across all GPUs.
    shared_bytes = 65536

    if qd.lang.impl.get_max_shared_memory_bytes(is_lowerbound_ok=True) < shared_bytes:
        pytest.skip("Device does not support large dynamic shared memory")

    block_dim = 128
    nBlocks = 64
    N = nBlocks * block_dim
    v_np = np.random.randn(N).astype(np.float32)
    d_np = np.random.randn(N).astype(np.float32)

    # Compute a[i] = v[i] * sum(d), i.e. scale each v[i] by the sum of d.
    # The reference uses a naive double loop; the shared-memory version tiles
    # d into shared memory blocks for cooperative loading.

    @qd.kernel
    def scaled_reduce_native(
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

    @qd.kernel(graph=graph)
    def scaled_reduce_shared(
        v: qd.types.ndarray(ndim=1),
        d: qd.types.ndarray(ndim=1),
        a: qd.types.ndarray(ndim=1),
    ):
        qd.loop_config(block_dim=block_dim)
        for i in range(nBlocks * block_dim):
            tid = i % block_dim
            pad = qd.simt.block.SharedArray((shared_bytes // 4,), qd.f32)
            acc = 0.0
            v_val = v[i]
            for k in range(nBlocks):
                pad[tid] = d[k * block_dim + tid]
                qd.simt.block.sync()
                for j in range(block_dim):
                    acc += v_val * pad[j]
                qd.simt.block.sync()
            a[i] = acc

    # graph requires device-resident arrays (qd.ndarray or CUDA torch
    # tensors), not host-resident numpy arrays
    v_arr = qd.ndarray(dtype=qd.f32, shape=(N,))
    d_arr = qd.ndarray(dtype=qd.f32, shape=(N,))
    v_arr.from_numpy(v_np)
    d_arr.from_numpy(d_np)

    reference = qd.ndarray(dtype=qd.f32, shape=(N,))
    a_arr = qd.ndarray(dtype=qd.f32, shape=(N,))
    scaled_reduce_native(v_arr, d_arr, reference)
    scaled_reduce_shared(v_arr, d_arr, a_arr)
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
