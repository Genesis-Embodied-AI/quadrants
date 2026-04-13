import platform

import numpy as np
import pytest
from pytest import approx

import quadrants as qd
from quadrants.lang.simt import subgroup

from tests import test_utils


def _skip_if_f64_unsupported(dtype):
    if dtype != qd.f64:
        return
    arch = qd.lang.impl.current_cfg().arch
    if arch == qd.metal:
        pytest.skip("Metal does not support f64 in buffer-backed snode/kernel I/O")
    if arch == qd.vulkan and platform.system() == "Darwin":
        pytest.skip("MoltenVK does not support f64")


@test_utils.test(arch=qd.cuda)
def test_all_nonzero():
    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.all_nonzero(qd.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 1
        a[i] = -1

    foo()

    for i in range(32):
        assert a[i] == 1

    b[np.random.randint(0, 32)] = 0

    foo()

    for i in range(32):
        assert a[i] == 0


@test_utils.test(arch=qd.cuda)
def test_sync_all_nonzero():
    a = qd.field(dtype=qd.i32, shape=256)
    b = qd.field(dtype=qd.i32, shape=256)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=256)
        for i in range(256):
            a[i] = qd.simt.block.sync_all_nonzero(b[i])

    for i in range(256):
        b[i] = 1
        a[i] = -1

    foo()

    for i in range(256):
        assert a[i] == 1

    b[np.random.randint(0, 256)] = 0

    foo()

    for i in range(256):
        assert a[i] == 0


@test_utils.test(arch=qd.cuda)
def test_any_nonzero():
    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.any_nonzero(qd.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 0
        a[i] = -1

    foo()

    for i in range(32):
        assert a[i] == 0

    b[np.random.randint(0, 32)] = 1

    foo()

    for i in range(32):
        assert a[i] == 1


@test_utils.test(arch=qd.cuda)
def test_sync_any_nonzero():
    a = qd.field(dtype=qd.i32, shape=256)
    b = qd.field(dtype=qd.i32, shape=256)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=256)
        for i in range(256):
            a[i] = qd.simt.block.sync_any_nonzero(b[i])

    for i in range(256):
        b[i] = 0
        a[i] = -1

    foo()

    for i in range(256):
        assert a[i] == 0

    b[np.random.randint(0, 256)] = 1

    foo()

    for i in range(256):
        assert a[i] == 1


@test_utils.test(arch=qd.cuda)
def test_sync_count_nonzero():
    a = qd.field(dtype=qd.i32, shape=256)
    b = qd.field(dtype=qd.i32, shape=256)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=256)
        for i in range(256):
            a[i] = qd.simt.block.sync_count_nonzero(b[i])

    for i in range(256):
        b[i] = 0
        a[i] = -1

    foo()

    for i in range(256):
        assert a[i] == 0

    random_idx_count = np.random.randint(0, 256)
    random_idx = np.random.choice(256, random_idx_count, replace=False)
    for i in range(random_idx_count):
        b[random_idx[i]] = 1

    foo()

    for i in range(256):
        assert a[i] == random_idx_count


@test_utils.test(arch=qd.cuda)
def test_unique():
    a = qd.field(dtype=qd.u32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def check():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.unique(qd.u32(0xFFFFFFFF), b[i])

    for i in range(32):
        b[i] = 0
        a[i] = -1

    check()

    for i in range(32):
        assert a[i] == 1

    for i in range(32):
        b[i] = i + 100

    check()

    for i in range(32):
        assert a[i] == 1

    b[np.random.randint(0, 32)] = 0

    check()

    for i in range(32):
        assert a[i] == 0


@test_utils.test(arch=qd.cuda)
def test_ballot():
    a = qd.field(dtype=qd.u32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.ballot(b[i])

    key = 0
    for i in range(32):
        b[i] = i % 2
        key += b[i] * pow(2, i)

    foo()

    for i in range(32):
        assert a[i] == key


@test_utils.test(arch=qd.cuda)
def test_shfl_sync_i32():
    a = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_sync_i32(qd.u32(0xFFFFFFFF), a[i], 0)

    for i in range(32):
        a[i] = i + 1

    foo()

    for i in range(1, 32):
        assert a[i] == 1


@test_utils.test(arch=qd.cuda)
def test_shfl_sync_f32():
    a = qd.field(dtype=qd.f32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_sync_f32(qd.u32(0xFFFFFFFF), a[i], 0)

    for i in range(32):
        a[i] = i + 1.0

    foo()

    for i in range(1, 32):
        assert a[i] == approx(1.0, abs=1e-4)


@test_utils.test(arch=qd.cuda)
def test_shfl_up_i32():
    # TODO
    pass


@test_utils.test(arch=qd.cuda)
def test_shfl_xor_i32():
    a = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            for j in range(5):
                offset = 1 << j
                a[i] += qd.simt.warp.shfl_xor_i32(qd.u32(0xFFFFFFFF), a[i], offset)

    value = 0
    for i in range(32):
        a[i] = i
        value += i

    foo()

    for i in range(32):
        assert a[i] == value


@test_utils.test(arch=qd.cuda)
def test_shfl_down_i32():
    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_down_i32(qd.u32(0xFFFFFFFF), b[i], 1)

    for i in range(32):
        b[i] = i * i

    foo()

    for i in range(31):
        assert a[i] == b[i + 1]

    # TODO: make this test case stronger


@test_utils.test(arch=qd.cuda)
def test_shfl_up_i32():
    a = qd.field(dtype=qd.i32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_up_i32(qd.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i

    foo()

    for i in range(1, 32):
        assert a[i] == (i - 1) * (i - 1)


@test_utils.test(arch=qd.cuda)
def test_shfl_up_f32():
    a = qd.field(dtype=qd.f32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_up_f32(qd.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i * 0.9

    foo()

    for i in range(1, 32):
        assert a[i] == approx((i - 1) * (i - 1) * 0.9, abs=1e-4)


@test_utils.test(arch=qd.cuda)
def test_shfl_down_f32():
    a = qd.field(dtype=qd.f32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = qd.simt.warp.shfl_down_f32(qd.u32(0xFFFFFFFF), a[i], 1)

    for i in range(32):
        a[i] = i * i * 0.9

    foo()

    for i in range(31):
        assert a[i] == approx((i + 1) * (i + 1) * 0.9, abs=1e-4)


@test_utils.test(arch=qd.cuda)
def test_match_any():
    # Skip match_any test for Pascal
    if qd.lang.impl.get_cuda_compute_capability() < 70:
        pytest.skip("match_any not supported on Pascal")

    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(16):
            a[i] = 0
            a[i + 16] = 1

        for i in range(32):
            b[i] = qd.simt.warp.match_any(qd.u32(0xFFFFFFFF), a[i])

    foo()

    for i in range(16):
        assert b[i] == 65535
    for i in range(16):
        assert b[i + 16] == (2**32 - 2**16)


@test_utils.test(arch=qd.cuda)
def test_match_all():
    # Skip match_all test for Pascal
    if qd.lang.impl.get_cuda_compute_capability() < 70:
        pytest.skip("match_all not supported on Pascal")

    a = qd.field(dtype=qd.i32, shape=32)
    b = qd.field(dtype=qd.u32, shape=32)
    c = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = 1
        for i in range(32):
            b[i] = qd.simt.warp.match_all(qd.u32(0xFFFFFFFF), a[i])

        a[0] = 2
        for i in range(32):
            c[i] = qd.simt.warp.match_all(qd.u32(0xFFFFFFFF), a[i])

    foo()

    for i in range(32):
        assert b[i] == (2**32 - 1)

    for i in range(32):
        assert c[i] == 0


@test_utils.test(arch=qd.cuda)
def test_active_mask():
    a = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=16)
        for i in range(32):
            a[i] = qd.simt.warp.active_mask()

    foo()

    for i in range(32):
        assert a[i] == 65535


@test_utils.test(arch=qd.cuda)
def test_warp_sync():
    a = qd.field(dtype=qd.u32, shape=32)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=32)
        for i in range(32):
            a[i] = i
        qd.simt.warp.sync(qd.u32(0xFFFFFFFF))
        for i in range(16):
            a[i] = a[i + 16]

    foo()

    for i in range(32):
        assert a[i] == i % 16 + 16


@test_utils.test(arch=qd.cuda)
def test_block_sync():
    N = 1024
    a = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            # Make the 0-th thread runs slower intentionally
            for j in range(N - i):
                a[i] = j
            qd.simt.block.sync()
            if i > 0:
                a[i] = a[0]

    foo()

    for i in range(N):
        assert a[i] == N - 1


# TODO: replace this with a stronger test case
@test_utils.test(arch=qd.cuda)
def test_grid_memfence():
    N = 1000
    BLOCK_SIZE = 1
    a = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        block_counter = 0
        qd.loop_config(block_dim=BLOCK_SIZE)
        for i in range(N):
            a[i] = 1
            qd.simt.grid.memfence()

            # Execute a prefix sum after all blocks finish
            actual_order_of_block = qd.atomic_add(block_counter, 1)
            if actual_order_of_block == N - 1:
                for j in range(1, N):
                    a[j] += a[j - 1]

    foo()

    for i in range(N):
        assert a[i] == i + 1


# Higher level primitives test
def _test_subgroup_reduce(op, group_op, np_op, size, initial_value, dtype):
    field = qd.field(dtype, (size))
    if dtype == qd.i32 or dtype == qd.i64:
        rand_values = np.random.randint(1, 100, size=(size))
        field.from_numpy(rand_values)
    if dtype == qd.f32 or dtype == qd.f64:
        rand_values = np.random.random(size=(size)).astype(np.float32)
        field.from_numpy(rand_values)

    @qd.kernel
    def reduce_all() -> dtype:
        sum = qd.cast(initial_value, dtype)
        for i in field:
            value = field[i]
            reduce_value = group_op(value)
            if subgroup.elect():
                op(sum, reduce_value)
        return sum

    if dtype == qd.i32 or dtype == qd.i64:
        assert reduce_all() == np_op(rand_values)
    else:
        assert reduce_all() == approx(np_op(rand_values), 3e-4)


# We use 2677 as size because it is a prime number
# i.e. any device other than a subgroup size of 1 should have one non active group


@test_utils.test(arch=qd.vulkan, exclude=[(qd.vulkan, "Darwin")])
def test_subgroup_reduction_add_i32():
    _test_subgroup_reduce(qd.atomic_add, subgroup.reduce_add, np.sum, 2677, 0, qd.i32)


@test_utils.test(arch=qd.vulkan)
def test_subgroup_reduction_add_f32():
    _test_subgroup_reduce(qd.atomic_add, subgroup.reduce_add, np.sum, 2677, 0, qd.f32)


# @test_utils.test(arch=qd.vulkan)
# def test_subgroup_reduction_mul_i32():
#     _test_subgroup_reduce(qd.atomic_add, subgroup.reduce_mul, np.prod, 8, 1, qd.f32)


@test_utils.test(arch=qd.vulkan, exclude=[(qd.vulkan, "Darwin")])
def test_subgroup_reduction_max_i32():
    _test_subgroup_reduce(qd.atomic_max, subgroup.reduce_max, np.max, 2677, 0, qd.i32)


@test_utils.test(arch=qd.vulkan)
def test_subgroup_reduction_max_f32():
    _test_subgroup_reduce(qd.atomic_max, subgroup.reduce_max, np.max, 2677, 0, qd.f32)


@test_utils.test(arch=qd.vulkan)
def test_subgroup_reduction_min_f32():
    _test_subgroup_reduce(qd.atomic_max, subgroup.reduce_max, np.max, 2677, 0, qd.f32)


def _init_field(field, n, dtype):
    for i in range(n):
        field[i] = (i + 1) if dtype == qd.i32 else 1.0000000000001 * (i + 1)


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_broadcast(dtype):
    """Broadcast lane 0's value to all lanes via shuffle."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    a = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = subgroup.shuffle(a[i], qd.u32(0))

    _init_field(a, N, dtype)

    expected = a[0]
    foo()

    # Lanes 0-3 are guaranteed to be in the same subgroup (min size is 4).
    for i in range(4):
        assert a[i] == expected


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_roundtrip(dtype):
    """Each lane shuffles to its own ID (identity shuffle)."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    a = qd.field(dtype=dtype, shape=N)
    diff = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            result = subgroup.shuffle(a[i], qd.cast(lane, qd.u32))
            diff[i] = result - a[i]

    _init_field(a, N, dtype)

    foo()

    for i in range(N):
        assert diff[i] == 0


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_cross_lane(dtype):
    """Each lane in a group of 4 reads from a different lane in the group."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            group_base = (lane // 4) * 4
            # Each lane reads from lane (group_base + 3 - probe_id),
            # i.e. lane 0 reads from lane 3, lane 1 from lane 2, etc.
            src_lane = group_base + 3 - lane % 4
            dst[i] = subgroup.shuffle(src[i], qd.cast(src_lane, qd.u32))

    _init_field(src, N, dtype)

    foo()

    # Lanes 0-3 are guaranteed to be in the same subgroup.
    # Lane 0 should have lane 3's value, lane 1 should have lane 2's, etc.
    assert dst[0] == src[3]
    assert dst[1] == src[2]
    assert dst[2] == src[1]
    assert dst[3] == src[0]


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_xor_pattern(dtype):
    """XOR shuffle: each lane reads from lane_id ^ 1 (swap neighbors)."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            dst[i] = subgroup.shuffle(src[i], qd.cast(lane ^ 1, qd.u32))

    _init_field(src, N, dtype)

    foo()

    # Lanes 0-3 are in the same subgroup: 0<->1 swap, 2<->3 swap
    assert dst[0] == src[1]
    assert dst[1] == src[0]
    assert dst[2] == src[3]
    assert dst[3] == src[2]


@test_utils.test(arch=qd.gpu)
def test_subgroup_invocation_id_range():
    """Verify invocation IDs are non-negative."""
    N = 64
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = subgroup.invocation_id()

    foo()

    for i in range(N):
        assert 0 <= a[i]


@test_utils.test(arch=qd.gpu)
def test_subgroup_size_range():
    """min/max subgroup size must match hardware expectations."""
    arch = qd.lang.impl.current_cfg().arch
    min_sg = qd.simt.min_subgroup_size()
    max_sg = qd.simt.max_subgroup_size()

    if arch == qd.cuda:
        assert min_sg == 32 and max_sg == 32
    elif arch == qd.vulkan:
        assert 1 <= min_sg <= max_sg <= 128
    elif arch == qd.metal:
        assert min_sg == 32 and max_sg == 32
    elif arch == qd.amdgpu:
        assert min_sg >= 32 and max_sg >= min_sg


@pytest.mark.parametrize("sg_size", [8, 16, 32, 64])
@test_utils.test()
def test_subgroup_size_validation(sg_size):
    """For each subgroup size, check it's accepted or rejected depending on arch.
    When accepted on a GPU backend, also verify invocation_id() correctness."""
    arch = qd.lang.impl.current_cfg().arch

    _valid = {
        qd.cuda: {32},
        qd.metal: {32},
        qd.amdgpu: {32},
    }

    if arch in (qd.cpu, qd.x64, qd.arm64):
        should_raise = True
        match = "not supported on CPU"
    elif arch in _valid:
        should_raise = sg_size not in _valid[arch]
        match = "not valid"
    elif arch == qd.vulkan:
        min_sg = qd.simt.min_subgroup_size()
        max_sg = qd.simt.max_subgroup_size()
        should_raise = not (min_sg <= sg_size <= max_sg)
        match = "not valid for Vulkan"
    else:
        pytest.skip(f"untested arch {arch}")
        return

    out = qd.ndarray(dtype=qd.i32, shape=(sg_size,))

    @qd.kernel
    def k(result: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=sg_size, subgroup_size=sg_size)
        for i in range(sg_size):
            result[i] = qd.simt.subgroup.invocation_id()

    if should_raise:
        with pytest.raises(ValueError, match=match):
            k(out)
    else:
        k(out)
        ids = out.to_numpy()
        assert set(ids) == set(
            range(sg_size)
        ), f"subgroup_size={sg_size}: expected IDs 0..{sg_size - 1}, got {sorted(set(ids))}"


@test_utils.test(arch=qd.vulkan)
def test_vulkan_default_subgroup_size_32():
    """Without explicit subgroup_size, Vulkan should pin subgroup size to 32
    (when VK_EXT_subgroup_size_control is available), producing IDs 0..31."""
    N = 32
    out = qd.ndarray(dtype=qd.i32, shape=(N,))

    @qd.kernel
    def read_ids(result: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=N)
        for i in range(N):
            result[i] = qd.simt.subgroup.invocation_id()

    read_ids(out)
    ids = out.to_numpy()
    assert set(ids) == set(range(N)), f"Expected IDs 0..{N-1}, got {sorted(set(ids))}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_size_with_struct_for():
    """subgroup_size should propagate through struct-for loops over fields."""
    N = 32
    x = qd.field(qd.i32, shape=(N,))
    ids_field = qd.field(qd.i32, shape=(N,))

    @qd.kernel
    def fill():
        for i in x:
            x[i] = i

    @qd.kernel
    def read_ids():
        qd.loop_config(block_dim=N, subgroup_size=32)
        for i in x:
            ids_field[i] = qd.simt.subgroup.invocation_id()

    fill()
    read_ids()
    ids = ids_field.to_numpy()
    assert all(0 <= v < 32 for v in ids), f"Got out-of-range invocation IDs: {ids}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_size_with_adaptive_block_dim():
    """subgroup_size should work when block_dim is not explicitly set
    (i.e. block_dim_adaptive=True, the default)."""
    N = 32
    out = qd.ndarray(dtype=qd.i32, shape=(N,))

    @qd.kernel
    def read_ids(result: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(subgroup_size=32)
        for i in range(N):
            result[i] = qd.simt.subgroup.invocation_id()

    read_ids(out)
    ids = out.to_numpy()
    assert all(0 <= v < 32 for v in ids), f"Got out-of-range invocation IDs: {ids}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_size_survives_second_call():
    """subgroup_size must produce correct results on second call (exercises
    offline cache serialization round-trip for the kernel)."""
    N = 32
    out = qd.ndarray(dtype=qd.i32, shape=(N,))

    @qd.kernel
    def read_ids(result: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=N, subgroup_size=N)
        for i in range(N):
            result[i] = qd.simt.subgroup.invocation_id()

    read_ids(out)
    first_ids = out.to_numpy().copy()
    assert set(first_ids) == set(range(N))

    read_ids(out)
    second_ids = out.to_numpy()
    assert set(second_ids) == set(range(N))
    np.testing.assert_array_equal(first_ids, second_ids)


@test_utils.test(arch=qd.gpu, offline_cache=False)
def test_subgroup_size_in_ir_dump(tmp_path, monkeypatch):
    """subgroup_size should appear in IR dumps when set."""
    monkeypatch.setenv("QD_DUMP_IR", "1")
    qd.lang.impl.current_cfg().debug_dump_path = str(tmp_path)

    N = 32
    out = qd.ndarray(dtype=qd.i32, shape=(N,))

    @qd.kernel
    def k(result: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=N, subgroup_size=32)
        for i in range(N):
            result[i] = i

    k(out)
    qd.sync()

    offload_files = list(tmp_path.glob("*after_offload*"))
    assert len(offload_files) > 0, f"No after_offload IR dumps in {tmp_path}"
    combined = "\n".join(f.read_text() for f in offload_files)
    assert "subgroup_size=32" in combined, f"subgroup_size=32 not found in IR dump:\n{combined}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_size_with_graph():
    """subgroup_size should work with CUDA graphs (graph=True)."""
    N = 32
    out = qd.ndarray(dtype=qd.i32, shape=(N,))

    @qd.kernel(graph=True)
    def read_ids(result: qd.types.ndarray(dtype=qd.i32, ndim=1)):
        qd.loop_config(block_dim=N, subgroup_size=32)
        for i in range(N):
            result[i] = qd.simt.subgroup.invocation_id()

    read_ids(out)
    ids = out.to_numpy()
    assert set(ids) == set(range(N)), f"Expected IDs 0..{N-1}, got {sorted(set(ids))}"

    read_ids(out)
    ids2 = out.to_numpy()
    np.testing.assert_array_equal(ids, ids2)


@test_utils.test(arch=qd.vulkan)
def test_vulkan_subgroup_id_survives_reinit():
    """Regression test: SubgroupLocalInvocationId must stay stable across
    repeated qd.init(vulkan)/qd.reset() cycles.  An NVIDIA driver bug
    corrupts it after ~11 vkDestroyInstance/vkCreateInstance cycles;
    the fix is to reuse the VkInstance."""
    N = 16
    NUM_CYCLES = 20
    reference = None

    for cycle in range(NUM_CYCLES):
        qd.init(arch=qd.vulkan)

        ids = qd.ndarray(dtype=qd.i32, shape=(N,))

        @qd.kernel
        def read_ids(out: qd.types.ndarray(dtype=qd.i32, ndim=1)):
            qd.loop_config(block_dim=N)
            for i in range(N):
                out[i] = subgroup.invocation_id()

        read_ids(ids)
        result = ids.to_numpy().tolist()

        if reference is None:
            reference = result
        else:
            assert result == reference, f"cycle {cycle}: subgroup IDs changed — " f"got {result}, expected {reference}"

        qd.reset()
