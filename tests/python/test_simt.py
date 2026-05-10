import platform

import numpy as np
import pytest
from pytest import approx

import quadrants as qd
from quadrants.lang.simt import subgroup

from tests import test_utils


def _skip_if_f64_unsupported(dtype):
    arch = qd.lang.impl.current_cfg().arch
    if dtype in (qd.i64, qd.u64) and arch == qd.metal:
        pytest.skip("64-bit integer types not supported on Metal")
    if dtype in (qd.i64, qd.u64) and arch == qd.vulkan and platform.system() == "Darwin":
        pytest.skip("MoltenVK does not support 64-bit integer types")
    if dtype != qd.f64:
        return
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


@test_utils.test(arch=qd.gpu)
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


@test_utils.test(arch=qd.gpu)
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


@test_utils.test(arch=qd.gpu)
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


@test_utils.test(arch=qd.gpu)
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


# Smoke test for `block.mem_fence()`. We can't easily provoke a memory-ordering bug deterministically, so this just
# ensures the call compiles and the kernel runs end-to-end on every supported GPU backend (CUDA / AMDGPU / Vulkan /
# Metal).
@test_utils.test(arch=qd.gpu)
def test_block_mem_fence_smoke():
    N = 32
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = i
            qd.simt.block.mem_fence()
            qd.simt.block.sync()

    foo()

    for i in range(N):
        assert a[i] == i


# Verify that `block.mem_fence()` can be called from divergent control flow without deadlocking. This is the property
# that distinguishes a memory fence from a thread-converging barrier and was the user-facing motivation for the rename
# `mem_sync -> mem_fence`. Before the CUDA dispatch was switched from `block_barrier` to `block_mem_fence` (i.e. NVPTX
# `__syncthreads()` vs. `__threadfence_block()`), this test would hang on CUDA because thread 0 would wait at the
# barrier forever for the other 31 lanes that early-return without reaching the call site. AMDGPU lowers to a
# workgroup-scope `fence`, Vulkan / Metal lower to `OpMemoryBarrier(ScopeWorkgroup, ...)`; none of these require thread
# convergence, so the divergent pattern is valid on every backend.
@test_utils.test(arch=qd.gpu)
def test_block_mem_fence_divergent_control_flow():
    N = 32
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            tid = qd.simt.block.thread_idx()
            if tid == 0:
                qd.simt.block.mem_fence()
            a[i] = tid

    foo()

    for i in range(N):
        assert a[i] == i


# NOTE: a producer-consumer memory-ordering test for `block.mem_fence()` (i.e. thread 0 publishes data + flag with a
# fence between, another thread spin-waits on flag then reads data) was attempted in this PR and removed. Every variant
# deadlocks on CUDA before pytest's process-level timeout fires:
#
# 1. Plain shared-memory load `while flag[0] == 0: pass` -- LLVM hoists the load out of the loop because there is no
#    `volatile` qualifier and no memory-clobbering call inside the loop body. The consumer never re-reads the flag.
#
# 2. Plain shared-memory load with `block.mem_fence()` inside the loop body -- LLVM's `nvvm_membar_cta` intrinsic is
#    `IntrInaccessibleMemOnly`, which clobbers the "inaccessible" memory category but does not invalidate the
#    optimizer's cached view of `addrspace(3)` (shared memory) loads it has already seen. The flag load is still
#    hoisted.
#
# 3. Atomic-add load `while qd.atomic_add(flag[0], 0) == 0: pass` -- adding zero appears to be elided somewhere in the
#    Quadrants -> LLVM lowering pipeline; the loop also fails to terminate.
#
# 4. Atomic-or load `while qd.atomic_or(flag[0], 0) == 0: pass` -- same outcome as (3).
#
# Even if one of these did terminate, atomic-flagged variants would weaken the test -- atomic ops on shared memory are
# relaxed-ordering at minimum and acq-rel on CUDA, so the atomic itself would provide much of the ordering that
# `block.mem_fence()` is meant to provide independently.
#
# The fundamental gap is that Quadrants' Python API does not currently expose a `volatile`-flavored shared-array
# primitive, which is the standard tool for writing producer-consumer ordering tests in CUDA / Vulkan / Metal. Adding
# such a primitive (e.g. `block.SharedArray(..., volatile=True)`) is a Quadrants frontend feature outside this PR.
#
# What we do test, in lieu of a full producer-consumer ordering test:
#   - `test_block_mem_fence_smoke`: end-to-end compile + run on every backend.
#   - `test_block_mem_fence_divergent_control_flow`: the fence is a fence, not a thread-converging barrier. This
#     catches the regression we actually fixed in this PR.
#
# The convergence test exercises every backend's mem_fence through divergent control flow, which is the hard
# correctness property and the practical motivation for renaming `mem_sync -> mem_fence`. Pure memory-ordering
# correctness against compiler reordering of adjacent shared-memory stores is left as a future enhancement.


# Deprecation aliases: the old names still work, and emit DeprecationWarning on first use. pytest.warns enables
# `simplefilter("always")` for its scope, bypassing the project-wide
# `warnings.filterwarnings("once", ..., module="quadrants")` set in `quadrants/lang/misc.py`.
@test_utils.test(arch=qd.gpu)
def test_block_mem_sync_deprecated_alias():
    a = qd.field(dtype=qd.i32, shape=1)

    with pytest.warns(DeprecationWarning, match=r"qd\.simt\.block\.mem_sync"):

        @qd.kernel
        def foo():
            a[0] = 7
            qd.simt.block.mem_sync()

        foo()

    assert a[0] == 7


# Portable test for `block.global_thread_idx()`. Runs on every supported GPU backend; in particular, verifies the
# SPIR-V dispatch path that was previously unreachable due to a Python-side dispatch bug.
@test_utils.test(arch=qd.gpu)
def test_block_global_thread_idx_portable():
    N = 64
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = qd.simt.block.global_thread_idx()

    foo()

    for i in range(N):
        assert a[i] == i


# Portable test for `block.thread_idx()`. Sets `block_dim == grid_dim_total` (single-block launch) so the in-block
# index equals the global index, then verifies on every GPU backend.
@test_utils.test(arch=qd.gpu)
def test_block_thread_idx_portable():
    N = 64
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = qd.simt.block.thread_idx()

    foo()

    for i in range(N):
        assert a[i] == i


# Multi-block coverage for `block.thread_idx()`: with block_dim=8 and loop total 32, the kernel runs across 4 blocks
# and the in-block index must reset to 0 at each block boundary. Without this case, a regression that aliased
# `block.thread_idx()` to `block.global_thread_idx()` (or vice versa) would slip past the single-block portable tests.
# CUDA / AMDGPU lower this to the `tid.x` SREG; Vulkan / Metal lower it to `gl_LocalInvocationID.x` (which is what
# required the `OpEntryPoint` interface fix earlier in this PR).
@test_utils.test(arch=qd.gpu)
def test_block_thread_idx_multi_block():
    N = 32
    BLOCK = 8
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=BLOCK)
        for i in range(N):
            a[i] = qd.simt.block.thread_idx()

    foo()

    for i in range(N):
        assert a[i] == i % BLOCK


# Multi-block coverage for `block.global_thread_idx()`: same shape as the test above, but the expected values span the
# full grid (0..N-1) rather than wrapping per block. Together with `test_block_thread_idx_multi_block` this
# distinguishes the two ops on every backend — a `global_thread_idx == thread_idx` aliasing regression fails one of the
# two.
@test_utils.test(arch=qd.gpu)
def test_block_global_thread_idx_multi_block():
    N = 32
    BLOCK = 8
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=BLOCK)
        for i in range(N):
            a[i] = qd.simt.block.global_thread_idx()

    foo()

    for i in range(N):
        assert a[i] == i


# The old SPIR-V-only no-arg subgroup reductions (`subgroup.reduce_add` / `reduce_mul` / `reduce_min`
# / `reduce_max` / `reduce_and` / `reduce_or` / `reduce_xor`) and their Vulkan-specific tests have
# been removed.  See `test_subgroup_reduce_add` / `test_subgroup_reduce_all_add` below for the
# portable sized-reduction tests, and add equivalent sized portable replacements for the other
# reductions on top of `shuffle_down` / `shuffle` if needed.


def _init_field(field, n, dtype):
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(n):
        field[i] = (i + 1) if dtype in int_dtypes else 1.0000000000001 * (i + 1)


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


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_down(dtype):
    """shuffle_down: each lane reads from lane_id + offset."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.shuffle_down(src[i], qd.u32(1))

    _init_field(src, N, dtype)

    foo()

    # Lane 0 reads from lane 1, lane 1 from lane 2, lane 2 from lane 3
    # (within the guaranteed min subgroup of 4 lanes, lane 3's result is undefined)
    assert dst[0] == src[1]
    assert dst[1] == src[2]
    assert dst[2] == src[3]


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_up(dtype):
    """shuffle_up: each lane reads from lane_id - offset."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.shuffle_up(src[i], qd.u32(1))

    _init_field(src, N, dtype)

    foo()

    # Lane 1 reads from lane 0, lane 2 from lane 1, lane 3 from lane 2 (within the guaranteed min subgroup of 4 lanes,
    # lane 0's result is undefined).
    assert dst[1] == src[0]
    assert dst[2] == src[1]
    assert dst[3] == src[2]


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_xor(dtype):
    """shuffle_xor: each lane reads from lane_id ^ mask. Wrapper version of the manual XOR pattern tested in
    test_subgroup_shuffle_xor_pattern."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.shuffle_xor(src[i], qd.u32(1))

    _init_field(src, N, dtype)

    foo()

    # Lanes 0-3 are in the same subgroup: 0<->1 swap, 2<->3 swap.
    assert dst[0] == src[1]
    assert dst[1] == src[0]
    assert dst[2] == src[3]
    assert dst[3] == src[2]


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_broadcast_first(dtype):
    """broadcast_first: every lane gets lane 0's value. Portable @qd.func wrapper over broadcast(value, 0)."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    a = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = subgroup.broadcast_first(a[i])

    _init_field(a, N, dtype)
    expected = a[0]

    foo()

    # Lanes 0-3 are guaranteed to share a subgroup (min size is 4).
    for i in range(4):
        assert a[i] == expected


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_down_reduction(dtype):
    """Tree reduction via shuffle_down, summing 4 values."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            val = src[i]
            val = val + subgroup.shuffle_down(val, qd.u32(2))
            val = val + subgroup.shuffle_down(val, qd.u32(1))
            dst[i] = val

    _init_field(src, N, dtype)

    foo()

    # Lane 0 should have sum of lanes 0-3 (within the min subgroup of 4)
    expected = sum(src[i] for i in range(4))
    assert abs(dst[0] - expected) < 1e-5


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_add(dtype, log2_size):
    """Portable shuffle_down tree reduction: lane 0 of each 2**log2_size group has the sum."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.reduce_add(src[i], log2_size)

    _init_field(src, N, dtype)
    foo()

    group_size = 1 << log2_size
    expected = sum(src[i] for i in range(group_size))
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    if dtype in int_dtypes:
        assert dst[0] == expected
    else:
        assert abs(dst[0] - expected) < 1e-4 * abs(expected)


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_all_add(dtype, log2_size):
    """Portable butterfly XOR reduction: every lane in each 2**log2_size group has the sum."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.reduce_all_add(src[i], log2_size)

    _init_field(src, N, dtype)
    foo()

    group_size = 1 << log2_size
    expected = sum(src[i] for i in range(group_size))
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(group_size):
        if dtype in int_dtypes:
            assert dst[i] == expected, f"lane {i}: got {dst[i]}, expected {expected}"
        else:
            assert abs(dst[i] - expected) < 1e-4 * abs(expected), f"lane {i}: got {dst[i]}, expected {expected}"


# Portable Hillis-Steele inclusive scans share the same kernel + Python verification pattern; the only thing that
# varies is the operator and which dtypes are legal for it.  `_check_inclusive_scan` factors out the kernel launch,
# dtype skip, and per-lane check.

_INT_DTYPES = (qd.i32, qd.i64, qd.u64)


def _check_inclusive_scan(scan_func, py_op, dtype, log2_size, src_init):
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = scan_func(src[i], log2_size)

    src_init(src, N, dtype)
    foo()

    group_size = 1 << log2_size
    # Verify every group across the full 64-lane launch.  group_size <= 32 by the size contract, so each group fits
    # inside one CUDA / Metal / RDNA subgroup.  Lanes 0-31 and 32-63 form two independent subgroups that run the scan
    # side by side; checking both, plus every group within each subgroup when log2_size < 5, exercises (a) the
    # `lane_in_group >= offset` mask that isolates partial-subgroup groups from each other and (b) the absence of
    # cross-subgroup leakage in the underlying shuffle_up.
    for g in range(N // group_size):
        group_base = g * group_size
        running = src[group_base]
        for k in range(group_size):
            if k > 0:
                running = py_op(running, src[group_base + k])
            global_lane = group_base + k
            got = dst[global_lane]
            if dtype in _INT_DTYPES:
                assert got == running, f"group {g} lane {k} (global {global_lane}): got {got}, expected {running}"
            else:
                assert abs(got - running) < 1e-4 * max(
                    abs(running), 1.0
                ), f"group {g} lane {k} (global {global_lane}): got {got}, expected {running}"


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_add(dtype, log2_size):
    """Portable inclusive prefix sum: lane k of each 2**log2_size group has sum(src[group_base..group_base+k+1])."""
    _check_inclusive_scan(subgroup.inclusive_add, lambda a, b: a + b, dtype, log2_size, _init_field)


def _init_small_int_or_float(field, n, dtype):
    """Bound the 32-way product to 2**8 == 256 (i32-safe, f32-exact): most lanes hold 1, every fourth lane holds 2.
    With log2_size=5 (32 lanes / group), 8 lanes contribute a factor of 2 → product ≤ 256.  Smaller log2_size only sees
    a subset."""
    for i in range(n):
        field[i] = 2 if (i % 4 == 0) else 1


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_mul(dtype, log2_size):
    """Inclusive prefix product.  Inputs are 1 / 2 mixed so the 32-way product is at most 2**8 == 256 (well within i32
    and f32-exact)."""
    _check_inclusive_scan(subgroup.inclusive_mul, lambda a, b: a * b, dtype, log2_size, _init_small_int_or_float)


def _init_varied_int_or_float(field, n, dtype):
    """Mix increasing and decreasing values so prefix-min / prefix-max have non-trivial transitions across the
    group."""
    for i in range(n):
        # 11, 7, 13, 3, 17, 5, 19, ...  -- varied, non-monotonic, non-negative
        field[i] = ((i * 7 + 11) % 23) + 1


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_min(dtype, log2_size):
    """Inclusive prefix min."""
    _check_inclusive_scan(subgroup.inclusive_min, lambda a, b: min(a, b), dtype, log2_size, _init_varied_int_or_float)


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_max(dtype, log2_size):
    """Inclusive prefix max."""
    _check_inclusive_scan(subgroup.inclusive_max, lambda a, b: max(a, b), dtype, log2_size, _init_varied_int_or_float)


def _init_bitwise_int(field, n, dtype):
    """Initialise with bit-varied integer values so prefix &/|/^ have meaningful transitions (bits flip on and off
    across lanes)."""
    for i in range(n):
        field[i] = (i * 5 + 0x37) & 0xFF


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_and(dtype, log2_size):
    """Inclusive prefix bitwise-AND.  Integer dtypes only."""
    _skip_if_f64_unsupported(dtype)  # also handles 64-bit-int Metal/MoltenVK skips
    _check_inclusive_scan(subgroup.inclusive_and, lambda a, b: a & b, dtype, log2_size, _init_bitwise_int)


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_or(dtype, log2_size):
    """Inclusive prefix bitwise-OR.  Integer dtypes only."""
    _skip_if_f64_unsupported(dtype)
    _check_inclusive_scan(subgroup.inclusive_or, lambda a, b: a | b, dtype, log2_size, _init_bitwise_int)


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_xor(dtype, log2_size):
    """Inclusive prefix bitwise-XOR.  Integer dtypes only."""
    _skip_if_f64_unsupported(dtype)
    _check_inclusive_scan(subgroup.inclusive_xor, lambda a, b: a ^ b, dtype, log2_size, _init_bitwise_int)


# --- Exclusive scans ------------------------------------------------------------------
#
# Same kernel + verification shape as the inclusive tests above; the only differences are (1) lane 0 of each group is
# expected to hold the operator's identity, and (2) for min/max the wrapper takes an explicit `identity` arg that we
# pass through here.


def _check_exclusive_scan(scan_func, py_op, py_identity, dtype, log2_size, src_init, *, takes_identity_arg=False):
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    if takes_identity_arg:
        identity_value = py_identity

        @qd.kernel
        def foo():
            qd.loop_config(block_dim=N)
            for i in range(N):
                dst[i] = scan_func(src[i], log2_size, identity_value)

    else:

        @qd.kernel
        def foo():
            qd.loop_config(block_dim=N)
            for i in range(N):
                dst[i] = scan_func(src[i], log2_size)

    src_init(src, N, dtype)
    foo()

    group_size = 1 << log2_size
    # Verify every group across the full 64-lane launch (see `_check_inclusive_scan` for the rationale).
    # exclusive[group_base] == identity; exclusive[group_base + k] for k > 0
    # == op-reduce(src[group_base..group_base + k]).
    for g in range(N // group_size):
        group_base = g * group_size
        for k in range(group_size):
            if k == 0:
                expected = py_identity
            else:
                expected = src[group_base]
                for j in range(1, k):
                    expected = py_op(expected, src[group_base + j])
            global_lane = group_base + k
            got = dst[global_lane]
            if dtype in _INT_DTYPES:
                assert got == expected, f"group {g} lane {k} (global {global_lane}): got {got}, expected {expected}"
            else:
                assert abs(got - expected) < 1e-4 * max(
                    abs(expected), 1.0
                ), f"group {g} lane {k} (global {global_lane}): got {got}, expected {expected}"


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_add(dtype, log2_size):
    """Exclusive prefix sum.  Lane 0 of each group is 0."""
    _check_exclusive_scan(subgroup.exclusive_add, lambda a, b: a + b, 0, dtype, log2_size, _init_field)


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_mul(dtype, log2_size):
    """Exclusive prefix product.  Lane 0 of each group is 1."""
    _check_exclusive_scan(subgroup.exclusive_mul, lambda a, b: a * b, 1, dtype, log2_size, _init_small_int_or_float)


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_min(dtype, log2_size):
    """Exclusive prefix min.  Lane 0 of each group is the explicit `identity` we pass.  Use a sentinel larger than any
    element produced by `_init_varied_int_or_float` (max is 23)."""
    identity = 1_000_000 if dtype == qd.i32 else 1e30
    _check_exclusive_scan(
        subgroup.exclusive_min,
        lambda a, b: min(a, b),
        identity,
        dtype,
        log2_size,
        _init_varied_int_or_float,
        takes_identity_arg=True,
    )


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_max(dtype, log2_size):
    """Exclusive prefix max.  Lane 0 of each group is the explicit `identity` we pass.  Use a sentinel smaller than
    any element produced by `_init_varied_int_or_float` (min is 1)."""
    identity = -1_000_000 if dtype == qd.i32 else -1e30
    _check_exclusive_scan(
        subgroup.exclusive_max,
        lambda a, b: max(a, b),
        identity,
        dtype,
        log2_size,
        _init_varied_int_or_float,
        takes_identity_arg=True,
    )


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_and(dtype, log2_size):
    """Exclusive prefix bitwise-AND.  Lane 0 of each group is all-bits-set."""
    _skip_if_f64_unsupported(dtype)
    if dtype == qd.i32:
        identity = -1
    elif dtype == qd.i64:
        identity = -1
    else:  # u64
        identity = (1 << 64) - 1
    _check_exclusive_scan(subgroup.exclusive_and, lambda a, b: a & b, identity, dtype, log2_size, _init_bitwise_int)


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_or(dtype, log2_size):
    """Exclusive prefix bitwise-OR.  Lane 0 of each group is 0."""
    _skip_if_f64_unsupported(dtype)
    _check_exclusive_scan(subgroup.exclusive_or, lambda a, b: a | b, 0, dtype, log2_size, _init_bitwise_int)


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_xor(dtype, log2_size):
    """Exclusive prefix bitwise-XOR.  Lane 0 of each group is 0."""
    _skip_if_f64_unsupported(dtype)
    _check_exclusive_scan(subgroup.exclusive_xor, lambda a, b: a ^ b, 0, dtype, log2_size, _init_bitwise_int)


# Voting / predicate ops.  All three are group-scoped over 2**log2_size lanes; the scenario tables below exercise (a)
# every-lane-true / every-lane-false, (b) a single odd lane in one group with the rest all-true / all-false (group
# isolation), and (c) a sparse pattern that lands several groups in the all-true case and several in the mixed case so
# the per-group reduction is the only thing distinguishing them.  We verify every group across the full 64-lane launch
# (so log2_size in {1..4} covers the multi-group case and log2_size==5 covers the full-warp / CUDA fast-path case, and
# the launch spans two CUDA / Metal / RDNA subgroups so cross-subgroup leakage would also be caught).


@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_all_true(log2_size):
    """``all_true(predicate, log2_size)`` is ``i32(all(predicate != 0))`` over each ``2**log2_size`` group, broadcast
    to every lane in the group."""
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.all_true(src[i], log2_size)

    group_size = 1 << log2_size

    def run_and_check(label, src_values):
        for i in range(N):
            src[i] = src_values[i]
        foo()
        for g in range(N // group_size):
            base = g * group_size
            expected = int(all(src_values[base + k] != 0 for k in range(group_size)))
            for k in range(group_size):
                got = dst[base + k]
                assert (
                    got == expected
                ), f"{label} group {g} lane {k} (global {base + k}): got {got}, expected {expected}"

    run_and_check("all-1", [1] * N)
    run_and_check("all-0", [0] * N)
    mixed = [1] * N
    mixed[3] = 0
    run_and_check("zero-at-3", mixed)
    run_and_check("sparse-zeros", [(0 if (i % 7 == 0) else 1) for i in range(N)])
    # Non-binary truthy values: locks the `predicate != 0` cast.  Values include 0, positive ints, and negatives --
    # anything non-zero must count as true.
    run_and_check("nonbinary-mixed", [((i * 17) % 13) - 6 for i in range(N)])


@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_any_true(log2_size):
    """``any_true(predicate, log2_size)`` is ``i32(any(predicate != 0))`` over each ``2**log2_size`` group, broadcast
    to every lane in the group."""
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.any_true(src[i], log2_size)

    group_size = 1 << log2_size

    def run_and_check(label, src_values):
        for i in range(N):
            src[i] = src_values[i]
        foo()
        for g in range(N // group_size):
            base = g * group_size
            expected = int(any(src_values[base + k] != 0 for k in range(group_size)))
            for k in range(group_size):
                got = dst[base + k]
                assert (
                    got == expected
                ), f"{label} group {g} lane {k} (global {base + k}): got {got}, expected {expected}"

    run_and_check("all-1", [1] * N)
    run_and_check("all-0", [0] * N)
    one_at = [0] * N
    one_at[3] = 1
    run_and_check("one-at-3", one_at)
    run_and_check("sparse-ones", [(1 if (i % 7 == 0) else 0) for i in range(N)])
    # Non-binary truthy values: locks the `predicate != 0` cast.  Same pattern as `test_subgroup_all_true`; mixes 0,
    # positive, and negative ints.
    run_and_check("nonbinary-mixed", [((i * 17) % 13) - 6 for i in range(N)])


@pytest.mark.parametrize("dtype", [qd.i32, qd.i64, qd.u64, qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_all_equal(dtype, log2_size):
    """``all_equal(value, log2_size)`` is ``i32(all values equal)`` over each ``2**log2_size`` group, broadcast to
    every lane in the group.  Equality is the backend's native ``==``; we restrict scenarios to exactly-representable
    values (small integers / their f32+f64 castings) so float ``==`` is unambiguous."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.all_equal(src[i], log2_size)

    group_size = 1 << log2_size

    def run_and_check(label, src_values):
        for i in range(N):
            src[i] = src_values[i]
        foo()
        for g in range(N // group_size):
            base = g * group_size
            base_val = src_values[base]
            expected = int(all(src_values[base + k] == base_val for k in range(group_size)))
            for k in range(group_size):
                got = dst[base + k]
                assert (
                    got == expected
                ), f"{label} group {g} lane {k} (global {base + k}): got {got}, expected {expected}"

    run_and_check("all-same", [42] * N)
    run_and_check("all-distinct", list(range(N)))
    run_and_check(
        "same-per-group",
        [g for g in range(N // group_size) for _ in range(group_size)],
    )
    run_and_check(
        "one-outlier-per-group",
        [(99 if (i % group_size) == (group_size - 1) else 7) for i in range(N)],
    )


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@pytest.mark.parametrize("log2_size", [1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_all_equal_float_contract(dtype, log2_size):
    """``all_equal`` on floats uses the backend's native ``==``: ``NaN != NaN`` and ``+0.0 == -0.0``, matching SPIR-V
    ``OpGroupNonUniformAllEqual``.  Locks both contracts so a future refactor (e.g. swapping in ``__match_all_sync`` on
    CUDA) can't silently regress to bit-equality."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.all_equal(src[i], log2_size)

    group_size = 1 << log2_size

    def run_and_check(label, src_values, expected_per_group):
        for i in range(N):
            src[i] = src_values[i]
        foo()
        for g in range(N // group_size):
            base = g * group_size
            expected = expected_per_group(g)
            for k in range(group_size):
                got = dst[base + k]
                assert (
                    got == expected
                ), f"{label} group {g} lane {k} (global {base + k}): got {got}, expected {expected}"

    nan = float("nan")

    # +0.0 == -0.0 on the backend, so groups mixing +/- zero are all-equal.  Lane 0 of each group is +0.0 (group_base
    # is always even for log2_size >= 1), every other lane in the group compares its value with +0.0 and gets True.
    run_and_check(
        "plus_minus_zero",
        [(-0.0 if (i & 1) else 0.0) for i in range(N)],
        lambda g: 1,
    )
    # NaN != NaN: a group containing any NaN reports 0.  Place NaN at the start of every group so every group fails.
    run_and_check(
        "nan_at_group_start",
        [(nan if (i % group_size) == 0 else 1.0) for i in range(N)],
        lambda g: 0,
    )
    # NaN at one lane: only the affected group reports 0, the rest are all-1.0 -> 1.
    one_nan = [1.0] * N
    one_nan[17] = nan
    affected_group = 17 // group_size
    run_and_check("one_nan_at_17", one_nan, lambda g: 0 if g == affected_group else 1)
    # All NaN: every lane's `value == base` is False (NaN != NaN), so every group is 0.
    run_and_check("all_nan", [nan] * N, lambda g: 0)


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
def test_subgroup_sync():
    """Smoke test that ``subgroup.sync()`` traces, codegens, and runs on every GPU backend.

    Verifies the trivial "sync inside a uniform-CF kernel doesn't break the emitted code" contract on CUDA
    (``__syncwarp(0xFFFFFFFF)``), AMDGPU (``llvm.amdgcn.wave.barrier``), and SPIR-V (``OpControlBarrier(Subgroup,
    Subgroup, 0)``).  We do not attempt to test reconvergence semantics here — that would require deliberately
    divergent control flow plus a memory-visible side-channel and is too flaky to be a unit test.
    """
    N = 64
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            subgroup.sync()
            a[i] = subgroup.invocation_id()

    foo()
    for i in range(N):
        assert a[i] >= 0


@test_utils.test(arch=qd.gpu)
def test_subgroup_mem_fence():
    """Smoke test that ``subgroup.mem_fence()`` traces, codegens, and runs on every GPU backend: CUDA
    (``__threadfence_block()``), AMDGPU (LLVM workgroup-scope ``fence``), and SPIR-V (``OpMemoryBarrier(Subgroup,
    AcquireRelease | UniformMemory | WorkgroupMemory)``).

    Like ``test_subgroup_sync``, we verify only that the kernel compiles and runs.  Testing actual memory-ordering
    semantics requires constructing a producer/consumer race that only the fence makes legal, which is hard to write
    portably and easy to make flaky.
    """
    N = 64
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = subgroup.invocation_id()
            subgroup.mem_fence()

    foo()
    for i in range(N):
        assert a[i] >= 0


@test_utils.test(arch=qd.gpu)
def test_subgroup_group_size():
    """``subgroup.group_size()`` returns the active subgroup size.

    Lowers to a constant ``32`` on CUDA, ``llvm.amdgcn.wavefrontsize`` on AMDGPU (constant-folded by the AMDGPU backend
    to 32 or 64 depending on wavefront mode), and ``OpSubgroupSize`` on SPIR-V.  We verify (a) every lane sees the same
    value and (b) that value is one of the sizes the spec actually allows on real hardware ({32, 64}).
    """
    N = 128
    a = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            a[i] = subgroup.group_size()

    foo()
    seen = a[0]
    assert seen in (32, 64), f"unexpected group_size {seen}"
    for i in range(N):
        assert a[i] == seen, f"group_size disagrees: a[0]={seen}, a[{i}]={a[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_elect():
    """``subgroup.elect()`` returns ``1`` on lane 0 of every subgroup, ``0`` elsewhere.

    Implemented as a ``@qd.func`` wrapper over ``invocation_id() == 0``, so it works on every backend that lowers
    ``invocation_id``.  We verify, in a single kernel, that:

    * ``elect()`` returns 0 or 1.
    * Every elected lane has ``invocation_id() == 0``, and every non-elected lane has ``invocation_id() != 0`` — i.e.
      lane 0 is exactly the elected one.
    * The total elected count equals ``N / group_size()`` — one per subgroup.
    """
    N = 256
    elected = qd.field(dtype=qd.i32, shape=N)
    lane_id = qd.field(dtype=qd.i32, shape=N)
    sg_size = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            elected[i] = subgroup.elect()
            lane_id[i] = subgroup.invocation_id()
            sg_size[i] = subgroup.group_size()

    foo()

    sg = sg_size[0]
    assert sg in (32, 64), f"unexpected group_size {sg}"

    total_elected = 0
    for i in range(N):
        assert elected[i] in (0, 1), f"elect() returned non-bool {elected[i]} at i={i}"
        if elected[i] == 1:
            assert lane_id[i] == 0, f"elected lane has invocation_id={lane_id[i]}, expected 0"
            total_elected += 1
        else:
            assert lane_id[i] != 0, f"non-elected lane has invocation_id=0 at i={i}"

    assert total_elected == N // sg, f"expected {N // sg} elected lanes (N={N} / sg_size={sg}), got {total_elected}"


def _drain_deprecation_warnings(records):
    return [r for r in records if issubclass(r.category, DeprecationWarning)]


def test_subgroup_barrier_deprecation_warn_once(monkeypatch):
    """``subgroup.barrier()`` is a deprecated alias for ``subgroup.sync()``.  It must emit a single
    ``DeprecationWarning`` on first use (regardless of how many times it is called) and forward to ``sync()``.
    Pure-Python unit test: ``sync`` is monkey-patched to a no-op so the test does not require a Quadrants kernel
    context."""
    import warnings as _w

    from quadrants.lang.simt import subgroup as sg

    sg._barrier_deprecation_warned = False
    calls = []
    monkeypatch.setattr(sg, "sync", lambda: calls.append("sync"))

    with _w.catch_warnings(record=True) as records:
        _w.simplefilter("always", DeprecationWarning)
        sg.barrier()
        sg.barrier()
        sg.barrier()

    deprecations = _drain_deprecation_warnings(records)
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    msg = str(deprecations[0].message)
    assert "qd.simt.subgroup.barrier()" in msg
    assert "qd.simt.subgroup.sync()" in msg
    assert calls == ["sync", "sync", "sync"], calls


def test_subgroup_memory_barrier_deprecation_warn_once(monkeypatch):
    """``subgroup.memory_barrier()`` is a deprecated alias for ``subgroup.mem_fence()``.  Mirror of
    ``test_subgroup_barrier_deprecation_warn_once``."""
    import warnings as _w

    from quadrants.lang.simt import subgroup as sg

    sg._memory_barrier_deprecation_warned = False
    calls = []
    monkeypatch.setattr(sg, "mem_fence", lambda: calls.append("mem_fence"))

    with _w.catch_warnings(record=True) as records:
        _w.simplefilter("always", DeprecationWarning)
        sg.memory_barrier()
        sg.memory_barrier()
        sg.memory_barrier()

    deprecations = _drain_deprecation_warnings(records)
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    msg = str(deprecations[0].message)
    assert "qd.simt.subgroup.memory_barrier()" in msg
    assert "qd.simt.subgroup.mem_fence()" in msg
    assert calls == ["mem_fence", "mem_fence", "mem_fence"], calls


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
