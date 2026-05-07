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


# TODO: replace this with a stronger test case. The test relies on a grid-scope memory fence
# ordering the `a[i] = 1` write before the per-block atomic-add counter, so that the "last
# block" branch can read all the `a[i]` values back. CUDA / AMDGPU honor this strictly via
# their `_mem_fence` intrinsics; Vulkan honors it via `OpMemoryBarrier(ScopeDevice, ...)`;
# Metal honors it via MSL `atomic_thread_fence(memory_scope_device)` on Apple Silicon /
# macOS 10.13+ (see the support-table caveat in `block.md`). On very old Apple Intel GPUs
# this test may need investigation; for now we run on the full GPU set.
@test_utils.test(arch=qd.gpu)
def test_grid_mem_fence():
    N = 1000
    BLOCK_SIZE = 1
    a = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        block_counter = 0
        qd.loop_config(block_dim=BLOCK_SIZE)
        for i in range(N):
            a[i] = 1
            qd.simt.grid.mem_fence()

            # Execute a prefix sum after all blocks finish
            actual_order_of_block = qd.atomic_add(block_counter, 1)
            if actual_order_of_block == N - 1:
                for j in range(1, N):
                    a[j] += a[j - 1]

    foo()

    for i in range(N):
        assert a[i] == i + 1


# Smoke test for `block.mem_fence()`. We can't easily provoke a memory-ordering bug
# deterministically, so this just ensures the call compiles and the kernel runs end-to-end on
# every supported GPU backend (CUDA / AMDGPU / Vulkan / Metal).
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


# Verify that `block.mem_fence()` can be called from divergent control flow without deadlocking.
# This is the property that distinguishes a memory fence from a thread-converging barrier and
# was the user-facing motivation for the rename `mem_sync -> mem_fence`. Before the CUDA
# dispatch was switched from `block_barrier` to `block_mem_fence` (i.e. NVPTX `__syncthreads()`
# vs. `__threadfence_block()`), this test would hang on CUDA because thread 0 would wait at
# the barrier forever for the other 31 lanes that early-return without reaching the call site.
# AMDGPU lowers to a workgroup-scope `fence`, Vulkan / Metal lower to `OpMemoryBarrier
# (ScopeWorkgroup, ...)`; none of these require thread convergence, so the divergent pattern is
# valid on every backend.
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


# Producer-consumer memory-ordering test for `block.mem_fence()`.
#
# Thread 0 (producer) publishes `data[0] = 100 + it` and then atomically sets `flag[0] = 1`,
# with `block.mem_fence()` between the two stores. Thread `BLOCK - 1` (consumer) atomically
# spin-loads `flag`, then fences and reads `data`. The kernel runs `N_ITERS` rounds and we
# assert the consumer always sees the latest published `data`.
#
# Honest caveat about what this test actually exercises:
#
#   On every supported backend, atomic operations on shared memory are at least relaxed-
#   ordering and on CUDA they are acquire-release. So the atomic flag itself provides
#   *some* of the ordering that `block.mem_fence()` is independently meant to provide. As a
#   result, this test does NOT cleanly isolate `block.mem_fence()` -- it would also pass if
#   the fence were a pure no-op, as long as the atomic flag's implicit ordering is enough.
#
# What it does still catch:
#   1. The regression we actually fixed in this PR -- `block.mem_fence()` lowering to a
#      thread-converging barrier rather than a pure fence -- because the producer (thread 0)
#      and consumer (thread BLOCK-1) hit the fence in different control-flow paths; a
#      barrier-style lowering deadlocks the kernel.
#   2. End-to-end compilation, kernel launch, and shared-memory + atomic-on-shared-memory
#      interactions for the producer/consumer pattern on every backend.
#   3. `block.SharedArray` + `block.mem_fence` + `qd.atomic_add` on shared memory used
#      together in a non-trivial way.
#
# What it does NOT catch:
#   * A `block.mem_fence()` that becomes a complete no-op AND the producer's two stores get
#     compiler-reordered to (`flag = 1; data = 100+it`). On CUDA / AMDGPU / SPIR-V the
#     opaque `mem_fence` call site is currently treated as a memory clobber by LLVM, so the
#     compiler does not reorder around it even when the runtime effect is empty. A test
#     that strictly isolates the fence's runtime effect would require a `volatile`-flavored
#     shared-array primitive that Quadrants does not currently expose.
#
# Layout choice: `BLOCK = 128` puts the producer (thread 0) and consumer (thread BLOCK-1)
# in different subgroups on every supported backend (CUDA warp = 32; AMDGPU wave = 32 or 64;
# Vulkan / Metal subgroup is vendor-defined but <= 128 on every shipping target). Threads
# in different subgroups have independent forward progress on every modern GPU, so the
# consumer's spin does not deadlock waiting on a producer it shares lockstep execution
# with. We deliberately do NOT use `block.sync()` between the producer's two stores --
# that would test sync ordering instead of fence ordering.
@test_utils.test(arch=qd.gpu)
def test_block_mem_fence_producer_consumer():
    N_ITERS = 64
    BLOCK = 128
    out = qd.field(dtype=qd.i32, shape=N_ITERS)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=BLOCK)
        for tid in range(BLOCK):
            flag = qd.simt.block.SharedArray((1,), qd.i32)
            data = qd.simt.block.SharedArray((1,), qd.i32)

            for it in range(N_ITERS):
                if tid == 0:
                    data[0] = 0
                    flag[0] = 0
                qd.simt.block.sync()

                if tid == 0:
                    data[0] = 100 + it
                    qd.simt.block.mem_fence()
                    qd.atomic_add(flag[0], 1)
                elif tid == BLOCK - 1:
                    while qd.atomic_add(flag[0], 0) == 0:
                        pass
                    qd.simt.block.mem_fence()
                    out[it] = data[0]

                qd.simt.block.sync()

    foo()

    for it in range(N_ITERS):
        assert out[it] == 100 + it, f"iter {it}: got {out[it]}, expected {100 + it}"


# Deprecation aliases: the old names still work, and emit DeprecationWarning on first use.
# pytest.warns enables `simplefilter("always")` for its scope, bypassing the project-wide
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


@test_utils.test(arch=qd.gpu)
def test_grid_memfence_deprecated_alias():
    a = qd.field(dtype=qd.i32, shape=1)

    with pytest.warns(DeprecationWarning, match=r"qd\.simt\.grid\.memfence"):

        @qd.kernel
        def foo():
            a[0] = 11
            qd.simt.grid.memfence()

        foo()

    assert a[0] == 11


# Portable test for `block.global_thread_idx()`. Runs on every supported GPU backend; in particular,
# verifies the SPIR-V dispatch path that was previously unreachable due to a Python-side dispatch bug.
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


# Portable test for `block.thread_idx()`. Sets `block_dim == grid_dim_total` (single-block
# launch) so the in-block index equals the global index, then verifies on every GPU backend.
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


# Multi-block coverage for `block.thread_idx()`: with block_dim=8 and loop total 32, the kernel
# runs across 4 blocks and the in-block index must reset to 0 at each block boundary. Without
# this case, a regression that aliased `block.thread_idx()` to `block.global_thread_idx()` (or
# vice versa) would slip past the single-block portable tests. CUDA / AMDGPU lower this to the
# `tid.x` SREG; Vulkan / Metal lower it to `gl_LocalInvocationID.x` (which is what required the
# `OpEntryPoint` interface fix earlier in this PR).
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


# Multi-block coverage for `block.global_thread_idx()`: same shape as the test above, but the
# expected values span the full grid (0..N-1) rather than wrapping per block. Together with
# `test_block_thread_idx_multi_block` this distinguishes the two ops on every backend — a
# `global_thread_idx == thread_idx` aliasing regression fails one of the two.
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
