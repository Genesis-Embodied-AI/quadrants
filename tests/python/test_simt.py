import math
import platform

import numpy as np
import pytest
from pytest import approx

import quadrants as qd
from quadrants.lang.simt import block, subgroup

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


# Scenario tables for the sized reduce / scan tests below.  We deliberately sample (dtype, log2_size) rather than
# take the full cartesian product, because the two axes are orthogonal in the lowering:
#
# * Different ``log2_size`` values exercise different unroll depths of the same ``shuffle_down`` / ``shuffle_xor`` /
#   ``shuffle_up`` tree (one extra step per increment).  The tree shape is the same regardless of dtype, so once we
#   have verified the tree behaves correctly at one dtype we mostly just need spot checks at the other dtypes.
# * Different dtypes exercise different lowering paths inside ``subgroup.shuffle*`` (e.g. 64-bit values are split into
#   two 32-bit shuffles on AMDGPU, ``f64`` and ``i64`` are skipped on Metal / MoltenVK).  That lowering is
#   independent of ``log2_size`` -- a bug in the i64 path is just as visible at ``log2_size = 1`` as at ``log2_size =
#   5``.
#
# So every scenario table holds ``log2_size = 1`` (the shortest tree, catches one-step-only bugs) and ``log2_size =
# 5`` (the full-wave32 / 32-lane-tile-on-wave64 case, the most common production size) for ``i32``, plus one row
# per non-i32 dtype at ``log2_size = 5``.  We don't include ``log2_size = 6`` in the bulk matrix -- that's covered by
# the dedicated ``test_subgroup_*_log2_size_6`` tests further down, which only run on AMDGPU wave64.  Net effect:
# ~6 cases per test instead of the previous cartesian ~25, ~4x fewer pytest invocations across the whole sized
# reduce / scan suite.

# All five dtypes supported (i32, i64, u64, f32, f64).
_SCENARIOS_FULL_DTYPE = [
    (qd.i32, 1),
    (qd.i32, 5),
    (qd.i64, 5),
    (qd.u64, 5),
    (qd.f32, 5),
    (qd.f64, 5),
]

# i32 + floats (no 64-bit ints) -- used by ``inclusive_mul_tiled`` / ``exclusive_mul_tiled`` (64-bit-int product would
# overflow the tests' 1/2-mixed input) and the min / max scans (only ``inclusive_add_tiled`` / ``exclusive_add_tiled``
# test the four-way i32 / i64 / u64 / f32 / f64 matrix; min / max stick to one int width plus the two floats).
_SCENARIOS_I32_AND_FLOATS = [
    (qd.i32, 1),
    (qd.i32, 5),
    (qd.f32, 5),
    (qd.f64, 5),
]

# Integer dtypes only -- used by the bitwise ops (``inclusive_and_tiled`` / ``or`` / ``xor`` and their exclusive
# variants), which don't accept float dtypes.
_SCENARIOS_INT = [
    (qd.i32, 1),
    (qd.i32, 5),
    (qd.i64, 5),
    (qd.u64, 5),
]

# Same shape as ``_SCENARIOS_FULL_DTYPE`` but with ``log2_size = 0`` instead of ``1`` as the boundary case -- the
# segmented_reduce tests accept ``log2_size = 0`` as a degenerate group-of-1 case (the implicit-head-at-group-base
# fallback collapses to "each lane is its own segment").  Used by segmented_reduce_{add, min, max}.
_SCENARIOS_SEGMENTED = [
    (qd.i32, 0),
    (qd.i32, 5),
    (qd.i64, 5),
    (qd.u64, 5),
    (qd.f32, 5),
    (qd.f64, 5),
]

# Float dtypes only -- used by ``all_equal_float_contract`` which locks the ``NaN != NaN`` / ``+0.0 == -0.0`` shape on
# floats specifically.  ``log2_size = 1`` covers the smallest comparison group; ``5`` covers the full-warp shortcut on
# CUDA.
_SCENARIOS_FLOAT = [
    (qd.f32, 1),
    (qd.f32, 5),
    (qd.f64, 5),
]


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


# The test relies on `grid.mem_fence()` ordering each block's *non-atomic* `a[i] = 1` write before the subsequent
# per-block atomic-add counter. CUDA and AMDGPU honor this strictly via their `mem_fence` intrinsics; native Vulkan
# (Linux / Windows) honors it via `OpMemoryBarrier(ScopeDevice, ...)`. Metal does NOT, even on Apple Silicon: MSL
# `atomic_thread_fence(memory_scope_device)` -- which is what MoltenVK / SPIRV-Cross translate
# `OpMemoryBarrier(ScopeDevice, ...)` to -- only orders *atomic* memory accesses across the device, not plain stores;
# this is a documented Metal limitation called out in the `grid.mem_fence()` Metal caveat in `grid.md`. So we exclude
# the native `metal` backend, and additionally skip `vulkan` on macOS at runtime, since on macOS Vulkan is really
# MoltenVK lowering SPIR-V to MSL and inherits the same limitation. Workloads that need cross-workgroup ordering on
# Metal (or Vulkan-on-Mac) have to publish through atomic stores (e.g. `qd.atomic_or(a[i], 1)`) rather than rely on a
# plain store + fence.
@test_utils.test(arch=qd.gpu, exclude=qd.metal)
def test_grid_mem_fence():
    if qd.lang.impl.current_cfg().arch == qd.vulkan and platform.system() == "Darwin":
        pytest.skip("Vulkan on macOS is MoltenVK->Metal; non-atomic stores are not ordered by grid.mem_fence")
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


# Verifies the deprecation warning fires for the legacy `qd.simt.grid.memfence()` spelling. `pytest.warns` enables
# `simplefilter("always")` for its scope, bypassing the project-wide
# `warnings.filterwarnings("once", ..., module="quadrants")` set in `quadrants/lang/misc.py`.
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
# distinguishes the two ops on every backend - a `global_thread_idx == thread_idx` aliasing regression fails one of the
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
# been removed.  See `test_subgroup_reduce_add_tiled` / `test_subgroup_reduce_all_add_tiled` below
# for the portable sized-reduction tests, and add equivalent sized portable replacements for the
# other reductions on top of `shuffle_down` / `shuffle` if needed.


def _init_field(field, n, dtype):
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(n):
        field[i] = (i + 1) if dtype in int_dtypes else 1.0000000000001 * (i + 1)


# --- Block reduce tests ----------------------------------------------------------------
#
# `qd.simt.block.reduce_{add,min,max}` is a two-stage block reduce: per-subgroup `shuffle_down` tree, lane 0 of each
# subgroup publishes the subgroup aggregate to shared memory, then thread 0 sequentially folds the subgroup aggregates.
# Result is valid in thread 0 only; the `reduce_all_*` variants broadcast it to every thread via one extra
# `block.sync()` plus a one-slot shared-memory hop.
#
# We exercise three regimes per arch by parameterizing on subgroups-per-block rather than absolute block_dim:
# 1 subgroup (single-subgroup short-circuit path - no shared memory, no cross-subgroup fold), 4 subgroups
# (multi-subgroup), 8 subgroups (multi-subgroup, larger).  The host-side ``_arch_subgroup_size()`` maps to ``block_dim``
# at test-body entry, so wave32 archs (CUDA / Metal / NVIDIA Vulkan) get ``[32, 128, 256]`` and wave64 (AMDGPU) gets
# ``[64, 256, 512]`` - both cover the single-subgroup short-circuit + multi-subgroup paths without skipping anything
# at collection time.  Inside the kernel, the subgroup size is still read from ``subgroup.group_size()`` at compile
# time, so the same source compiles correctly on every backend without an API knob.

_BLOCK_REDUCE_DTYPES = [qd.i32, qd.f32]
_BLOCK_REDUCE_SG_PER_BLOCK = [1, 4, 8]


def _arch_subgroup_size():
    """Return the subgroup size for the active arch (host side).

    AMDGPU is pinned to wave64 in Quadrants; every other supported arch is wave32.  This is the host-side mirror of
    the kernel-side ``subgroup.group_size()`` and is used by block-* tests to derive ``block_dim`` from a
    subgroups-per-block parameter so each arch tests its own canonical sizes.
    """
    return 64 if qd.lang.impl.current_cfg().arch == qd.amdgpu else 32


def _ref_reduce_add(values):
    return sum(values)


def _ref_reduce_min(values):
    return min(values)


def _ref_reduce_max(values):
    return max(values)


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_reduce_add(dtype, sg_per_block):
    """Block sum-reduce: thread 0 of each block holds `sum(src[block_base:block_base+block_dim])`."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=NUM_BLOCKS)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            agg = block.reduce_add(src[i], block_dim, dtype)
            if tid == 0:
                dst[i // block_dim] = agg

    _init_field(src, N, dtype)
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_reduce_add(block_vals)
        if dtype == qd.i32:
            assert dst[b] == expected, f"block {b}: got {dst[b]}, expected {expected}"
        else:
            assert abs(dst[b] - expected) < 1e-4 * abs(expected), f"block {b}: got {dst[b]}, expected {expected}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_reduce_min(dtype, sg_per_block):
    """Block min-reduce: thread 0 of each block holds `min(src[block_base:block_base+block_dim])`."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=NUM_BLOCKS)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            agg = block.reduce_min(src[i], block_dim, dtype)
            if tid == 0:
                dst[i // block_dim] = agg

    # Permuted (non-monotone) initialisation so the min depends on lanes other than the first / last.
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1  # in [1, 997]; stable hash, no collisions w/ block_dim values up to 256
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_reduce_min(block_vals)
        if dtype == qd.i32:
            assert dst[b] == expected, f"block {b}: got {dst[b]}, expected {expected}"
        else:
            assert abs(dst[b] - expected) < 1e-5, f"block {b}: got {dst[b]}, expected {expected}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_reduce_max(dtype, sg_per_block):
    """Block max-reduce: thread 0 of each block holds `max(src[block_base:block_base+block_dim])`."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=NUM_BLOCKS)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            agg = block.reduce_max(src[i], block_dim, dtype)
            if tid == 0:
                dst[i // block_dim] = agg

    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_reduce_max(block_vals)
        if dtype == qd.i32:
            assert dst[b] == expected, f"block {b}: got {dst[b]}, expected {expected}"
        else:
            assert abs(dst[b] - expected) < 1e-5, f"block {b}: got {dst[b]}, expected {expected}"


@pytest.mark.parametrize("sg_per_block", [4, 8])
@test_utils.test(arch=qd.gpu)
def test_block_reduce_add_many_blocks(sg_per_block):
    """Block sum-reduce stress test: many concurrent blocks driving the multi-subgroup shared-memory combine path.

    Regression guard for an AMDGPU bug where ``block.sync()`` lowered to a bare ``llvm.amdgcn.s.barrier`` (without the
    surrounding ``fence release / acquire syncscope("workgroup")`` pair that HIP's ``__syncthreads()`` emits via
    ``__work_group_barrier``). On RDNA3 (``gfx1100``) at ``BLOCK_DIM>=4*SUBGROUP_SIZE`` and ``NUM_BLOCKS>=~200``, the
    ``shared[w]`` reads in ``block.reduce``'s inter-subgroup fold would intermittently see uninitialized LDS, leaking
    seemingly-random 4-byte values into the per-block aggregate; lower block counts and ``sg_per_block in [1, 2]``
    didn't trip it. Patched in ``quadrants/runtime/llvm/llvm_context.cpp`` to inline the fence-barrier-fence sequence
    in the AMDGPU ``block_barrier`` body. We stay on ``i32`` add for deterministic comparison and only stress
    ``sg_per_block in [4, 8]`` (the multi-subgroup combine path); single-subgroup short-circuits the LDS hop and isn't
    affected.
    """
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 512
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=NUM_BLOCKS)
    rng = np.random.default_rng(20260513)
    src_np = rng.integers(low=-1000, high=1000, size=N, dtype=np.int32)
    src.from_numpy(src_np)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            tid = i % block_dim
            agg = block.reduce_add(src[i], block_dim, qd.i32)
            if tid == 0:
                dst[i // block_dim] = agg

    foo()
    got = dst.to_numpy()
    expected = src_np.reshape(NUM_BLOCKS, block_dim).sum(axis=1, dtype=np.int32)
    bad = np.where(got != expected)[0]
    assert (
        len(bad) == 0
    ), f"{len(bad)} of {NUM_BLOCKS} block reductions wrong; first 5 bad blocks {bad[:5]}, got {got[bad[:5]]}, expected {expected[bad[:5]]}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_reduce_all_add(dtype, sg_per_block):
    """Block sum-reduce broadcast: every thread of each block holds the block-wide sum.

    Verifies the broadcast variant by writing the per-thread output to a flat field, then asserting every thread of a
    given block reads the same aggregate.
    """
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.reduce_all_add(src[i], block_dim, dtype)

    _init_field(src, N, dtype)
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_reduce_add(block_vals)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected, f"block {b} thread {j}: got {actual}, expected {expected}"
            else:
                assert abs(actual - expected) < 1e-4 * abs(
                    expected
                ), f"block {b} thread {j}: got {actual}, expected {expected}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_reduce_all_min(dtype, sg_per_block):
    """Block min-reduce broadcast: every thread reads the block-wide min."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.reduce_all_min(src[i], block_dim, dtype)

    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_reduce_min(block_vals)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected, f"block {b} thread {j}: got {actual}, expected {expected}"
            else:
                assert abs(actual - expected) < 1e-5, f"block {b} thread {j}: got {actual}, expected {expected}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_reduce_all_max(dtype, sg_per_block):
    """Block max-reduce broadcast: every thread reads the block-wide max."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.reduce_all_max(src[i], block_dim, dtype)

    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_reduce_max(block_vals)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected, f"block {b} thread {j}: got {actual}, expected {expected}"
            else:
                assert abs(actual - expected) < 1e-5, f"block {b} thread {j}: got {actual}, expected {expected}"


# --- Block scan tests ------------------------------------------------------------------
#
# `qd.simt.block.{inclusive,exclusive}_{add,min,max}` is a two-stage block scan: per-subgroup Hillis-Steele scan via
# shuffle, last lane of each subgroup publishes the subgroup aggregate to shared memory, then every thread sequentially
# folds the cross-subgroup prefix and applies its own subgroup's prefix.  Every thread receives a valid result.
#
# We exercise the same three regimes as block reduce (1 / 4 / 8 subgroups per block, derived to absolute block_dim by
# the host helper at test-body entry) and assert per-thread against a sequential CPU oracle.  The min / max tests use
# a permuted (non-monotone) input so the scan result genuinely depends on every prefix step, not just the trailing or
# leading element.


def _ref_inclusive_scan_add(values):
    out = []
    acc = 0
    for v in values:
        acc = acc + v
        out.append(acc)
    return out


def _ref_exclusive_scan_add(values):
    out = []
    acc = 0
    for v in values:
        out.append(acc)
        acc = acc + v
    return out


def _ref_inclusive_scan_op(values, op, identity):
    out = []
    acc = identity
    first = True
    for v in values:
        acc = v if first else op(acc, v)
        first = False
        out.append(acc)
    return out


def _ref_exclusive_scan_op(values, op, identity):
    out = []
    acc = identity
    for v in values:
        out.append(acc)
        acc = op(acc, v)
    return out


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_inclusive_add(dtype, sg_per_block):
    """Block inclusive prefix sum: thread `i` holds `sum(src[block_base..i])`."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.inclusive_add(src[i], block_dim, dtype)

    _init_field(src, N, dtype)
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_inclusive_scan_add(block_vals)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected[j], f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            else:
                assert abs(actual - expected[j]) < 1e-4 * abs(
                    expected[j] + 1.0
                ), f"block {b} thread {j}: got {actual}, expected {expected[j]}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_exclusive_add(dtype, sg_per_block):
    """Block exclusive prefix sum: thread `i` holds `sum(src[block_base..i-1])`; thread 0 holds 0."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.exclusive_add(src[i], block_dim, dtype)

    _init_field(src, N, dtype)
    foo()

    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_exclusive_scan_add(block_vals)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected[j], f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            else:
                # First thread's expected is 0; gate the relative tolerance so it doesn't blow up.
                tol_base = abs(expected[j]) if abs(expected[j]) > 1.0 else 1.0
                assert (
                    abs(actual - expected[j]) < 1e-4 * tol_base
                ), f"block {b} thread {j}: got {actual}, expected {expected[j]}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_inclusive_min(dtype, sg_per_block):
    """Block inclusive prefix min."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.inclusive_min(src[i], block_dim, dtype)

    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    py_min = lambda a, b: a if a < b else b  # noqa: E731 (intentional 1-line lambda for ref oracle)
    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_inclusive_scan_op(block_vals, py_min, 0)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected[j], f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            else:
                assert abs(actual - expected[j]) < 1e-5, f"block {b} thread {j}: got {actual}, expected {expected[j]}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_inclusive_max(dtype, sg_per_block):
    """Block inclusive prefix max."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.inclusive_max(src[i], block_dim, dtype)

    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    py_max = lambda a, b: a if a > b else b  # noqa: E731
    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_inclusive_scan_op(block_vals, py_max, 0)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected[j], f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            else:
                assert abs(actual - expected[j]) < 1e-5, f"block {b} thread {j}: got {actual}, expected {expected[j]}"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_exclusive_min(dtype, sg_per_block):
    """Block exclusive prefix min; thread 0 holds the dtype-derived identity (``+inf`` / ``np.iinfo(dtype).max``)."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.exclusive_min(src[i], block_dim, dtype)

    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    sentinel = np.iinfo(np.int32).max if dtype == qd.i32 else float("inf")
    py_min = lambda a, b: a if a < b else b  # noqa: E731
    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_exclusive_scan_op(block_vals, py_min, sentinel)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected[j], f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            elif math.isinf(expected[j]):
                # Thread 0 of each block gets the +inf identity; ``inf - inf`` is NaN, so check by equality / sign.
                assert math.isinf(actual) and actual > 0, f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            else:
                assert abs(actual - expected[j]) < 1e-5, f"block {b} thread {j}: got {actual}, expected {expected[j]}"


# --- Block radix rank tests ------------------------------------------------------------
#
# `qd.simt.block.radix_rank_match_atomic_or` implements the atomic-OR match-and-count radix-rank strategy on top of the
# portable subgroup primitives (lanemask_le, sync, shuffle) and the block exclusive scan defined above.  Block size
# and digit count are both 256 (one digit per thread); each thread contributes one u32 key.
#
# We test the algorithm end-to-end against a CPU oracle:
#
#   - rank[i] = excl_prefix[digit[i]] + (#j < i with digit[j] == digit[i])
#   - bins[d] = count of keys whose digit equals d
#   - excl_prefix[d] = sum(bins[0..d-1])
#
# Inputs are mixed: a low-entropy distribution that hits every digit multiple times (so the leader-election +
# atomic_or match path actually has work to do) and a uniform random distribution (covers the case where most digits
# get ~1 key each).  Both distributions also probe the subgroup-level dedup logic with multiple keys-per-subgroup
# landing in the same digit bin.


_RADIX_BITS = 8
_RADIX_DIGITS = 1 << _RADIX_BITS  # 256
_BLOCK_DIM_RR = _RADIX_DIGITS  # algorithm requires block_dim == RADIX_DIGITS


def _ref_radix_rank(keys, bit_start, num_bits):
    """CPU oracle for `block.radix_rank_match_atomic_or`.

    Returns ``(ranks, bins, excl_prefix)`` over a single tile of ``len(keys)`` u32 keys.  ``ranks[i]`` is the stable
    rank of ``keys[i]`` when keys are sorted by their ``[bit_start, bit_start + num_bits)`` digit; threads with the
    same digit are ordered by their original index.
    """
    n = len(keys)
    digits_count = 1 << num_bits
    mask = (1 << num_bits) - 1
    digits = [(int(k) >> bit_start) & mask for k in keys]
    bins = [0] * digits_count
    for d in digits:
        bins[d] += 1
    excl_prefix = [0] * digits_count
    for d in range(1, digits_count):
        excl_prefix[d] = excl_prefix[d - 1] + bins[d - 1]
    ranks = [0] * n
    seen = [0] * digits_count
    for i in range(n):
        d = digits[i]
        ranks[i] = excl_prefix[d] + seen[d]
        seen[d] += 1
    return ranks, bins, excl_prefix


@pytest.mark.parametrize(
    "key_pattern,bit_start,num_bits",
    [
        ("low_entropy", 0, 8),  # 16 distinct digits each appearing 16 times - heavy match path traffic
        ("uniform", 0, 8),  # full 8-bit uniform - most digits get 1 key, some get 0
        ("uniform_high_bits", 8, 8),  # digit drawn from bits [8, 16) - exercises bit_start > 0
    ],
)
@test_utils.test(arch=qd.gpu)
def test_block_radix_rank_match_atomic_or(key_pattern, bit_start, num_bits):
    """End-to-end test of `block.radix_rank_match_atomic_or` against a CPU oracle.

    Single block of ``RADIX_DIGITS == 256`` threads with one key each; we verify per-thread ``rank`` plus the per-digit
    ``bins`` and ``excl_prefix`` outparams.
    """
    keys_in = qd.field(dtype=qd.u32, shape=_BLOCK_DIM_RR)
    ranks_out = qd.field(dtype=qd.i32, shape=_BLOCK_DIM_RR)
    bins_out = qd.field(dtype=qd.i32, shape=_RADIX_DIGITS)
    excl_prefix_out = qd.field(dtype=qd.i32, shape=_RADIX_DIGITS)

    rng = np.random.default_rng(seed=1234)
    if key_pattern == "low_entropy":
        # Pick 16 distinct digit values and put 16 copies of each in random positions.  Picks land at every digit
        # boundary that the [bit_start, bit_start+num_bits) extraction would isolate.
        base_digits = rng.choice(_RADIX_DIGITS, size=16, replace=False)
        keys_py = np.repeat(base_digits.astype(np.uint32), 16)
        rng.shuffle(keys_py)
        # Stuff the digit into the relevant bits, leave the rest random so bit_start > 0 still has work.
        upper = rng.integers(0, 1 << 16, size=_BLOCK_DIM_RR, dtype=np.uint32)
        keys_py = ((upper << np.uint32(8)) | keys_py.astype(np.uint32)).astype(np.uint32)
    elif key_pattern == "uniform":
        keys_py = rng.integers(0, 1 << 16, size=_BLOCK_DIM_RR, dtype=np.uint32)
    elif key_pattern == "uniform_high_bits":
        keys_py = rng.integers(0, 1 << 24, size=_BLOCK_DIM_RR, dtype=np.uint32)
    else:
        raise ValueError(key_pattern)

    for i in range(_BLOCK_DIM_RR):
        keys_in[i] = int(keys_py[i])

    @qd.kernel
    def kern():
        qd.loop_config(block_dim=_BLOCK_DIM_RR)
        for i in range(_BLOCK_DIM_RR):
            tid = i % _BLOCK_DIM_RR
            bins_smem = block.SharedArray((_RADIX_DIGITS,), qd.i32)
            excl_smem = block.SharedArray((_RADIX_DIGITS,), qd.i32)
            rank = block.radix_rank_match_atomic_or(
                keys_in[i],
                _BLOCK_DIM_RR,
                _RADIX_BITS,
                bit_start,
                num_bits,
                bins_smem,
                excl_smem,
            )
            ranks_out[i] = rank
            if tid < _RADIX_DIGITS:
                bins_out[tid] = bins_smem[tid]
                excl_prefix_out[tid] = excl_smem[tid]

    kern()

    ref_ranks, ref_bins, ref_excl = _ref_radix_rank(keys_py.tolist(), bit_start, num_bits)

    actual_bins = [bins_out[d] for d in range(_RADIX_DIGITS)]
    assert actual_bins == ref_bins, f"bins mismatch (pattern={key_pattern})"

    actual_excl = [excl_prefix_out[d] for d in range(_RADIX_DIGITS)]
    assert actual_excl == ref_excl, f"excl_prefix mismatch (pattern={key_pattern})"

    actual_ranks = [ranks_out[i] for i in range(_BLOCK_DIM_RR)]
    # Ranks must be a permutation of [0, n) - uniqueness check first so any duplicate is caught even if the sorted
    # invariant below silently masks it.
    assert sorted(actual_ranks) == list(
        range(_BLOCK_DIM_RR)
    ), f"ranks not a permutation of [0, {_BLOCK_DIM_RR}) for pattern={key_pattern}"
    assert actual_ranks == ref_ranks, f"ranks mismatch (pattern={key_pattern})"


@pytest.mark.parametrize("dtype", _BLOCK_REDUCE_DTYPES)
@pytest.mark.parametrize("sg_per_block", _BLOCK_REDUCE_SG_PER_BLOCK)
@test_utils.test(arch=qd.gpu)
def test_block_exclusive_max(dtype, sg_per_block):
    """Block exclusive prefix max; thread 0 holds the dtype-derived identity (``-inf`` / ``np.iinfo(dtype).min``)."""
    block_dim = sg_per_block * _arch_subgroup_size()
    NUM_BLOCKS = 4
    N = NUM_BLOCKS * block_dim
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=block_dim)
        for i in range(N):
            dst[i] = block.exclusive_max(src[i], block_dim, dtype)

    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for i in range(N):
        v = ((i * 1009) % 997) + 1
        src[i] = v if dtype in int_dtypes else 1.0 * v
    foo()

    sentinel = np.iinfo(np.int32).min if dtype == qd.i32 else float("-inf")
    py_max = lambda a, b: a if a > b else b  # noqa: E731
    for b in range(NUM_BLOCKS):
        block_vals = [src[b * block_dim + j] for j in range(block_dim)]
        expected = _ref_exclusive_scan_op(block_vals, py_max, sentinel)
        for j in range(block_dim):
            actual = dst[b * block_dim + j]
            if dtype == qd.i32:
                assert actual == expected[j], f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            elif math.isinf(expected[j]):
                # Thread 0 of each block gets the -inf identity; ``-inf - -inf`` is NaN, so check by equality / sign.
                assert math.isinf(actual) and actual < 0, f"block {b} thread {j}: got {actual}, expected {expected[j]}"
            else:
                assert abs(actual - expected[j]) < 1e-5, f"block {b} thread {j}: got {actual}, expected {expected[j]}"


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


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_add_tiled(dtype, log2_size):
    """Portable shuffle_down tree reduction: lane 0 of each 2**log2_size group has the sum."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.reduce_add_tiled(src[i], log2_size)

    _init_field(src, N, dtype)
    foo()

    group_size = 1 << log2_size
    expected = sum(src[i] for i in range(group_size))
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    if dtype in int_dtypes:
        assert dst[0] == expected
    else:
        assert abs(dst[0] - expected) < 1e-4 * abs(expected)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_all_add_tiled(dtype, log2_size):
    """Portable butterfly XOR reduction: every lane in each 2**log2_size group has the sum."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.reduce_all_add_tiled(src[i], log2_size)

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


# Min / max reductions share the same kernel + verifier shape as `reduce_add_tiled` / `reduce_all_add_tiled`.
# `_check_reduce_lane0` and `_check_reduce_all` factor it out so the four tests below stay one-liners.
# Non-monotonic input (`_init_varied_int_or_float`) is required so each group's min / max actually depends on the
# reduction running over every lane in the group, not just the first or last one.


def _check_reduce_lane0(reduce_func, py_op, dtype, log2_size, src_init):
    """Verify lane-0 reductions: lane 0 of each 2**log2_size group holds op-reduce(src[group_base..]).

    Checks every group across the full 64-lane launch (two independent subgroups) so we exercise both the in-group
    reduction tree and the absence of cross-subgroup leakage.
    """
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = reduce_func(src[i], log2_size)

    src_init(src, N, dtype)
    foo()

    group_size = 1 << log2_size
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for g in range(N // group_size):
        group_base = g * group_size
        expected = src[group_base]
        for k in range(1, group_size):
            expected = py_op(expected, src[group_base + k])
        got = dst[group_base]
        if dtype in int_dtypes:
            assert got == expected, f"group {g} lane 0 (global {group_base}): got {got}, expected {expected}"
        else:
            assert abs(got - expected) < 1e-5 * max(
                abs(expected), 1.0
            ), f"group {g} lane 0 (global {group_base}): got {got}, expected {expected}"


def _check_reduce_all(reduce_func, py_op, dtype, log2_size, src_init):
    """Verify broadcast reductions: every lane of each 2**log2_size group holds op-reduce(src[group_base..])."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = reduce_func(src[i], log2_size)

    src_init(src, N, dtype)
    foo()

    group_size = 1 << log2_size
    int_dtypes = (qd.i32, qd.i64, qd.u64)
    for g in range(N // group_size):
        group_base = g * group_size
        expected = src[group_base]
        for k in range(1, group_size):
            expected = py_op(expected, src[group_base + k])
        for k in range(group_size):
            global_lane = group_base + k
            got = dst[global_lane]
            if dtype in int_dtypes:
                assert got == expected, f"group {g} lane {k} (global {global_lane}): got {got}, expected {expected}"
            else:
                assert abs(got - expected) < 1e-5 * max(
                    abs(expected), 1.0
                ), f"group {g} lane {k} (global {global_lane}): got {got}, expected {expected}"


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_min_tiled(dtype, log2_size):
    """Portable shuffle_down tree min: lane 0 of each 2**log2_size group has the group min."""
    _check_reduce_lane0(subgroup.reduce_min_tiled, lambda a, b: min(a, b), dtype, log2_size, _init_varied_int_or_float)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_max_tiled(dtype, log2_size):
    """Portable shuffle_down tree max: lane 0 of each 2**log2_size group has the group max."""
    _check_reduce_lane0(subgroup.reduce_max_tiled, lambda a, b: max(a, b), dtype, log2_size, _init_varied_int_or_float)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_all_min_tiled(dtype, log2_size):
    """Portable butterfly min: every lane in each 2**log2_size group has the group min."""
    _check_reduce_all(
        subgroup.reduce_all_min_tiled, lambda a, b: min(a, b), dtype, log2_size, _init_varied_int_or_float
    )


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_all_max_tiled(dtype, log2_size):
    """Portable butterfly max: every lane in each 2**log2_size group has the group max."""
    _check_reduce_all(
        subgroup.reduce_all_max_tiled, lambda a, b: max(a, b), dtype, log2_size, _init_varied_int_or_float
    )


# Segmented reduce.  Tests use a head-flag pattern with multiple segments per group, varied widths, and at least one
# group where lane 0 is *not* a head (so we exercise the implicit-head-at-group_base fallback).  The Python verifier
# replicates the algorithm: lane i ends up holding sum(value[head_below..i+1]), where head_below is the largest lane
# index <= i within the lane's 2**log2_size group whose head_flag is non-zero, with group_base treated as an implicit
# head when no real head exists.


def _python_segmented_reduce(values, heads, group_size, op):
    """Reference: lane i within each group_size-sized group holds ``op-reduce(values[head_below..i+1])``.

    ``op`` is a binary function on Python scalars (e.g. ``operator.add``, ``min``, ``max``).
    """
    import functools as _ft

    n = len(values)
    out = [None] * n
    for g_base in range(0, n, group_size):
        for k in range(group_size):
            i = g_base + k
            # Find the highest lane <= i (within the group) that has a non-zero head, default to g_base.
            head = g_base
            for h in range(g_base, i + 1):
                if heads[h]:
                    head = h
            out[i] = _ft.reduce(op, values[head : i + 1])
    return out


def _python_segmented_reduce_add(values, heads, group_size):
    """Reference: lane i within each group_size-sized group holds sum(values[head_below..i+1])."""
    import operator as _op

    return _python_segmented_reduce(values, heads, group_size, _op.add)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_SEGMENTED)
@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_add_tiled(dtype, log2_size):
    """Segmented inclusive sum, sized by ``log2_size`` (covers full warp at log2_size=5)."""
    _skip_if_f64_unsupported(dtype)
    N = 32  # ballot covers the first 32 lanes
    src = qd.field(dtype=dtype, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], log2_size)

    _init_varied_int_or_float(src, N, dtype)
    # Head pattern: heads at lanes 0, 3, 7, 12, 19, 25 - varied gaps, plus one group where the group base is not a head
    # (e.g. for log2_size=2, lane 4 is a group base but not a head, exercising the implicit head fallback).
    head_lanes = {0, 3, 7, 12, 19, 25}
    heads_py = [1 if i in head_lanes else 0 for i in range(N)]
    for i in range(N):
        head[i] = heads_py[i]

    foo()

    src_py = [int(src[i]) if dtype in _INT_DTYPES else float(src[i]) for i in range(N)]
    expected = _python_segmented_reduce_add(src_py, heads_py, 1 << log2_size)
    for i in range(N):
        got = dst[i]
        ref = expected[i]
        if dtype in _INT_DTYPES:
            assert got == ref, f"lane {i}: got {got}, expected {ref}"
        else:
            assert abs(got - ref) < 1e-5 * max(abs(ref), 1.0), f"lane {i}: got {got}, expected {ref}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_add_tiled_no_heads():
    """No head flags set anywhere -> the whole group is one segment, equivalent to inclusive_add_tiled(v, log2_size)."""
    N = 32
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], 5)

    for i in range(N):
        src[i] = i + 1
        head[i] = 0

    foo()

    # Inclusive prefix sum of 1..32: dst[k] = (k+1)*(k+2)/2
    for i in range(N):
        expected = (i + 1) * (i + 2) // 2
        assert dst[i] == expected, f"lane {i}: got {dst[i]}, expected {expected}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_add_tiled_every_lane_is_head():
    """Every lane is a head -> output equals input (each segment is exactly one lane)."""
    N = 32
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], 5)

    for i in range(N):
        src[i] = (i * 7 + 11) % 23 + 1
        head[i] = 1

    foo()

    for i in range(N):
        assert dst[i] == src[i], f"lane {i}: got {dst[i]}, expected {src[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_add_tiled_truthy_predicate():
    """Non-binary truthy values (e.g. 7, 42) should be treated identically to 1."""
    N = 32
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst_binary = qd.field(dtype=qd.i32, shape=N)
    dst_truthy = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def run_binary():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst_binary[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], 5)

    @qd.kernel
    def run_truthy():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst_truthy[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i] * 42, 5)

    for i in range(N):
        src[i] = i + 1
        head[i] = 1 if i in {0, 5, 13, 22} else 0

    run_binary()
    run_truthy()

    for i in range(N):
        assert dst_binary[i] == dst_truthy[i], f"lane {i}: binary={dst_binary[i]}, truthy={dst_truthy[i]}"


# Segmented reduce min / max - share the head-pattern + Python-verifier strategy with `segmented_reduce_add_tiled`,
# but the input data is a *non-monotonic* sequence (via ``_init_varied_int_or_float``) so each segment's min / max
# actually depends on the data and not just on the segment endpoints.


def _check_segmented_reduce(qd_op, py_op, dtype, log2_size):
    _skip_if_f64_unsupported(dtype)
    N = 32  # ballot covers the first 32 lanes
    src = qd.field(dtype=dtype, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = qd_op(src[i], head[i], log2_size)

    _init_varied_int_or_float(src, N, dtype)
    head_lanes = {0, 3, 7, 12, 19, 25}
    heads_py = [1 if i in head_lanes else 0 for i in range(N)]
    for i in range(N):
        head[i] = heads_py[i]

    foo()

    src_py = [int(src[i]) if dtype in _INT_DTYPES else float(src[i]) for i in range(N)]
    expected = _python_segmented_reduce(src_py, heads_py, 1 << log2_size, py_op)
    for i in range(N):
        got = dst[i]
        ref = expected[i]
        if dtype in _INT_DTYPES:
            assert got == ref, f"lane {i}: got {got}, expected {ref}"
        else:
            assert abs(got - ref) < 1e-5 * max(abs(ref), 1.0), f"lane {i}: got {got}, expected {ref}"


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_SEGMENTED)
@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_min_tiled(dtype, log2_size):
    """Segmented inclusive min, sized by ``log2_size`` (covers full warp at log2_size=5)."""
    _check_segmented_reduce(subgroup.segmented_reduce_min_tiled, lambda a, b: min(a, b), dtype, log2_size)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_SEGMENTED)
@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_max_tiled(dtype, log2_size):
    """Segmented inclusive max, sized by ``log2_size`` (covers full warp at log2_size=5)."""
    _check_segmented_reduce(subgroup.segmented_reduce_max_tiled, lambda a, b: max(a, b), dtype, log2_size)


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_min_tiled_no_heads():
    """No head flags set anywhere -> the whole group is one segment, equivalent to inclusive_min_tiled(v, log2_size)."""
    N = 32
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_min_tiled(src[i], head[i], 5)

    src_vals = [(i * 7 + 11) % 23 + 1 for i in range(N)]
    for i in range(N):
        src[i] = src_vals[i]
        head[i] = 0

    foo()

    expected = []
    running = src_vals[0]
    expected.append(running)
    for i in range(1, N):
        running = min(running, src_vals[i])
        expected.append(running)
    for i in range(N):
        assert dst[i] == expected[i], f"lane {i}: got {dst[i]}, expected {expected[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_max_tiled_every_lane_is_head():
    """Every lane is a head -> output equals input (each segment is exactly one lane)."""
    N = 32
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_max_tiled(src[i], head[i], 5)

    for i in range(N):
        src[i] = (i * 13 + 5) % 29 - 7  # mixed-sign data so max isn't just monotone
        head[i] = 1

    foo()

    for i in range(N):
        assert dst[i] == src[i], f"lane {i}: got {dst[i]}, expected {src[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_min_tiled_truthy_predicate():
    """Non-binary truthy values (e.g. 7, 42) should be treated identically to 1."""
    N = 32
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst_binary = qd.field(dtype=qd.i32, shape=N)
    dst_truthy = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def run_binary():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst_binary[i] = subgroup.segmented_reduce_min_tiled(src[i], head[i], 5)

    @qd.kernel
    def run_truthy():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst_truthy[i] = subgroup.segmented_reduce_min_tiled(src[i], head[i] * 7, 5)

    for i in range(N):
        src[i] = (i * 11 + 3) % 17 + 1
        head[i] = 1 if i in {0, 4, 9, 18, 27} else 0

    run_binary()
    run_truthy()

    for i in range(N):
        assert dst_binary[i] == dst_truthy[i], f"lane {i}: binary={dst_binary[i]}, truthy={dst_truthy[i]}"


@pytest.mark.parametrize("log2_size", [0, 1, 2, 3, 4, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_add_tiled_block64(log2_size):
    """Run ``segmented_reduce_add_tiled`` with ``block_dim=64`` and identical patterns in both 32-lane halves.

    On wave32 hardware (CUDA, RDNA wave32, Vulkan / Metal on most desktop GPUs) this dispatches as two independent
    32-lane subgroups, so each half exercises the lanes-0..31 path that the existing N=32 tests already cover.

    On wave64 hardware (AMDGPU CDNA, GFX9, RDNA explicit-wave64) this runs as a single 64-lane wavefront.  Lanes
    32..63 then need correct results inside their own ``log2_size`` group - which only works after the half-local
    ``_segment_head_distance_tiled`` rewrite (this commit).  Without the fix, lanes 32..63 hit u32 overshift /
    out-of-range bit-mask arithmetic and produce garbage.

    The test is structured so it passes on wave32 either way; the wave64 fix is what makes it pass on wave64.  We
    don't have wave64 hardware to test on right now (MI300X pending), but this case will then exercise it.
    """
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], log2_size)

    head_lanes_in_half = {0, 3, 7, 12, 19, 25}
    for i in range(N):
        src[i] = (i % 32) + 1  # 1..32 mirrored in each half
        head[i] = 1 if (i % 32) in head_lanes_in_half else 0

    foo()

    src_py_half = [j + 1 for j in range(32)]
    head_py_half = [1 if j in head_lanes_in_half else 0 for j in range(32)]
    expected_half = _python_segmented_reduce(src_py_half, head_py_half, 1 << log2_size, lambda a, b: a + b)
    for i in range(N):
        ref = expected_half[i % 32]
        got = dst[i]
        assert got == ref, f"lane {i} (half-local {i % 32}): got {got}, expected {ref}"


# Portable Hillis-Steele inclusive scans share the same kernel + Python verification pattern; the only thing that
# varies is the operator and which dtypes are legal for it.  `_check_inclusive_scan` factors out the kernel launch,
# dtype skip, and per-lane check.

_INT_DTYPES = (qd.i32, qd.i64, qd.u32, qd.u64)

# Per-dtype numpy companion type, used by the exclusive_min / exclusive_max wide-int identity tests to spell the
# expected lane-0 identity (``np.iinfo(dtype).max`` for min, ``.min`` for max).  Bool / 8-bit / 16-bit are omitted
# because the underlying subgroup ops are exercised only at 32 / 64 bit widths.
_NP_FOR_INT_DTYPE = {
    qd.i32: np.int32,
    qd.i64: np.int64,
    qd.u32: np.uint32,
    qd.u64: np.uint64,
}


def _exclusive_min_lane0_identity(dtype):
    """Expected lane-0 value for ``exclusive_min_tiled`` -- mirrors what ``_typed_min_identity`` emits in
    ``quadrants/lang/simt/subgroup.py``: ``+inf`` for real dtypes, ``np.iinfo(dtype).max`` for integer dtypes."""
    if dtype in (qd.f32, qd.f64):
        return float("inf")
    return int(np.iinfo(_NP_FOR_INT_DTYPE[dtype]).max)


def _exclusive_max_lane0_identity(dtype):
    """Expected lane-0 value for ``exclusive_max_tiled`` -- ``-inf`` for real dtypes, ``np.iinfo(dtype).min`` for
    integer dtypes (``0`` for the unsigned widths)."""
    if dtype in (qd.f32, qd.f64):
        return float("-inf")
    return int(np.iinfo(_NP_FOR_INT_DTYPE[dtype]).min)


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


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_add_tiled(dtype, log2_size):
    """Portable inclusive prefix sum: lane k of each 2**log2_size group has sum(src[group_base..group_base+k+1])."""
    _check_inclusive_scan(subgroup.inclusive_add_tiled, lambda a, b: a + b, dtype, log2_size, _init_field)


def _init_small_int_or_float(field, n, dtype):
    """Bound the 32-way product to 2**8 == 256 (i32-safe, f32-exact): most lanes hold 1, every fourth lane holds 2.
    With log2_size=5 (32 lanes / group), 8 lanes contribute a factor of 2 → product ≤ 256.  Smaller log2_size only sees
    a subset."""
    for i in range(n):
        field[i] = 2 if (i % 4 == 0) else 1


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_I32_AND_FLOATS)
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_mul_tiled(dtype, log2_size):
    """Inclusive prefix product.  Inputs are 1 / 2 mixed so the 32-way product is at most 2**8 == 256 (well within i32
    and f32-exact)."""
    _check_inclusive_scan(subgroup.inclusive_mul_tiled, lambda a, b: a * b, dtype, log2_size, _init_small_int_or_float)


def _init_varied_int_or_float(field, n, dtype):
    """Mix increasing and decreasing values so prefix-min / prefix-max have non-trivial transitions across the
    group."""
    for i in range(n):
        # 11, 7, 13, 3, 17, 5, 19, ...  -- varied, non-monotonic, non-negative
        field[i] = ((i * 7 + 11) % 23) + 1


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_I32_AND_FLOATS)
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_min_tiled(dtype, log2_size):
    """Inclusive prefix min."""
    _check_inclusive_scan(
        subgroup.inclusive_min_tiled, lambda a, b: min(a, b), dtype, log2_size, _init_varied_int_or_float
    )


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_I32_AND_FLOATS)
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_max_tiled(dtype, log2_size):
    """Inclusive prefix max."""
    _check_inclusive_scan(
        subgroup.inclusive_max_tiled, lambda a, b: max(a, b), dtype, log2_size, _init_varied_int_or_float
    )


def _init_bitwise_int(field, n, dtype):
    """Initialise with bit-varied integer values so prefix &/|/^ have meaningful transitions (bits flip on and off
    across lanes)."""
    for i in range(n):
        field[i] = (i * 5 + 0x37) & 0xFF


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_INT)
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_and_tiled(dtype, log2_size):
    """Inclusive prefix bitwise-AND.  Integer dtypes only."""
    _skip_if_f64_unsupported(dtype)  # also handles 64-bit-int Metal/MoltenVK skips
    _check_inclusive_scan(subgroup.inclusive_and_tiled, lambda a, b: a & b, dtype, log2_size, _init_bitwise_int)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_INT)
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_or_tiled(dtype, log2_size):
    """Inclusive prefix bitwise-OR.  Integer dtypes only."""
    _skip_if_f64_unsupported(dtype)
    _check_inclusive_scan(subgroup.inclusive_or_tiled, lambda a, b: a | b, dtype, log2_size, _init_bitwise_int)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_INT)
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_xor_tiled(dtype, log2_size):
    """Inclusive prefix bitwise-XOR.  Integer dtypes only."""
    _skip_if_f64_unsupported(dtype)
    _check_inclusive_scan(subgroup.inclusive_xor_tiled, lambda a, b: a ^ b, dtype, log2_size, _init_bitwise_int)


# --- Exclusive scans ------------------------------------------------------------------
#
# Same kernel + verification shape as the inclusive tests above; the only differences are (1) lane 0 of each group is
# expected to hold the operator's identity, and (2) for min/max the wrapper takes an explicit `identity` arg that we
# pass through here.


def _check_exclusive_scan(scan_func, py_op, py_identity, dtype, log2_size, src_init):
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
            elif math.isinf(expected) or math.isnan(expected):
                assert got == expected, f"group {g} lane {k} (global {global_lane}): got {got}, expected {expected}"
            else:
                assert abs(got - expected) < 1e-4 * max(
                    abs(expected), 1.0
                ), f"group {g} lane {k} (global {global_lane}): got {got}, expected {expected}"


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_add_tiled(dtype, log2_size):
    """Exclusive prefix sum.  Lane 0 of each group is 0."""
    _check_exclusive_scan(subgroup.exclusive_add_tiled, lambda a, b: a + b, 0, dtype, log2_size, _init_field)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_I32_AND_FLOATS)
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_mul_tiled(dtype, log2_size):
    """Exclusive prefix product.  Lane 0 of each group is 1."""
    _check_exclusive_scan(
        subgroup.exclusive_mul_tiled, lambda a, b: a * b, 1, dtype, log2_size, _init_small_int_or_float
    )


# Unlike inclusive_mul / inclusive_min / inclusive_max / exclusive_mul which all live in
# ``_SCENARIOS_I32_AND_FLOATS`` (no dtype-specific identity code path -- they either don't need a lane-0 sentinel or
# inherit the universal ``1`` literal), exclusive_min / exclusive_max emit a dtype-typed identity constant in the
# generated IR (``np.iinfo(dtype).{max,min}`` / ``+-inf``).  The u64 max path in particular routes through
# ``_clamp_unsigned_to_range`` (val > int64 max -> two's-complement conversion to -1).  Extend the parameter set
# here so every supported int width and signedness exercises the identity-emission code in ``_typed_min_identity`` /
# ``_typed_max_identity``.
_SCENARIOS_EXCLUSIVE_MINMAX = _SCENARIOS_I32_AND_FLOATS + [(qd.i64, 5), (qd.u32, 5), (qd.u64, 5)]


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_EXCLUSIVE_MINMAX)
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_min_tiled(dtype, log2_size):
    """Exclusive prefix min.  Lane 0 of each group is the dtype-typed identity (+inf for floats,
    ``np.iinfo(dtype).max`` for ints) -- the wrapper auto-derives it from ``value``'s dtype, so the caller doesn't
    pass one."""
    _check_exclusive_scan(
        subgroup.exclusive_min_tiled,
        lambda a, b: min(a, b),
        _exclusive_min_lane0_identity(dtype),
        dtype,
        log2_size,
        _init_varied_int_or_float,
    )


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_EXCLUSIVE_MINMAX)
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_max_tiled(dtype, log2_size):
    """Exclusive prefix max.  Lane 0 of each group is the dtype-typed identity (-inf for floats,
    ``np.iinfo(dtype).min`` for ints; ``0`` for unsigned widths)."""
    _check_exclusive_scan(
        subgroup.exclusive_max_tiled,
        lambda a, b: max(a, b),
        _exclusive_max_lane0_identity(dtype),
        dtype,
        log2_size,
        _init_varied_int_or_float,
    )


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_INT)
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_and_tiled(dtype, log2_size):
    """Exclusive prefix bitwise-AND.  Lane 0 of each group is all-bits-set."""
    _skip_if_f64_unsupported(dtype)
    if dtype == qd.i32:
        identity = -1
    elif dtype == qd.i64:
        identity = -1
    else:  # u64
        identity = (1 << 64) - 1
    _check_exclusive_scan(
        subgroup.exclusive_and_tiled, lambda a, b: a & b, identity, dtype, log2_size, _init_bitwise_int
    )


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_INT)
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_or_tiled(dtype, log2_size):
    """Exclusive prefix bitwise-OR.  Lane 0 of each group is 0."""
    _skip_if_f64_unsupported(dtype)
    _check_exclusive_scan(subgroup.exclusive_or_tiled, lambda a, b: a | b, 0, dtype, log2_size, _init_bitwise_int)


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_INT)
@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_xor_tiled(dtype, log2_size):
    """Exclusive prefix bitwise-XOR.  Lane 0 of each group is 0."""
    _skip_if_f64_unsupported(dtype)
    _check_exclusive_scan(subgroup.exclusive_xor_tiled, lambda a, b: a ^ b, 0, dtype, log2_size, _init_bitwise_int)


# Voting / predicate ops.  All three are group-scoped over 2**log2_size lanes; the scenario tables below exercise (a)
# every-lane-true / every-lane-false, (b) a single odd lane in one group with the rest all-true / all-false (group
# isolation), and (c) a sparse pattern that lands several groups in the all-true case and several in the mixed case so
# the per-group reduction is the only thing distinguishing them.  We verify every group across the full 64-lane launch
# (so log2_size in {1..4} covers the multi-group case and log2_size==5 covers the full-warp / CUDA fast-path case, and
# the launch spans two CUDA / Metal / RDNA subgroups so cross-subgroup leakage would also be caught).


@pytest.mark.parametrize("log2_size", [1, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_all_true_tiled(log2_size):
    """``all_true_tiled(predicate, log2_size)`` is ``i32(all(predicate != 0))`` over each ``2**log2_size`` group,
    broadcast to every lane in the group."""
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.all_true_tiled(src[i], log2_size)

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


@pytest.mark.parametrize("log2_size", [1, 5])
@test_utils.test(arch=qd.gpu)
def test_subgroup_any_true_tiled(log2_size):
    """``any_true_tiled(predicate, log2_size)`` is ``i32(any(predicate != 0))`` over each ``2**log2_size`` group,
    broadcast to every lane in the group."""
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.any_true_tiled(src[i], log2_size)

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


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FULL_DTYPE)
@test_utils.test(arch=qd.gpu)
def test_subgroup_all_equal_tiled(dtype, log2_size):
    """``all_equal_tiled(value, log2_size)`` is ``i32(all values equal)`` over each ``2**log2_size`` group, broadcast to
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
            dst[i] = subgroup.all_equal_tiled(src[i], log2_size)

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


@pytest.mark.parametrize("dtype, log2_size", _SCENARIOS_FLOAT)
@test_utils.test(arch=qd.gpu)
def test_subgroup_all_equal_tiled_float_contract(dtype, log2_size):
    """``all_equal_tiled`` on floats uses the backend's native ``==``: ``NaN != NaN`` and ``+0.0 == -0.0``, matching
    SPIR-V ``OpGroupNonUniformAllEqual``.  Locks both contracts so a future refactor (e.g. swapping in
    ``__match_all_sync`` on CUDA) can't silently regress to bit-equality."""
    _skip_if_f64_unsupported(dtype)
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.all_equal_tiled(src[i], log2_size)

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
def test_subgroup_ballot_first_n_all_true():
    """``ballot_first_n(p=1, n=32)`` with every lane voting true returns the full 32-bit bitmask (lanes 0..31)."""
    N = 32
    result = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            result[i] = subgroup.ballot_first_n(1, 32)

    foo()

    for i in range(N):
        assert result[i] == 0xFFFFFFFF, f"lane {i}: ballot returned {result[i]:#x}, expected 0xFFFFFFFF"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_first_n_all_false():
    """``ballot_first_n(p=0, n=32)`` with every lane voting false returns zero."""
    N = 32
    result = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            result[i] = subgroup.ballot_first_n(0, 32)

    foo()

    for i in range(N):
        assert result[i] == 0, f"lane {i}: ballot returned {result[i]}, expected 0"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_first_n_even_lanes():
    """Even-numbered lanes vote true; odd lanes vote false.  Result is the alternating bit pattern 0x55555555."""
    N = 32
    result = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            result[i] = subgroup.ballot_first_n(1 - lane % 2, 32)

    foo()

    mask = result[0]
    assert mask == 0x55555555, f"got {mask:#x}, expected 0x55555555 (even lanes set, odd lanes clear)"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_first_n_popcount():
    """popcount of ``ballot_first_n(1, 32)`` equals 32 (the u32 mask is full); group_size may be 32 or 64."""
    N = 32
    ballot_val = qd.field(dtype=qd.u32, shape=N)
    sg_size = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            ballot_val[i] = subgroup.ballot_first_n(1, 32)
            sg_size[i] = subgroup.group_size()

    foo()

    bv = ballot_val[0]
    sz = sg_size[0]
    actual_popcount = bin(bv).count("1")
    assert actual_popcount == 32, f"popcount({bv:#x}) = {actual_popcount}, expected 32 (subgroup size {sz})"


@pytest.mark.parametrize("n", [1, 4, 8, 16, 24, 31, 32])
@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_first_n_partial(n):
    """``ballot_first_n(1, n)`` with all lanes voting true returns ``(1 << n) - 1`` (or ``0xFFFFFFFF`` at ``n=32``).

    Exercises the ``n < 32`` masking path: lanes ``[n, 32)`` must contribute zero to the result, so bits ``[n, 32)``
    of the mask must be clear regardless of those lanes' actual predicates.  ``n == 32`` exercises the shortcut
    (no masking).
    """
    N = 32
    result = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            result[i] = subgroup.ballot_first_n(1, n)

    foo()

    expected = 0xFFFFFFFF if n == 32 else (1 << n) - 1
    for i in range(N):
        assert result[i] == expected, f"n={n} lane {i}: got {result[i]:#x}, expected {expected:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_first_n_partial_truthy_per_lane():
    """``ballot_first_n(predicate, n)`` with a per-lane-varying predicate: each lane votes ``lane & 1``, so the
    result over the first ``n`` lanes is ``(0xAAAAAAAA & ((1 << n) - 1))``.  Verifies that the masking really does
    select lanes ``< n`` and not e.g. ``< 32`` always."""
    N = 32
    result = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            result[i] = subgroup.ballot_first_n(lane % 2, 16)

    foo()

    expected = 0xAAAAAAAA & 0xFFFF  # lanes 0..15: bits 1, 3, 5, ..., 15 set; bits 16..31 clear
    for i in range(N):
        assert result[i] == expected, f"lane {i}: got {result[i]:#x}, expected {expected:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_all_true():
    """``ballot(1)`` with every lane voting true returns a u64 bitmask covering the whole subgroup.
    On wave32 we expect ``0xFFFFFFFF`` (low 32 bits set, high 32 zero); on wave64 we expect ``0xFFFFFFFFFFFFFFFF``.

    Uses ``block_dim=64`` so every wave has every lane active on both wave32 (two full 32-lane waves) and wave64
    (one full 64-lane wave) - required to satisfy the ``ballot`` "all lanes active" contract on wave64.
    """
    N = 64
    result = qd.field(dtype=qd.u64, shape=N)
    sg_size = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            result[i] = subgroup.ballot(1)
            sg_size[i] = subgroup.group_size()

    foo()

    sz = sg_size[0]
    assert sz in (32, 64), f"unexpected group_size {sz}"
    expected = (1 << sz) - 1  # 0xFFFFFFFF on wave32, 0xFFFFFFFFFFFFFFFF on wave64
    for i in range(N):
        assert result[i] == expected, f"lane {i}: got {result[i]:#x}, expected {expected:#x} (subgroup size {sz})"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_all_false():
    """``ballot(0)`` returns zero everywhere - verifies no spurious bits leak in from
    uninitialised wave64 high-half lanes."""
    N = 32
    result = qd.field(dtype=qd.u64, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            result[i] = subgroup.ballot(0)

    foo()

    for i in range(N):
        assert result[i] == 0, f"lane {i}: got {result[i]:#x}, expected 0"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_even_lanes():
    """Even lanes vote true; odd lanes vote false.  Result: ``0x5555...`` over the whole subgroup width.

    On wave32 we get ``0x0000000055555555`` (low 32 bits, high 32 zero); on wave64 we get
    ``0x5555555555555555`` (all 64 bits in the alternating pattern).

    Uses ``block_dim=64`` to keep every lane active on wave64 (required by the ``ballot`` "all lanes
    active" contract); on wave32 the workgroup splits into two waves and each wave's ballot covers its own 32 lanes.
    """
    N = 64
    result = qd.field(dtype=qd.u64, shape=N)
    sg_size = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            result[i] = subgroup.ballot(1 - lane % 2)
            sg_size[i] = subgroup.group_size()

    foo()

    sz = sg_size[0]
    assert sz in (32, 64), f"unexpected group_size {sz}"
    if sz == 32:
        expected = 0x55555555
    else:
        expected = 0x5555555555555555
    for i in range(N):
        assert result[i] == expected, f"lane {i}: got {result[i]:#x}, expected {expected:#x} (sg_size {sz})"


@test_utils.test(arch=qd.gpu)
def test_subgroup_ballot_high_half_only():
    """Only lanes ``>= 32`` vote true.  On wave32 the result is zero (no such lanes exist); on wave64 the result is
    ``0xFFFFFFFF00000000``.  This is the test that distinguishes a *correct* wave64 implementation from a wave32-only
    one: a broken wave64 path that always uses the i32 ballot would silently report 0 here."""
    N = 64
    result = qd.field(dtype=qd.u64, shape=N)
    sg_size = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            result[i] = subgroup.ballot(qd.i32(lane >= qd.u32(32)))
            sg_size[i] = subgroup.group_size()

    foo()

    sz = sg_size[0]
    assert sz in (32, 64), f"unexpected group_size {sz}"
    if sz == 32:
        # No lane has lane_id >= 32, so the result is zero.
        expected = 0
    else:
        # Lanes 32..63 vote true, lanes 0..31 vote false: result is 0xFFFFFFFF00000000.
        expected = 0xFFFFFFFF00000000
    for i in range(N):
        # Each subgroup independently produces `expected`; check the first lane of each subgroup.
        if i % sz == 0:
            assert result[i] == expected, f"subgroup-leader lane {i}: got {result[i]:#x}, expected {expected:#x}"


# Lane masks: each lane queries its own mask via ``invocation_id()`` and we cross-check against the closed-form
# ``(1 << lane) - 1`` etc.  We cap at 32 lanes (the ``u32`` mask range), regardless of ``group_size()`` - wave64 lanes
# 32..63 are not representable in this op.


@test_utils.test(arch=qd.gpu)
def test_subgroup_lanemask_lt():
    """``lanemask_lt(lane)`` returns ``(1 << lane) - 1`` for every lane in ``[0, 31]``."""
    N = 32
    out = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.lanemask_lt(subgroup.invocation_id())

    foo()
    for i in range(N):
        expected = (1 << i) - 1
        assert out[i] == expected, f"lane {i}: got {out[i]:#x}, expected {expected:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_lanemask_le():
    """``lanemask_le(lane)`` covers bits ``[0..lane]``; lane 31 must give ``0xFFFFFFFF``."""
    N = 32
    out = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.lanemask_le(subgroup.invocation_id())

    foo()
    for i in range(N):
        expected = ((1 << i) | ((1 << i) - 1)) & 0xFFFFFFFF
        assert out[i] == expected, f"lane {i}: got {out[i]:#x}, expected {expected:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_lanemask_eq():
    """``lanemask_eq(lane)`` is exactly one bit at ``lane``."""
    N = 32
    out = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.lanemask_eq(subgroup.invocation_id())

    foo()
    for i in range(N):
        expected = 1 << i
        assert out[i] == expected, f"lane {i}: got {out[i]:#x}, expected {expected:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_lanemask_gt():
    """``lanemask_gt(lane)`` covers bits strictly above ``lane``; lane 31 must give 0."""
    N = 32
    out = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.lanemask_gt(subgroup.invocation_id())

    foo()
    for i in range(N):
        expected = (~(((1 << i) | ((1 << i) - 1)))) & 0xFFFFFFFF
        assert out[i] == expected, f"lane {i}: got {out[i]:#x}, expected {expected:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_lanemask_ge():
    """``lanemask_ge(lane)`` covers bits ``>= lane``; lane 0 must give ``0xFFFFFFFF``."""
    N = 32
    out = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.lanemask_ge(subgroup.invocation_id())

    foo()
    for i in range(N):
        expected = (~((1 << i) - 1)) & 0xFFFFFFFF
        assert out[i] == expected, f"lane {i}: got {out[i]:#x}, expected {expected:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_lanemask_explicit_lane_id():
    """Pass an explicit (non-current-lane) ``lane_id`` to verify the op generalises beyond the CUDA built-in form
    (which only takes the current lane).  Use a per-lane-varying expression so the kernel does not constant-fold."""
    N = 32
    out_lt = qd.field(dtype=qd.u32, shape=N)
    out_eq = qd.field(dtype=qd.u32, shape=N)
    out_ge = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            # `target = (lane * 3 + 1) % 32` is dynamically uniform across lanes only at call-site granularity, but the
            # primitive itself is portable across per-lane-varying inputs (it's pure arithmetic, no shuffle / ballot).
            target = (subgroup.invocation_id() * 3 + 1) % 32
            out_lt[i] = subgroup.lanemask_lt(target)
            out_eq[i] = subgroup.lanemask_eq(target)
            out_ge[i] = subgroup.lanemask_ge(target)

    foo()
    for i in range(N):
        target = (i * 3 + 1) % 32
        assert out_lt[i] == (1 << target) - 1, f"lt lane {i}: target={target}, got {out_lt[i]:#x}"
        assert out_eq[i] == (1 << target), f"eq lane {i}: target={target}, got {out_eq[i]:#x}"
        assert out_ge[i] == ((~((1 << target) - 1)) & 0xFFFFFFFF), f"ge lane {i}: target={target}, got {out_ge[i]:#x}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_lanemask_consistency():
    """Cross-check the family: ``lt | eq == le``, ``ge | lt == 0xFFFFFFFF``, ``gt & le == 0``, etc."""
    N = 32
    lt_f = qd.field(dtype=qd.u32, shape=N)
    le_f = qd.field(dtype=qd.u32, shape=N)
    eq_f = qd.field(dtype=qd.u32, shape=N)
    gt_f = qd.field(dtype=qd.u32, shape=N)
    ge_f = qd.field(dtype=qd.u32, shape=N)

    @qd.kernel
    def foo():
        qd.loop_config(block_dim=N)
        for i in range(N):
            lane = subgroup.invocation_id()
            lt_f[i] = subgroup.lanemask_lt(lane)
            le_f[i] = subgroup.lanemask_le(lane)
            eq_f[i] = subgroup.lanemask_eq(lane)
            gt_f[i] = subgroup.lanemask_gt(lane)
            ge_f[i] = subgroup.lanemask_ge(lane)

    foo()
    for i in range(N):
        lt, le, eq, gt, ge = lt_f[i], le_f[i], eq_f[i], gt_f[i], ge_f[i]
        assert (lt | eq) == le, f"lane {i}: lt|eq = {lt | eq:#x}, le = {le:#x}"
        assert (ge | lt) == 0xFFFFFFFF, f"lane {i}: ge|lt = {(ge | lt):#x}, expected 0xFFFFFFFF"
        assert (gt & le) == 0, f"lane {i}: gt&le = {gt & le:#x}, expected 0"
        assert (gt & eq) == 0, f"lane {i}: gt&eq = {gt & eq:#x}, expected 0"
        assert (lt & eq) == 0, f"lane {i}: lt&eq = {lt & eq:#x}, expected 0"
        assert (eq & ge) == eq, f"lane {i}: eq&ge = {eq & ge:#x}, expected {eq:#x}"


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
    Subgroup, 0)``).  We do not attempt to test reconvergence semantics here - that would require deliberately
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
    * Every elected lane has ``invocation_id() == 0``, and every non-elected lane has ``invocation_id() != 0`` - i.e.
      lane 0 is exactly the elected one.
    * The total elected count equals ``N / group_size()`` - one per subgroup.
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


def test_subgroup_ballot_full_subgroup_deprecation_warn_once(monkeypatch):
    """``subgroup.ballot_full_subgroup(predicate)`` is a deprecated alias for ``subgroup.ballot(predicate)`` after
    the rename to "tiled forms get a _tiled suffix; full-subgroup ops are unsuffixed".  Same one-warning-per-process
    contract as ``barrier()`` / ``memory_barrier()`` -- exactly one ``DeprecationWarning`` over many calls, and the
    predicate forwards through to ``ballot()`` unchanged."""
    import warnings as _w

    from quadrants.lang.simt import subgroup as sg

    sg._ballot_full_subgroup_deprecation_warned = False
    calls = []
    monkeypatch.setattr(sg, "ballot", lambda p: calls.append(p) or 0)

    with _w.catch_warnings(record=True) as records:
        _w.simplefilter("always", DeprecationWarning)
        sg.ballot_full_subgroup(1)
        sg.ballot_full_subgroup(0)
        sg.ballot_full_subgroup(1)

    deprecations = _drain_deprecation_warnings(records)
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    msg = str(deprecations[0].message)
    assert "qd.simt.subgroup.ballot_full_subgroup()" in msg
    assert "qd.simt.subgroup.ballot()" in msg
    assert calls == [1, 0, 1], calls


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
            assert result == reference, f"cycle {cycle}: subgroup IDs changed - " f"got {result}, expected {reference}"

        qd.reset()


# --------------------------------------------------------------------------------------------------------------------
# `subgroup.group_size()` / `subgroup.log2_group_size()` returning Python `int` and feeding into `qd.template()`
# --------------------------------------------------------------------------------------------------------------------


@test_utils.test(arch=qd.gpu)
def test_subgroup_group_size_returns_python_int():
    """``subgroup.group_size()`` returns a plain Python ``int`` callable from host scope, with the value matching the
    active backend's subgroup width (32 on CUDA / SPIR-V wave32 devices, 64 on AMDGPU).
    """
    sz = subgroup.group_size()
    assert isinstance(sz, int), f"expected int, got {type(sz).__name__}"
    assert sz in (32, 64), f"unexpected subgroup size {sz}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_log2_group_size_returns_python_int():
    """``log2_group_size()`` is ``int(log2(group_size()))`` with a power-of-two assert; values are 5 or 6."""
    l2 = subgroup.log2_group_size()
    assert isinstance(l2, int), f"expected int, got {type(l2).__name__}"
    assert (1 << l2) == subgroup.group_size(), f"log2_group_size {l2} doesn't match group_size {subgroup.group_size()}"
    assert l2 in (5, 6), f"unexpected log2_group_size {l2}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_group_size_folds_into_kernel_ir():
    """``subgroup.group_size()`` inside a kernel body folds to a constant literal - verified by storing it into a field
    and reading back the host-side value of ``group_size()``."""
    N = 64
    out = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.group_size()

    k()
    host_size = subgroup.group_size()
    for i in range(N):
        assert out[i] == host_size, f"lane {i}: kernel saw {out[i]}, host sees {host_size}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_group_size_stable_across_reinit(req_arch, req_options):
    """``group_size()`` / ``log2_group_size()`` return the same value after a ``qd.reset()`` + ``qd.init()`` on the
    same backend.  Regression guard: a stale-cached subgroup size on the ``Program`` could silently survive reset and
    corrupt kernels launched on the second init (e.g. if the SPIR-V probe path didn't re-run, or if the LLVM backends
    cached the wrong arch constant).  Three cycles to catch order-dependent caches.

    Takes ``req_arch`` / ``req_options`` from the fixture so the re-init mirrors what conftest's ``wanted_arch`` does
    for the first init (notably ``device_memory_GB`` and ``print_full_traceback``).  Don't read
    ``impl.current_cfg().arch`` after the first ``qd.reset()`` -- the live config goes away with the runtime."""
    sz_ref = subgroup.group_size()
    l2_ref = subgroup.log2_group_size()
    assert isinstance(sz_ref, int) and isinstance(l2_ref, int)
    assert (1 << l2_ref) == sz_ref

    init_options = dict(req_options or {})
    init_options.setdefault("print_full_traceback", True)
    init_options.setdefault("enable_fallback", False)

    for cycle in range(3):
        qd.reset()
        qd.init(arch=req_arch, **init_options)
        sz = subgroup.group_size()
        l2 = subgroup.log2_group_size()
        assert sz == sz_ref, f"cycle {cycle}: group_size changed across reinit: {sz_ref} -> {sz}"
        assert l2 == l2_ref, f"cycle {cycle}: log2_group_size changed across reinit: {l2_ref} -> {l2}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_log2_group_size_feeds_into_template():
    """``log2_group_size()`` can be passed directly into ``qd.template()`` arguments - e.g. ``reduce_add_tiled(v,
    log2_group_size())`` matches what hand-written ``log2_size=5/6`` would produce."""
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    out = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def fill_and_reduce():
        qd.loop_config(block_dim=N)
        for i in range(N):
            src[i] = 1
            out[i] = subgroup.reduce_add_tiled(src[i], subgroup.log2_group_size())

    fill_and_reduce()
    # Lane 0 of each subgroup gets the full sum: group_size on each.
    g = subgroup.group_size()
    for sg in range(N // g):
        assert out[sg * g] == g, f"subgroup {sg} lane 0: got {out[sg * g]}, expected {g}"


# --------------------------------------------------------------------------------------------------------------------
# Full-subgroup wrappers: smoke-test each full-subgroup op against its ``_tiled`` form with explicit
# ``log2_size=log2_group_size()``.  Picks N to be ``group_size()`` so the full-subgroup form is meaningful on both
# wave32 and wave64.
# --------------------------------------------------------------------------------------------------------------------


def _check_full_matches_tiled(full_fn, tiled_fn, *, dtype=qd.i32, host_init=None, atol=1e-5):
    """Run ``full_fn(v)`` and ``tiled_fn(v, log2_size=log2_group_size())`` over the active subgroup and confirm they
    produce identical results in every lane.  Both calls are inside the same kernel so the compile-time-resolved
    ``log2_group_size()`` lines up with whatever subgroup width Quadrants picked for the launch.

    ``dtype``: lane-value dtype (defaults to ``qd.i32``).  Float dtypes use a tolerance compare.
    ``host_init(src, n)``: optional host-side initializer.  If ``None``, fills ``src[i] = i + 1`` from inside the
    kernel (fine for add / min / max / scans where the running aggregate stays bounded for ``N <= 64``); pass a custom
    initializer for mul / and / or / xor where the per-step aggregate would otherwise overflow or collapse.
    """
    _skip_if_f64_unsupported(dtype)
    N = subgroup.group_size()
    src = qd.field(dtype=dtype, shape=N)
    out_full = qd.field(dtype=dtype, shape=N)
    out_base = qd.field(dtype=dtype, shape=N)
    l2 = subgroup.log2_group_size()

    @qd.kernel
    def fill_in_kernel():
        qd.loop_config(block_dim=N)
        for i in range(N):
            src[i] = i + 1
            out_full[i] = full_fn(src[i])
            out_base[i] = tiled_fn(src[i], l2)

    @qd.kernel
    def run_only():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out_full[i] = full_fn(src[i])
            out_base[i] = tiled_fn(src[i], l2)

    if host_init is None:
        fill_in_kernel()
    else:
        host_init(src, N)
        run_only()

    int_dtypes = (qd.i32, qd.i64, qd.u32, qd.u64)
    for i in range(N):
        if dtype in int_dtypes:
            assert out_full[i] == out_base[i], f"lane {i}: full={out_full[i]}, base={out_base[i]}"
        else:
            ref = float(out_base[i])
            got = float(out_full[i])
            assert abs(got - ref) <= atol * max(abs(ref), 1.0), f"lane {i}: full={got}, base={ref}"


def _init_full_small_int(src, n):
    """Bounded values for mul: most lanes 1, every fourth lane 2.  Caps the 64-way product at 2**16 (i32-safe)."""
    for i in range(n):
        src[i] = 2 if (i % 4 == 0) else 1


def _init_full_bitwise(src, n):
    """Per-lane single-bit pattern for bitwise scans: ``1 << (i % 7)``.  Non-zero on every lane so AND has signal, OR
    grows monotonically, XOR alternates."""
    for i in range(n):
        src[i] = 1 << (i % 7)


@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_add():
    _check_full_matches_tiled(subgroup.reduce_add, subgroup.reduce_add_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_all_add():
    _check_full_matches_tiled(subgroup.reduce_all_add, subgroup.reduce_all_add_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_min():
    _check_full_matches_tiled(subgroup.reduce_min, subgroup.reduce_min_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_max():
    _check_full_matches_tiled(subgroup.reduce_max, subgroup.reduce_max_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_all_min():
    _check_full_matches_tiled(subgroup.reduce_all_min, subgroup.reduce_all_min_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_all_max():
    _check_full_matches_tiled(subgroup.reduce_all_max, subgroup.reduce_all_max_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_add():
    _check_full_matches_tiled(subgroup.inclusive_add, subgroup.inclusive_add_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_min():
    _check_full_matches_tiled(subgroup.inclusive_min, subgroup.inclusive_min_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_max():
    _check_full_matches_tiled(subgroup.inclusive_max, subgroup.inclusive_max_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_mul():
    _check_full_matches_tiled(subgroup.inclusive_mul, subgroup.inclusive_mul_tiled, host_init=_init_full_small_int)


@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_and():
    _check_full_matches_tiled(subgroup.inclusive_and, subgroup.inclusive_and_tiled, host_init=_init_full_bitwise)


@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_or():
    _check_full_matches_tiled(subgroup.inclusive_or, subgroup.inclusive_or_tiled, host_init=_init_full_bitwise)


@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_xor():
    _check_full_matches_tiled(subgroup.inclusive_xor, subgroup.inclusive_xor_tiled, host_init=_init_full_bitwise)


@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_add():
    _check_full_matches_tiled(subgroup.exclusive_add, subgroup.exclusive_add_tiled)


@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_mul():
    _check_full_matches_tiled(subgroup.exclusive_mul, subgroup.exclusive_mul_tiled, host_init=_init_full_small_int)


@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_and():
    _check_full_matches_tiled(subgroup.exclusive_and, subgroup.exclusive_and_tiled, host_init=_init_full_bitwise)


@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_or():
    _check_full_matches_tiled(subgroup.exclusive_or, subgroup.exclusive_or_tiled, host_init=_init_full_bitwise)


@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_xor():
    _check_full_matches_tiled(subgroup.exclusive_xor, subgroup.exclusive_xor_tiled, host_init=_init_full_bitwise)


@test_utils.test(arch=qd.gpu)
def test_subgroup_all_equal():
    """All-equal case is 1 across the full subgroup; flipping one lane drops it to 0."""
    N = subgroup.group_size()
    src = qd.field(dtype=qd.i32, shape=N)
    out = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.all_equal(src[i])

    for i in range(N):
        src[i] = 7
    k()
    for i in range(N):
        assert out[i] == 1, f"all-equal case: lane {i} got {out[i]}"

    src[N // 3] = 8
    k()
    for i in range(N):
        assert out[i] == 0, f"one-differs case: lane {i} got {out[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_min():
    """``exclusive_min(value)`` matches ``exclusive_min_tiled(value, log2_group_size())`` lane-by-lane.  Lane 0 holds
    the dtype's auto-derived identity (``np.iinfo(qd.i32).max`` here)."""
    N = subgroup.group_size()
    src = qd.field(dtype=qd.i32, shape=N)
    out_full = qd.field(dtype=qd.i32, shape=N)
    out_base = qd.field(dtype=qd.i32, shape=N)
    l2 = subgroup.log2_group_size()

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out_full[i] = subgroup.exclusive_min(src[i])
            out_base[i] = subgroup.exclusive_min_tiled(src[i], l2)

    for i in range(N):
        src[i] = ((i * 7 + 11) % 23) + 1  # 11, 7, 13, 3, 17, 5, 19, ... -- varied, all < 24
    k()

    for i in range(N):
        assert out_full[i] == out_base[i], f"lane {i}: full={out_full[i]}, base={out_base[i]}"
    expected_identity = int(np.iinfo(np.int32).max)
    assert (
        out_full[0] == expected_identity
    ), f"lane 0 of first group should be dtype max ({expected_identity}), got {out_full[0]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_exclusive_max():
    """``exclusive_max(value)`` matches ``exclusive_max_tiled(value, log2_group_size())`` lane-by-lane.  Lane 0 holds
    the dtype's auto-derived identity (``np.iinfo(qd.i32).min`` here)."""
    N = subgroup.group_size()
    src = qd.field(dtype=qd.i32, shape=N)
    out_full = qd.field(dtype=qd.i32, shape=N)
    out_base = qd.field(dtype=qd.i32, shape=N)
    l2 = subgroup.log2_group_size()

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out_full[i] = subgroup.exclusive_max(src[i])
            out_base[i] = subgroup.exclusive_max_tiled(src[i], l2)

    for i in range(N):
        src[i] = ((i * 7 + 11) % 23) + 1
    k()

    for i in range(N):
        assert out_full[i] == out_base[i], f"lane {i}: full={out_full[i]}, base={out_base[i]}"
    expected_identity = int(np.iinfo(np.int32).min)
    assert (
        out_full[0] == expected_identity
    ), f"lane 0 of first group should be dtype min ({expected_identity}), got {out_full[0]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_min():
    """``segmented_reduce_min(value, head_flag)`` matches
    ``segmented_reduce_min_tiled(value, head_flag, log2_group_size())``."""
    N = subgroup.group_size()
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    out_full = qd.field(dtype=qd.i32, shape=N)
    out_base = qd.field(dtype=qd.i32, shape=N)
    l2 = subgroup.log2_group_size()

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out_full[i] = subgroup.segmented_reduce_min(src[i], head[i])
            out_base[i] = subgroup.segmented_reduce_min_tiled(src[i], head[i], l2)

    for i in range(N):
        src[i] = ((i * 11 + 5) % 31) + 1  # varied 1..31, non-monotonic
        head[i] = 1 if i % 5 == 0 else 0
    k()

    for i in range(N):
        assert out_full[i] == out_base[i], f"lane {i}: full={out_full[i]}, base={out_base[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_max():
    """``segmented_reduce_max(value, head_flag)`` matches
    ``segmented_reduce_max_tiled(value, head_flag, log2_group_size())``."""
    N = subgroup.group_size()
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    out_full = qd.field(dtype=qd.i32, shape=N)
    out_base = qd.field(dtype=qd.i32, shape=N)
    l2 = subgroup.log2_group_size()

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out_full[i] = subgroup.segmented_reduce_max(src[i], head[i])
            out_base[i] = subgroup.segmented_reduce_max_tiled(src[i], head[i], l2)

    for i in range(N):
        src[i] = ((i * 11 + 5) % 31) + 1
        head[i] = 1 if i % 5 == 0 else 0
    k()

    for i in range(N):
        assert out_full[i] == out_base[i], f"lane {i}: full={out_full[i]}, base={out_base[i]}"


# Float-dtype coverage of `_full` variants.  Wrappers are dtype-agnostic by construction (they just route to the base
# call with `log2_size = log2_group_size()`), so one f32 case per family is enough to catch a regression that would
# accidentally cast through i32 inside a wrapper.


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_add_float(dtype):
    _check_full_matches_tiled(subgroup.reduce_add, subgroup.reduce_add_tiled, dtype=dtype)


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_inclusive_add_float(dtype):
    _check_full_matches_tiled(subgroup.inclusive_add, subgroup.inclusive_add_tiled, dtype=dtype)


@pytest.mark.parametrize("dtype", [qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_add_float(dtype):
    """Float-dtype variant of `test_subgroup_segmented_reduce_add` -- inline because the head-flag-driven init doesn't
    fit the shared `_check_full_matches_tiled` template."""
    _skip_if_f64_unsupported(dtype)
    N = subgroup.group_size()
    src = qd.field(dtype=dtype, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    out_full = qd.field(dtype=dtype, shape=N)
    out_base = qd.field(dtype=dtype, shape=N)
    l2 = subgroup.log2_group_size()

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out_full[i] = subgroup.segmented_reduce_add(src[i], head[i])
            out_base[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], l2)

    for i in range(N):
        src[i] = (i + 1) * 1.5
        head[i] = 1 if i % 7 == 0 else 0
    k()

    for i in range(N):
        ref = float(out_base[i])
        got = float(out_full[i])
        assert abs(got - ref) <= 1e-4 * max(abs(ref), 1.0), f"lane {i}: full={got}, base={ref}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_all_true():
    """``all_true`` reduces across every lane in the subgroup.  Setting one lane false flips the result."""
    N = subgroup.group_size()
    flag = qd.field(dtype=qd.i32, shape=N)
    out = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.all_true(flag[i])

    for i in range(N):
        flag[i] = 1
    k()
    for i in range(N):
        assert out[i] == 1, f"all-true case: lane {i} got {out[i]}"

    flag[N // 2] = 0
    k()
    for i in range(N):
        assert out[i] == 0, f"one-false case: lane {i} got {out[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_any_true():
    """``any_true`` ORs across every lane: 0 only when every lane votes false."""
    N = subgroup.group_size()
    flag = qd.field(dtype=qd.i32, shape=N)
    out = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out[i] = subgroup.any_true(flag[i])

    for i in range(N):
        flag[i] = 0
    k()
    for i in range(N):
        assert out[i] == 0, f"all-false case: lane {i} got {out[i]}"

    flag[N // 2] = 1
    k()
    for i in range(N):
        assert out[i] == 1, f"one-true case: lane {i} got {out[i]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_segmented_reduce_add():
    """``segmented_reduce_add`` uses ``log2_size = log2_group_size()``; on wave32 backends that's 5 (covers the
    full warp), on AMDGPU wave64 it's 6 (covers the full wave).  Compares against the base call with the same
    ``log2_size``."""
    N = subgroup.group_size()
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    out_full = qd.field(dtype=qd.i32, shape=N)
    out_base = qd.field(dtype=qd.i32, shape=N)
    l2 = subgroup.log2_group_size()

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            out_full[i] = subgroup.segmented_reduce_add(src[i], head[i])
            out_base[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], l2)

    for i in range(N):
        src[i] = i + 1
        head[i] = 1 if i % 7 == 0 else 0
    k()

    for i in range(N):
        assert out_full[i] == out_base[i], f"lane {i}: full={out_full[i]}, base={out_base[i]}"


# --------------------------------------------------------------------------------------------------------------------
# `segmented_reduce_*` with `log2_size = 6` (wave64 / full AMDGPU subgroup).  On wave32 hardware this would violate
# `2**log2_size <= group_size()`, so the test is gated on AMDGPU.  Exercises the u64-bitmask branch of
# `_segment_head_distance_tiled`.
# --------------------------------------------------------------------------------------------------------------------


@test_utils.test(arch=qd.amdgpu)
def test_subgroup_segmented_reduce_add_tiled_log2_size_6():
    """``log2_size=6`` reduces across all 64 lanes of an AMDGPU wave64 wavefront.  Exercises the u64-bitmask path
    in ``_segment_head_distance_tiled``."""
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_add_tiled(src[i], head[i], 6)

    head_lanes = {0, 9, 23, 40, 57}
    for i in range(N):
        src[i] = i + 1
        head[i] = 1 if i in head_lanes else 0

    k()

    expected_py = _python_segmented_reduce(
        [i + 1 for i in range(N)],
        [1 if i in head_lanes else 0 for i in range(N)],
        64,
        lambda a, b: a + b,
    )
    for i in range(N):
        assert dst[i] == expected_py[i], f"lane {i}: got {dst[i]}, expected {expected_py[i]}"


@test_utils.test(arch=qd.amdgpu)
def test_subgroup_segmented_reduce_max_tiled_log2_size_6():
    """``log2_size=6`` ``segmented_reduce_max_tiled`` across a wave64 wavefront - same coverage as the ``_add``
    variant for the u64 path."""
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    head = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.segmented_reduce_max_tiled(src[i], head[i], 6)

    head_lanes = {0, 17, 50}
    for i in range(N):
        src[i] = (i * 37 + 11) % 101  # varied integer payload
        head[i] = 1 if i in head_lanes else 0

    k()

    expected_py = _python_segmented_reduce(
        [(i * 37 + 11) % 101 for i in range(N)],
        [1 if i in head_lanes else 0 for i in range(N)],
        64,
        max,
    )
    for i in range(N):
        assert dst[i] == expected_py[i], f"lane {i}: got {dst[i]}, expected {expected_py[i]}"


# --------------------------------------------------------------------------------------------------------------------
# Full-wave (``log2_size = 6``) absolute-correctness tests for the four shuffle-tree families.  Each is a thin
# wrapper around the existing per-family helper -- the helpers already loop over ``N // group_size`` groups, which
# at ``log2_size = 6`` and ``N = 64`` collapses to one group spanning the whole wave64 subgroup.  We deliberately
# don't expand the bulk ``[1, 2, 3, 4, 5]`` parameterization to include ``6``: the dtype-lowering paths are
# orthogonal to the unroll-depth path, so ``i32`` at ``log2_size = 6`` is enough to lock the cross-half shuffle
# step at offset 32 that the AMDGPU wave64 fix introduces.  The ``_full`` variant tests above already check the
# (matching) ``base(v, log2_group_size())`` shape for every op + dtype.  ``arch=qd.amdgpu`` is the only forced-wave64
# target today; the cross-half shuffle traffic these probe is implementation-defined on wave32 backends.
# --------------------------------------------------------------------------------------------------------------------


@test_utils.test(arch=qd.amdgpu)
def test_subgroup_reduce_add_tiled_log2_size_6():
    """``log2_size = 6`` ``reduce_add_tiled`` over a wave64 subgroup.  Exercises the cross-half ``shuffle_down(v, 32)``
    step of the tree, which on RDNA pre-fix silently dropped the upper-half contribution."""
    _check_reduce_lane0(subgroup.reduce_add_tiled, lambda a, b: a + b, qd.i32, 6, _init_field)


@test_utils.test(arch=qd.amdgpu)
def test_subgroup_reduce_all_add_tiled_log2_size_6():
    """``log2_size = 6`` ``reduce_all_add_tiled`` over a wave64 subgroup.  Butterfly uses ``shuffle_xor(v, 32)`` at the
    final step, which on RDNA pre-fix wrapped within SIMD32 and broadcast the wrong value to every lane."""
    _check_reduce_all(subgroup.reduce_all_add_tiled, lambda a, b: a + b, qd.i32, 6, _init_field)


@test_utils.test(arch=qd.amdgpu)
def test_subgroup_inclusive_add_tiled_log2_size_6():
    """``log2_size = 6`` ``inclusive_add_tiled`` over a wave64 subgroup.  Final Hillis-Steele step does
    ``shuffle_up(v, 32)``, exercising the cross-half path."""
    _check_inclusive_scan(subgroup.inclusive_add_tiled, lambda a, b: a + b, qd.i32, 6, _init_field)


@test_utils.test(arch=qd.amdgpu)
def test_subgroup_exclusive_add_tiled_log2_size_6():
    """``log2_size = 6`` ``exclusive_add_tiled`` over a wave64 subgroup.  Same shuffle_up tree as
    ``inclusive_add_tiled`` plus an extra ``shuffle_up`` to shift the result down by one lane and an
    identity-at-lane-0 substitution."""
    _check_exclusive_scan(subgroup.exclusive_add_tiled, lambda a, b: a + b, 0, qd.i32, 6, _init_field)


# --------------------------------------------------------------------------------------------------------------------
# Direct cross-half shuffle coverage.  These tests target the lane <-> lane >= 32 traffic that on AMD RDNA wave64
# hardware (gfx10+) used to silently wrap inside the 32-lane SIMD cluster: ``ds_bpermute`` is SIMD32-scoped, and prior
# to the ``permlane64``-based cross-half helper a lane in the bottom half could not read the top half (and vice
# versa).  All five tests are gated to ``log2_group_size() == 6`` so they only assert anything on real wave64
# hardware -- CUDA and SPIR-V backends with wave32 skip the absolute-correctness check (the cross-half partner is
# out of range there, which is implementation-defined).  CDNA (gfx9xx, MI300X) already had a wave64-wide
# ``ds_bpermute`` so its behaviour is unchanged by the fix; the new helper is observably a no-op on that path.
# --------------------------------------------------------------------------------------------------------------------


def _skip_unless_wave64():
    if subgroup.group_size() != 64:
        pytest.skip(f"requires wave64 subgroup; running on subgroup_size={subgroup.group_size()}")


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_xor_cross_half(dtype):
    """``shuffle_xor(v, 32)`` swaps the two halves of a wave64 subgroup.  Pre-fix, on AMD RDNA gfx10+ this would
    wrap within each SIMD32 and return ``v`` unchanged for every lane.  Gated on wave64."""
    _skip_if_f64_unsupported(dtype)
    _skip_unless_wave64()
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.shuffle_xor(src[i], qd.u32(32))

    _init_field(src, N, dtype)

    k()

    for i in range(N):
        partner = i ^ 32
        assert (
            dst[i] == src[partner]
        ), f"lane {i}: shuffle_xor(v, 32) returned src[{i}]={src[i]} but expected src[{partner}]={src[partner]}"


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_down_offset_32(dtype):
    """``shuffle_down(v, 32)``: each lane in [0, 32) reads from ``lane + 32``.  Lanes in [32, 64) read out-of-range
    (implementation-defined, not asserted).  Pre-fix this returned garbage on RDNA wave64."""
    _skip_if_f64_unsupported(dtype)
    _skip_unless_wave64()
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.shuffle_down(src[i], qd.u32(32))

    _init_field(src, N, dtype)

    k()

    for i in range(32):
        assert (
            dst[i] == src[i + 32]
        ), f"lane {i}: shuffle_down(v, 32) returned {dst[i]} but expected src[{i + 32}]={src[i + 32]}"


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_up_offset_32(dtype):
    """``shuffle_up(v, 32)``: each lane in [32, 64) reads from ``lane - 32``.  Lanes in [0, 32) read out-of-range
    (implementation-defined).  Pre-fix this returned garbage on RDNA wave64."""
    _skip_if_f64_unsupported(dtype)
    _skip_unless_wave64()
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.shuffle_up(src[i], qd.u32(32))

    _init_field(src, N, dtype)

    k()

    for i in range(32, 64):
        assert (
            dst[i] == src[i - 32]
        ), f"lane {i}: shuffle_up(v, 32) returned {dst[i]} but expected src[{i - 32}]={src[i - 32]}"


@pytest.mark.parametrize("dtype", [qd.i32, qd.f32, qd.f64])
@test_utils.test(arch=qd.gpu)
def test_subgroup_shuffle_absolute_lane_high_half(dtype):
    """``shuffle(v, lane >= 32)``: each lane in the bottom half reads from a fixed lane in the top half.  Pre-fix
    this was the canonical cross-half failure mode on RDNA wave64."""
    _skip_if_f64_unsupported(dtype)
    _skip_unless_wave64()
    N = 64
    src = qd.field(dtype=dtype, shape=N)
    dst_lo = qd.field(dtype=dtype, shape=N)
    dst_hi = qd.field(dtype=dtype, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            # Bottom half reads lane 47; top half reads lane 7.  Both directions exercise the cross-half path.
            dst_lo[i] = subgroup.shuffle(src[i], qd.u32(47))
            dst_hi[i] = subgroup.shuffle(src[i], qd.u32(7))

    _init_field(src, N, dtype)

    k()

    for i in range(N):
        assert dst_lo[i] == src[47], f"lane {i}: shuffle(v, 47) returned {dst_lo[i]} but expected src[47]={src[47]}"
        assert dst_hi[i] == src[7], f"lane {i}: shuffle(v, 7) returned {dst_hi[i]} but expected src[7]={src[7]}"


@test_utils.test(arch=qd.gpu)
def test_subgroup_reduce_add_absolute():
    """End-to-end correctness: ``reduce_add`` over a wave64 subgroup must equal the Python sum.  Pre-fix the
    top-half lanes never contributed on RDNA, so this returned half the expected value (or worse)."""
    _skip_unless_wave64()
    N = 64
    src = qd.field(dtype=qd.i32, shape=N)
    dst = qd.field(dtype=qd.i32, shape=N)

    @qd.kernel
    def k():
        qd.loop_config(block_dim=N)
        for i in range(N):
            dst[i] = subgroup.reduce_add(src[i])

    payload = [i * 3 + 1 for i in range(N)]
    for i in range(N):
        src[i] = payload[i]
    expected = sum(payload)

    k()

    # ``reduce_add_tiled`` returns the full sum in lane 0 of each tile; for log2_size = log2_group_size() the tile
    # is the whole subgroup, so we only check lane 0.  (The other lanes' values are implementation-defined.)
    assert dst[0] == expected, f"reduce_add lane 0: got {dst[0]}, expected {expected}"
