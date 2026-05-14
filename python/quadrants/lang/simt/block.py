# type: ignore
# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false

import warnings

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang import ops as _ops
from quadrants.lang.expr import make_expr_group
from quadrants.lang.kernel_impl import func as _func

# Import order matters: ``subgroup`` must come before ``reductions``.  ``reductions.py`` does ``from
# quadrants.lang.simt.subgroup import (ballot, invocation_id, ...)`` at its top, and ``subgroup.py`` does ``from
# quadrants.lang.simt.reductions import *`` at its bottom.  If ``reductions`` is imported here first, it triggers a
# circular load that leaves ``subgroup``'s wildcard re-export running while ``reductions.__all__`` isn't yet defined,
# so ``subgroup.reduce_add_tiled`` etc. silently end up missing.  Importing ``subgroup`` first (which then drives
# ``reductions`` to completion via the wildcard) keeps the fully-loaded layout downstream callers expect.  ``isort:
# skip_file`` would disable sorting for the whole file; the local ``noqa`` comments below scope the override to just
# these two lines.
from quadrants.lang.simt import subgroup as _subgroup  # noqa: I001  isort: skip
from quadrants.lang.simt import reductions as _reductions  # noqa: I001
from quadrants.lang.simt.reductions import _bin_add, _bin_max, _bin_min
from quadrants.lang.util import quadrants_scope
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32 as _i32
from quadrants.types.primitive_types import i64 as _i64
from quadrants.types.primitive_types import u32 as _u32
from quadrants.types.primitive_types import u64 as _u64


def arch_uses_spv(arch):
    return arch == _qd_core.vulkan or arch == _qd_core.metal


def sync():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupBarrier", with_runtime_context=False)
    raise ValueError(f"qd.block.shared_array is not supported for arch {arch}")


def sync_all_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        # Hardware-fused barrier+reduction on NVPTX (`barrier.cta.red.and.aligned.all.sync`).
        return impl.call_internal("block_barrier_and_i32", predicate, with_runtime_context=False)
    if arch == _qd_core.amdgpu or arch_uses_spv(arch):
        # AMDGPU and SPIR-V (Vulkan / Metal) emulate via shared memory + 2 barriers + an atomic; see
        # `_block_reduce_*_emulated` below for the pattern.
        return _block_reduce_all_nonzero_emulated(predicate)
    raise ValueError(f"qd.block.sync_all_nonzero is not supported for arch {arch}")


def sync_any_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_or_i32", predicate, with_runtime_context=False)
    if arch == _qd_core.amdgpu or arch_uses_spv(arch):
        return _block_reduce_any_nonzero_emulated(predicate)
    raise ValueError(f"qd.block.sync_any_nonzero is not supported for arch {arch}")


def sync_count_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_count_i32", predicate, with_runtime_context=False)
    if arch == _qd_core.amdgpu or arch_uses_spv(arch):
        return _block_reduce_count_nonzero_emulated(predicate)
    raise ValueError(f"qd.block.sync_count_nonzero is not supported for arch {arch}")


def mem_fence():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("block_mem_fence", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupMemoryBarrier", with_runtime_context=False)
    raise ValueError(f"qd.block.mem_fence is not supported for arch {arch}")


def mem_sync():
    warnings.warn(
        "qd.simt.block.mem_sync() is deprecated; use qd.simt.block.mem_fence() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mem_fence()


def thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("block_thread_idx", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("localInvocationId", with_runtime_context=False)
    raise ValueError(f"qd.block.thread_idx is not supported for arch {arch}")


def global_thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.get_runtime().compiling_callable.ast_builder().insert_thread_idx_expr()
    if arch_uses_spv(arch):
        return impl.call_internal("globalInvocationId", with_runtime_context=False)
    raise ValueError(f"qd.block.global_thread_idx is not supported for arch {arch}")


class SharedArray:
    _is_quadrants_class = True

    def __init__(self, shape, dtype):
        if isinstance(shape, int):
            self.shape = (shape,)
        elif (isinstance(shape, tuple) or isinstance(shape, list)) and all(isinstance(s, int) for s in shape):
            self.shape = shape
        else:
            raise ValueError(
                f"qd.simt.block.shared_array shape must be an integer or a tuple of integers, but got {shape}"
            )
        if isinstance(dtype, impl.MatrixType):
            dtype = dtype.tensor_type
        self.dtype = dtype
        self.shared_array_proxy = impl.expr_init_shared_array(self.shape, dtype)

    @quadrants_scope
    def subscript(self, *indices):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()
        return impl.Expr(
            ast_builder.expr_subscript(
                self.shared_array_proxy,
                make_expr_group(*indices),
                _qd_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
        )


# --- Block reductions ------------------------------------------------------------------
#
# Two-stage block reduce: each subgroup reduces its lanes via `shuffle_down`, lane 0 of every subgroup publishes the
# subgroup aggregate to shared memory, a `block.sync()` retires the publish, and thread 0 sequentially folds the
# subgroup aggregates with `op`.  Cost: `log2(subgroup_size)` shuffles + 1 shared-mem write/read per subgroup + 1
# `block.sync` + (NUM_SUBGROUPS - 1) ops on thread 0.  The subgroup size is read from `subgroup.group_size()` (a
# compile-time Python int) at the top of every block op, so callers never plumb it in.
#
# The per-subgroup step delegates to `reductions._reduce_tiled`, the generic-op private helper (alongside
# `reductions._inclusive_scan_tiled` / `_exclusive_scan_tiled`) that mirrors `subgroup.reduce_add_tiled` / `_min_tiled`
# / `_max_tiled` but takes a caller-supplied template operator -- so the same block skeleton covers add / min / max /
# mul / bitwise / custom monoids.


@_func
def reduce(value, block_dim: template(), op: template(), dtype: template()):
    """Block-scope reduction under a generic associative ``op``.  Result is valid in **thread 0 only**; other threads
    retain partial values.  Use `reduce_all` if you need the result on every thread.

    Args:
        value: per-thread input.
        block_dim: threads per block (template).  Must be a positive multiple of ``subgroup.group_size()`` (32 on CUDA
            / Metal / Vulkan-on-NVIDIA, 64 on AMDGPU).
        op: ``@qd.func`` taking two values and returning the same type as ``value``; callers can plug in custom
            associative monoids (bitwise ops, multiplicative, matrix-multiply, etc.) without re-implementing the
            per-subgroup + shared-mem skeleton.  See `reduce_add` for the standard sum specialization.
        dtype: scalar dtype for the inter-subgroup shared-memory staging slot (must match ``value``'s type).

    The calling thread's block-local index is read internally via `block.thread_idx()`; the subgroup size is read from
    `subgroup.group_size()` at compile time.  When the block is exactly one subgroup the shared-memory path is
    short-circuited at compile time and the call costs only the per-subgroup tree.
    """
    SUBGROUP_SIZE = impl.static(_subgroup.group_size())
    log2_subgroup = impl.static(_subgroup.log2_group_size())
    impl.static_assert(
        impl.static(block_dim % SUBGROUP_SIZE == 0 and block_dim >= SUBGROUP_SIZE),
        "block.reduce: block_dim must be a positive multiple of subgroup size",
    )
    NUM_SUBGROUPS = impl.static(block_dim // SUBGROUP_SIZE)

    subgroup_agg = _reductions._reduce_tiled(value, op, log2_subgroup)

    if impl.static(NUM_SUBGROUPS == 1):
        return subgroup_agg

    tid = thread_idx()
    subgroup_id = tid // SUBGROUP_SIZE
    lane_id = tid & impl.static(SUBGROUP_SIZE - 1)

    shared = SharedArray(impl.static((NUM_SUBGROUPS,)), dtype)
    if lane_id == 0:
        shared[subgroup_id] = subgroup_agg
    sync()

    result = subgroup_agg
    if tid == 0:
        result = shared[0]
        for w in impl.static(range(1, NUM_SUBGROUPS)):
            result = op(result, shared[impl.static(w)])
    return result


@_func
def reduce_all(value, block_dim: template(), op: template(), dtype: template()):
    """Block-scope reduction under a generic associative ``op``, broadcast to every thread.  Costs one extra
    ``block.sync()`` plus a one-slot shared-memory broadcast vs. `reduce`.  See `reduce` for the operator contract.
    """
    result = reduce(value, block_dim, op, dtype)
    bcast = SharedArray((1,), dtype)
    if thread_idx() == 0:
        bcast[0] = result
    sync()
    return bcast[0]


@_func
def reduce_add(value, block_dim: template(), dtype: template()):
    """Block-scope sum reduction.  Result valid in **thread 0 only**.  See `reduce` for the argument contract."""
    return reduce(value, block_dim, _bin_add, dtype)


@_func
def reduce_min(value, block_dim: template(), dtype: template()):
    """Block-scope min reduction.  Result valid in **thread 0 only**.  See `reduce` for the argument contract."""
    return reduce(value, block_dim, _bin_min, dtype)


@_func
def reduce_max(value, block_dim: template(), dtype: template()):
    """Block-scope max reduction.  Result valid in **thread 0 only**.  See `reduce` for the argument contract."""
    return reduce(value, block_dim, _bin_max, dtype)


@_func
def reduce_all_add(value, block_dim: template(), dtype: template()):
    """Block-scope sum reduction with the result broadcast to every thread.  See `reduce_add` for the cheaper
    thread-0-only variant and `reduce` for the argument contract.
    """
    return reduce_all(value, block_dim, _bin_add, dtype)


@_func
def reduce_all_min(value, block_dim: template(), dtype: template()):
    """Block-scope min reduction broadcast to every thread.  See `reduce_all_add`."""
    return reduce_all(value, block_dim, _bin_min, dtype)


@_func
def reduce_all_max(value, block_dim: template(), dtype: template()):
    """Block-scope max reduction broadcast to every thread.  See `reduce_all_add`."""
    return reduce_all(value, block_dim, _bin_max, dtype)


# --- Block scans -----------------------------------------------------------------------
#
# Two-stage block scan.  Each subgroup does a Hillis-Steele scan via `reductions.{_inclusive_scan_tiled,
# _exclusive_scan_tiled}`, the last lane of every subgroup publishes the subgroup aggregate to shared memory, then
# every thread sequentially folds the subgroup prefixes and applies its own subgroup's prefix to its scan value.
# All threads receive a valid result; cost: one subgroup scan + 1 shared-mem write/read per subgroup + 1
# `block.sync()` + (NUM_SUBGROUPS - 1) ops on every thread (the cross-subgroup prefix is computed redundantly to
# avoid a second barrier).
#
# Inclusive: subgroup aggregate at the last lane is just the inclusive value, written directly.  Exclusive: subgroup
# aggregate = `op(exclusive[last_lane], value[last_lane])`, since the exclusive scan does not include the last lane's
# input - we recover the inclusive total with one extra `op` on the publish path.


@_func
def inclusive_scan(value, block_dim: template(), op: template(), dtype: template()):
    """Block-scope inclusive scan under a generic associative ``op``.  Every thread receives a valid result.

    Args:
        value: per-thread input.
        block_dim: threads per block (template).  Must be a positive multiple of ``subgroup.group_size()`` (32 on CUDA
            / Metal / Vulkan-on-NVIDIA, 64 on AMDGPU).
        op: ``@qd.func`` taking two values and returning the same type as ``value``; callers can plug in custom
            associative monoids without re-implementing the per-subgroup + shared-mem skeleton.  See `inclusive_add`
            for the standard sum specialization.
        dtype: scalar dtype for the inter-subgroup shared-memory staging slot; must match ``value``'s type.

    The calling thread's block-local index is read internally via `block.thread_idx()`; the subgroup size is read from
    `subgroup.group_size()` at compile time.  When the block is exactly one subgroup the cross-subgroup shared-memory
    path is short-circuited at compile time and the call costs only the per-subgroup Hillis-Steele tree.
    """
    SUBGROUP_SIZE = impl.static(_subgroup.group_size())
    log2_subgroup = impl.static(_subgroup.log2_group_size())
    impl.static_assert(
        impl.static(block_dim % SUBGROUP_SIZE == 0 and block_dim >= SUBGROUP_SIZE),
        "block.inclusive_scan: block_dim must be a positive multiple of subgroup size",
    )
    NUM_SUBGROUPS = impl.static(block_dim // SUBGROUP_SIZE)

    inclusive = _reductions._inclusive_scan_tiled(value, op, log2_subgroup)

    if impl.static(NUM_SUBGROUPS == 1):
        return inclusive

    tid = thread_idx()
    subgroup_id = tid // SUBGROUP_SIZE
    lane_id = tid & impl.static(SUBGROUP_SIZE - 1)

    shared = SharedArray(impl.static((NUM_SUBGROUPS,)), dtype)
    if lane_id == impl.static(SUBGROUP_SIZE - 1):
        shared[subgroup_id] = inclusive
    sync()

    # Sequential exclusive prefix scan over subgroup aggregates; each thread captures its own subgroup's prefix.
    # Subgroup 0's prefix is unused (its inclusive value is already the prefix sum from the start of the block), so
    # we never read `subgroup_prefix` on subgroup 0; the placeholder there exists only to give the variable a
    # definite type.
    block_aggregate = shared[0]
    subgroup_prefix = block_aggregate
    for w in impl.static(range(1, NUM_SUBGROUPS)):
        if subgroup_id == impl.static(w):
            subgroup_prefix = block_aggregate
        addend = shared[impl.static(w)]
        block_aggregate = op(block_aggregate, addend)

    if subgroup_id != 0:
        inclusive = op(subgroup_prefix, inclusive)
    return inclusive


@_func
def exclusive_scan(value, block_dim: template(), op: template(), identity, dtype: template()):
    """Block-scope exclusive scan under a generic associative ``op`` with explicit ``identity``.  Every thread receives
    a valid result; thread 0 holds ``identity`` and thread ``i > 0`` holds ``op(v[0], ..., v[i-1])``.

    See `inclusive_scan` for the per-arg contract; in addition this op takes an explicit ``identity`` because exclusive
    scan needs a definite value for thread 0 (and for the sentinel paths in `exclusive_min` / `exclusive_max`).  See
    `exclusive_add` for the additive specialization which derives a zero identity automatically.
    """
    SUBGROUP_SIZE = impl.static(_subgroup.group_size())
    log2_subgroup = impl.static(_subgroup.log2_group_size())
    impl.static_assert(
        impl.static(block_dim % SUBGROUP_SIZE == 0 and block_dim >= SUBGROUP_SIZE),
        "block.exclusive_scan: block_dim must be a positive multiple of subgroup size",
    )
    NUM_SUBGROUPS = impl.static(block_dim // SUBGROUP_SIZE)

    exclusive = _reductions._exclusive_scan_tiled(value, op, identity, log2_subgroup)

    if impl.static(NUM_SUBGROUPS == 1):
        return exclusive

    tid = thread_idx()
    subgroup_id = tid // SUBGROUP_SIZE
    lane_id = tid & impl.static(SUBGROUP_SIZE - 1)

    shared = SharedArray(impl.static((NUM_SUBGROUPS,)), dtype)
    if lane_id == impl.static(SUBGROUP_SIZE - 1):
        # Subgroup aggregate = inclusive at last lane = exclusive[last] + value[last] under `op`.
        shared[subgroup_id] = op(exclusive, value)
    sync()

    block_aggregate = shared[0]
    subgroup_prefix = (
        identity  # subgroup 0's prefix is the identity; subsequent subgroups overwrite this in their own iteration
    )
    for w in impl.static(range(1, NUM_SUBGROUPS)):
        if subgroup_id == impl.static(w):
            subgroup_prefix = block_aggregate
        addend = shared[impl.static(w)]
        block_aggregate = op(block_aggregate, addend)

    if subgroup_id != 0:
        exclusive = op(subgroup_prefix, exclusive)
    return exclusive


@_func
def inclusive_add(value, block_dim: template(), dtype: template()):
    """Block-scope inclusive prefix sum.  After the call, thread ``i`` holds ``v[0] + v[1] + ... + v[i]``.  See
    `inclusive_scan` for the argument contract.
    """
    return inclusive_scan(value, block_dim, _bin_add, dtype)


@_func
def inclusive_min(value, block_dim: template(), dtype: template()):
    """Block-scope inclusive prefix min.  See `inclusive_scan` for the argument contract."""
    return inclusive_scan(value, block_dim, _bin_min, dtype)


@_func
def inclusive_max(value, block_dim: template(), dtype: template()):
    """Block-scope inclusive prefix max.  See `inclusive_scan` for the argument contract."""
    return inclusive_scan(value, block_dim, _bin_max, dtype)


@_func
def exclusive_add(value, block_dim: template(), dtype: template()):
    """Block-scope exclusive prefix sum.  After the call, thread ``i > 0`` holds ``v[0] + v[1] + ... + v[i-1]`` and
    thread 0 holds the additive identity (zero, in ``value``'s dtype, derived as ``value - value``).  See
    `exclusive_scan` for the argument contract.
    """
    return exclusive_scan(value, block_dim, _bin_add, value - value, dtype)


# Plain Python wrappers (not ``@func``): the identity for an exclusive min / max scan is uniquely determined by
# ``value``'s dtype, so we introspect it at compile time and emit a typed-constant identity Expr rather than asking
# callers to provide one.  Mirrors the subgroup convention (``subgroup.exclusive_min`` and friends).  The identity
# helpers (``_typed_min_identity`` / ``_typed_max_identity``) are reused from ``reductions.py`` so the per-dtype
# sentinel choices stay consistent across the two scopes.
def exclusive_min(value, block_dim: template(), dtype: template()):
    """Block-scope exclusive prefix min.  After the call, thread ``i > 0`` holds ``min(v[0], ..., v[i-1])`` and
    thread 0 holds the dtype-derived identity: ``+inf`` for real dtypes, ``np.iinfo(dtype).max`` for integer dtypes
    (``UINT_MAX`` for unsigned, ``INT_MAX`` for signed).  See `exclusive_scan` for the underlying contract.
    """
    return exclusive_scan(value, block_dim, _bin_min, _reductions._typed_min_identity(value), dtype)


def exclusive_max(value, block_dim: template(), dtype: template()):
    """Block-scope exclusive prefix max.  After the call, thread ``i > 0`` holds ``max(v[0], ..., v[i-1])`` and
    thread 0 holds the dtype-derived identity: ``-inf`` for real dtypes, ``np.iinfo(dtype).min`` for signed integer
    dtypes, ``0`` for unsigned and bool.  See `exclusive_scan` for the underlying contract.
    """
    return exclusive_scan(value, block_dim, _bin_max, _reductions._typed_max_identity(value), dtype)


# --- Block radix rank ------------------------------------------------------------------
#
# Block-level radix ranking via the atomic-OR match-and-count strategy.  Each thread holds a single ``u32`` key; the
# function returns the key's stable rank within the block under the digit `(key >> bit_start) & ((1 << num_bits) - 1)`,
# and writes the per-digit count and exclusive-prefix arrays to caller-supplied shared-memory outparams.
#
# The algorithm runs in six steps:
#
# 1. ComputeHistogramsSubgroup: each subgroup builds a private digit histogram in shared memory via ``atomic_add``.
# 2. ComputeOffsetsSubgroupUpsweep: every thread sums per-subgroup histograms column-wise to produce a block-wide
#    bin count for digit ``= tid``, while rewriting the subgroup histogram entries into per-subgroup running
#    exclusive prefixes.
# 3. ExclusiveSum on the per-thread bin counts — uses the block exclusive scan defined above.
# 4. ComputeOffsetsSubgroupDownsweep: add the block-wide exclusive prefix into every subgroup's offset entry.
# 5. ComputeRanksItem (atomic-OR match): per-subgroup match via ``atomic_or`` on a per-digit lane-mask, then leader
#    (highest set lane) does a single ``atomic_add`` on the subgroup offset and broadcasts via ``subgroup.shuffle``;
#    each thread's rank is ``subgroup_offset + popc(bin_mask & lanemask_le) - 1``.
# 6. Write bin count + exclusive prefix to the outparam shared arrays.
#
# Shared-memory layout (all i32, total ``2 * BLOCK_SUBGROUPS * RADIX_DIGITS`` ints, 4096 ints = 16 KiB at the default
# 8-subgroup / 256-digit configuration):
#
#     subgroup_offsets / subgroup_histograms : [0, BLOCK_SUBGROUPS * RADIX_DIGITS)        (union backing)
#     match_masks                    : [BLOCK_SUBGROUPS * RADIX_DIGITS, 2 * ...)
#
# Subgroup-scope barriers use ``subgroup.sync()`` (lowers to ``__syncwarp`` on CUDA,
# ``OpControlBarrier(ScopeSubgroup, ...)`` on SPIR-V, ``s_barrier`` on AMDGPU).  ``LaneMaskLe()`` (the PTX intrinsic
# that gives a lane its less-than-or-equal lane mask) is replaced by ``subgroup.lanemask_le(lane)`` from the portable
# subgroup primitives.


@_func
def _subgroup_sync_fence():
    """Subgroup-scope barrier + memory fence — CUDA ``__syncwarp`` semantics across every backend.

    Why both ops: on CUDA, `subgroup.sync()` already lowers to `__syncwarp` which folds in a memory fence, so the
    extra `subgroup.mem_fence()` is redundant (a `__threadfence_block`).  On SPIR-V, however, the codegen emits
    `subgroupBarrier` as `OpControlBarrier(ScopeSubgroup, ScopeSubgroup, 0)` - i.e. with **no** memory semantics -
    so a bare `subgroup.sync()` does *not* publish prior shared-memory writes to other lanes.  The radix rank algorithm
    relies on the `__syncwarp` invariant that, after the barrier, every lane sees every other lane's prior
    `atomic_or` / `atomic_add` to shared memory; pairing the barrier with `subgroup.mem_fence()` (which emits a real
    `OpMemoryBarrier(ScopeSubgroup, AcquireRelease | UniformMemory | WorkgroupMemory)`) restores that invariant.
    """
    _subgroup.sync()
    _subgroup.mem_fence()


@_func
def _radix_rank_match_atomic_or_wave32(
    key,
    block_dim: template(),
    radix_bits: template(),
    bit_start: template(),
    num_bits: template(),
    bins,
    excl_prefix,
):
    """Wave32 implementation of `radix_rank_match_atomic_or`. See the public wrapper for the contract.

    Match-mask region is ``i32``; atomic_or, ballot, clz, popcnt all operate on 32 bits.  This path is taken on CUDA,
    Vulkan-on-NVIDIA, and Metal — none of which require ``i64`` threadgroup atomics.
    """
    SUBGROUP_THREADS = impl.static(_subgroup.group_size())
    RADIX_DIGITS = impl.static(1 << radix_bits)
    BLOCK_SUBGROUPS = impl.static(block_dim // SUBGROUP_THREADS)
    NUM_BITS_MASK = impl.static((1 << num_bits) - 1)
    BINS_PER_LANE = impl.static(RADIX_DIGITS // SUBGROUP_THREADS)

    # ``smem_offsets`` (i32) backs the per-subgroup histograms (step 1), in-place column-sum upsweep (step 2), folded
    # prefixes (step 4), and the leader's atomic_add slot (step 5).  ``smem_match`` (i32) backs the per-digit ballot
    # mask in step 5.  These were previously unioned into a single ``i32`` SharedArray; splitting them keeps the
    # offsets path independent of the match-mask width so the wave64 sibling can pick ``i64`` for its match region.
    smem_offsets = SharedArray(impl.static((BLOCK_SUBGROUPS * RADIX_DIGITS,)), _i32)
    smem_match = SharedArray(impl.static((BLOCK_SUBGROUPS * RADIX_DIGITS,)), _i32)

    tid = thread_idx()
    subgroup_idx = tid // SUBGROUP_THREADS
    lane = _ops.cast(_subgroup.invocation_id(), _i32)

    # Step 1: zero per-subgroup histograms and match_masks.
    for b in impl.static(range(BINS_PER_LANE)):
        bin_idx = lane + impl.static(b * SUBGROUP_THREADS)
        smem_offsets[subgroup_idx * RADIX_DIGITS + bin_idx] = _i32(0)
        smem_match[subgroup_idx * RADIX_DIGITS + bin_idx] = _i32(0)
    _subgroup_sync_fence()

    # Each thread atomic-adds 1 to its subgroup's bin for ``digit``.
    digit = _ops.cast(_ops.bit_and(_ops.bit_shr(key, _u32(bit_start)), _u32(NUM_BITS_MASK)), _i32)
    _ops.atomic_add(smem_offsets[subgroup_idx * RADIX_DIGITS + digit], _i32(1))

    sync()  # Publish per-subgroup histograms before column-sum.

    # Step 2: per-thread column sum across subgroups for digit == tid.  Each thread collects the running exclusive
    # prefix into ``bin_count`` while overwriting the subgroup histogram entries with their per-subgroup exclusive
    # prefix.  After the loop, ``bin_count`` is the block-wide total for digit == tid.
    bin_count = _i32(0)
    for j_subgroup in impl.static(range(BLOCK_SUBGROUPS)):
        subgroup_count = smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid]
        smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid] = bin_count
        bin_count = bin_count + subgroup_count

    # Step 3: block-wide exclusive sum on the per-thread bin counts.
    exclusive_digit_prefix = exclusive_add(bin_count, block_dim, _i32)

    # Step 4: ComputeOffsetsSubgroupDownsweep — fold the block-wide exclusive prefix into every subgroup's offset.
    for j_subgroup in impl.static(range(BLOCK_SUBGROUPS)):
        smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid] = (
            smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid] + exclusive_digit_prefix
        )

    sync()  # Publish subgroup offsets before the per-key match phase.

    # Step 5: per-key atomic-OR match.  ``items_per_thread == 1``, so this runs once per thread.
    lane_mask = _i32(1) << lane
    lane_mask_le_v = _subgroup.lanemask_le(_subgroup.invocation_id())

    match_idx = subgroup_idx * RADIX_DIGITS + digit

    # Every thread ORs its lane_mask into the per-digit match mask of its subgroup.  Threads with the same digit collide
    # on the same shared-memory cell and produce a bitmask of "lanes in this subgroup that share this digit".
    _ops.atomic_or(smem_match[match_idx], lane_mask)
    _subgroup_sync_fence()

    # Read the bin_mask back and find the leader (highest matching lane) + intra-subgroup rank.  ``clz`` here MUST
    # run on the u32 (FindUMsb on SPIR-V): casting to i32 first triggers SPIR-V's FindSMsb, which for negative i32
    # (top bit set) returns the most-significant 0-bit instead of MSB-of-1, giving a leader that's one less than
    # the actual highest matching lane.  Concretely, with lane 31 holding the only key for its digit,
    # bin_mask = 0x80000000; FindSMsb on -2147483648 returns 30 (highest 0-bit), so 31 - 30 = 1 elects lane 1
    # instead of lane 31, and lane 31's shuffle reads from lane 1 (= 0) - observed as last-lane ranks off by one on
    # Vulkan / Metal.  Now that the subgroup layer dispatches FindUMsb for unsigned ``clz``, passing the u32 directly
    # emits the right intrinsic on every backend.
    bin_mask = _ops.cast(smem_match[match_idx], _u32)
    leader = _i32(31) - _ops.cast(_ops.clz(bin_mask), _i32)
    popc = _ops.popcnt(_ops.bit_and(bin_mask, lane_mask_le_v))

    # Leader claims `popc` slots from this subgroup's slice of the subgroup_offsets entry.
    subgroup_offset = _i32(0)
    if lane == leader:
        subgroup_offset = _ops.atomic_add(smem_offsets[subgroup_idx * RADIX_DIGITS + digit], _ops.cast(popc, _i32))

    # Leader broadcasts its claimed offset to every lane in the subgroup.
    subgroup_offset = _subgroup.shuffle(subgroup_offset, _ops.cast(leader, _u32))

    # Leader resets the match mask so subsequent passes (or items_per_thread > 1) start clean.
    if lane == leader:
        smem_match[match_idx] = _i32(0)
    _subgroup_sync_fence()

    rank = subgroup_offset + _ops.cast(popc, _i32) - _i32(1)

    # Step 6: publish bins + exclusive_digit_prefix to the caller-supplied outparams.  ``block_dim == RADIX_DIGITS`` so
    # every thread writes exactly one digit.  Followed by a ``block.sync()`` so the caller can read these arrays
    # without having to add their own retiring barrier.
    bins[tid] = bin_count
    excl_prefix[tid] = exclusive_digit_prefix
    sync()

    return rank


@_func
def _radix_rank_match_atomic_or_wave64(
    key,
    block_dim: template(),
    radix_bits: template(),
    bit_start: template(),
    num_bits: template(),
    bins,
    excl_prefix,
):
    """Wave64 implementation of `radix_rank_match_atomic_or`. See the public wrapper for the contract.

    Match-mask region is ``i64``; atomic_or on shared ``i64`` is native on AMDGPU LDS.  Subgroup ``lanemask_le`` is
    u32-only by contract (see ``subgroup.py``: "lane_id in [0, 31]"), so the 64-lane form is synthesized inline as
    ``one_at_lane | (one_at_lane - 1)`` — avoids the UB of shifting by 64 when lane == 63.

    Structural twin of the wave32 path; duplicated rather than parameterised because Quadrants' AST transformer
    doesn't carry locals across ``if impl.static`` branches and the smem_match dtype + match-phase widths are the only
    things that differ.
    """
    SUBGROUP_THREADS = impl.static(_subgroup.group_size())
    RADIX_DIGITS = impl.static(1 << radix_bits)
    BLOCK_SUBGROUPS = impl.static(block_dim // SUBGROUP_THREADS)
    NUM_BITS_MASK = impl.static((1 << num_bits) - 1)
    BINS_PER_LANE = impl.static(RADIX_DIGITS // SUBGROUP_THREADS)

    smem_offsets = SharedArray(impl.static((BLOCK_SUBGROUPS * RADIX_DIGITS,)), _i32)
    smem_match = SharedArray(impl.static((BLOCK_SUBGROUPS * RADIX_DIGITS,)), _i64)

    tid = thread_idx()
    subgroup_idx = tid // SUBGROUP_THREADS
    lane = _ops.cast(_subgroup.invocation_id(), _i32)

    # Step 1: zero per-subgroup histograms and match_masks.
    for b in impl.static(range(BINS_PER_LANE)):
        bin_idx = lane + impl.static(b * SUBGROUP_THREADS)
        smem_offsets[subgroup_idx * RADIX_DIGITS + bin_idx] = _i32(0)
        smem_match[subgroup_idx * RADIX_DIGITS + bin_idx] = _i64(0)
    _subgroup_sync_fence()

    digit = _ops.cast(_ops.bit_and(_ops.bit_shr(key, _u32(bit_start)), _u32(NUM_BITS_MASK)), _i32)
    _ops.atomic_add(smem_offsets[subgroup_idx * RADIX_DIGITS + digit], _i32(1))

    sync()

    bin_count = _i32(0)
    for j_subgroup in impl.static(range(BLOCK_SUBGROUPS)):
        subgroup_count = smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid]
        smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid] = bin_count
        bin_count = bin_count + subgroup_count

    exclusive_digit_prefix = exclusive_add(bin_count, block_dim, _i32)

    for j_subgroup in impl.static(range(BLOCK_SUBGROUPS)):
        smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid] = (
            smem_offsets[impl.static(j_subgroup * RADIX_DIGITS) + tid] + exclusive_digit_prefix
        )

    sync()

    # Step 5 - wave64 specifics: u64 ballot mask via inline ``one_at_lane | (one_at_lane - 1)`` (avoids UB on
    # lane=63), atomic_or on the i64 match cell, clz / popcnt on u64.  Leader formula is ``63 - clz(u64)``.
    lane_u64 = _ops.cast(lane, _u64)
    lane_mask = _u64(1) << lane_u64
    lane_mask_le_v = lane_mask | (lane_mask - _u64(1))

    match_idx = subgroup_idx * RADIX_DIGITS + digit

    _ops.atomic_or(smem_match[match_idx], _ops.cast(lane_mask, _i64))
    _subgroup_sync_fence()

    # u64 clz via FindUMsb-equivalent on every backend; the wave32 path's caveat about FindSMsb vs FindUMsb on i64
    # would apply on SPIR-V wave64 devices if those existed (today wave64 = AMDGPU only).
    bin_mask = _ops.cast(smem_match[match_idx], _u64)
    leader = _i32(63) - _ops.cast(_ops.clz(bin_mask), _i32)
    popc = _ops.popcnt(_ops.bit_and(bin_mask, lane_mask_le_v))

    subgroup_offset = _i32(0)
    if lane == leader:
        subgroup_offset = _ops.atomic_add(smem_offsets[subgroup_idx * RADIX_DIGITS + digit], _ops.cast(popc, _i32))

    subgroup_offset = _subgroup.shuffle(subgroup_offset, _ops.cast(leader, _u32))

    if lane == leader:
        smem_match[match_idx] = _i64(0)
    _subgroup_sync_fence()

    rank = subgroup_offset + _ops.cast(popc, _i32) - _i32(1)

    bins[tid] = bin_count
    excl_prefix[tid] = exclusive_digit_prefix
    sync()

    return rank


@_func
def radix_rank_match_atomic_or(
    key,
    block_dim: template(),
    radix_bits: template(),
    bit_start: template(),
    num_bits: template(),
    bins,
    excl_prefix,
):
    """Block-level radix rank via the atomic-OR match-and-count strategy.

    Returns the calling thread's stable rank within the block under digit
    ``(key >> bit_start) & ((1 << num_bits) - 1)``.

    Args:
        key: ``u32`` key, one per thread.
        block_dim: threads per block (template).  Must equal ``RADIX_DIGITS = 1 << radix_bits``: each digit gets
            exactly one thread for the per-thread bin/excl_prefix output.
        radix_bits: number of bits in the digit (template).  Typical onesweep value is 8, giving 256 digits.
        bit_start: starting bit of the digit (template).  Used as ``key >> bit_start``.
        num_bits: actual digit width in bits (template), with ``num_bits <= radix_bits``.  Bits
            ``[bit_start, bit_start + num_bits)`` of ``key`` are extracted.
        bins: ``block.SharedArray((1 << radix_bits,), qd.i32)`` outparam.  After the call, ``bins[d]`` holds the count
            of keys whose digit equals ``d``.  Caller is responsible for allocating this array exactly once per kernel.
        excl_prefix: ``block.SharedArray((1 << radix_bits,), qd.i32)`` outparam.  After the call, ``excl_prefix[d]``
            holds the exclusive prefix sum of ``bins`` up to digit ``d``.  Caller allocates as for ``bins``.

    The calling thread's block-local index is read internally via `block.thread_idx()`; the subgroup size is read from
    `subgroup.group_size()` at compile time.  Supports both wave32 (CUDA, Vulkan-on-NVIDIA, Metal) and wave64
    (AMDGPU - Quadrants pins every AMDGPU target to ``+wavefrontsize64``).  Dispatches to one of two private
    implementations at compile time based on subgroup size; the match-mask shared-memory region's dtype is the only
    semantic difference (``i32`` on wave32, ``i64`` on wave64), but Quadrants' AST transformer doesn't carry locals
    across ``if impl.static`` branches so the two paths are written as separate ``@func`` bodies.  Atomic ``or`` on
    ``i64`` shared memory is native on AMDGPU's LDS; wave32 backends never see the ``i64`` path so portability does
    not depend on SPIR-V / Metal supporting 64-bit threadgroup atomics.

    Pre/post: caller must guarantee uniform control flow on entry; the function inserts the necessary ``block.sync()``
    and ``subgroup.sync()`` retires.  After the call, ``bins`` and ``excl_prefix`` are visible to every thread without
    a further ``block.sync()`` (we sync internally before exit).

    Cost: ``~items_per_thread`` atomic_or + atomic_add per pass on shared memory + 2 ``block.sync()`` + 1 block
    exclusive scan + ``BLOCK_SUBGROUPS`` ops per thread for the column-sum upsweep.  Shared-memory footprint at the
    default ``radix_bits=8``: 4 KiB ``i32`` for subgroup offsets + 4 KiB ``i32`` (wave32) or 8 KiB ``i64`` (wave64)
    for the match-mask region - so 8 KiB total on wave32, 12 KiB on wave64.
    """
    SUBGROUP_THREADS = impl.static(_subgroup.group_size())
    impl.static_assert(
        impl.static(SUBGROUP_THREADS == 32 or SUBGROUP_THREADS == 64),
        "block.radix_rank_match_atomic_or: subgroup size must be 32 or 64",
    )
    RADIX_DIGITS = impl.static(1 << radix_bits)
    impl.static_assert(
        impl.static(block_dim == RADIX_DIGITS),
        "block.radix_rank_match_atomic_or: block_dim must equal RADIX_DIGITS (1 << radix_bits)",
    )
    if impl.static(SUBGROUP_THREADS == 32):
        return _radix_rank_match_atomic_or_wave32(key, block_dim, radix_bits, bit_start, num_bits, bins, excl_prefix)
    return _radix_rank_match_atomic_or_wave64(key, block_dim, radix_bits, bit_start, num_bits, bins, excl_prefix)


# Shared-memory emulation of CUDA's hardware-fused barrier-with-reduction ops, used on backends that lack a direct
# equivalent (AMDGPU has no NVPTX `barrier.cta.red.*` analog; SPIR-V's `OpGroupNonUniform*` only operate at subgroup
# scope reliably across Vulkan + Metal).
#
# Pattern: lane 0 zeroes a 1-element shared `i32` -> block.sync() -> every thread atomically folds its predicate into
# the slot -> block.sync() -> every thread reads the broadcasted result. Costs 2 barriers + 1 atomic (vs. CUDA's
# hardware fast path of 1 barrier+reduction). Slower than the CUDA path but functionally equivalent and portable. Each
# call-site allocates a fresh `SharedArray` so multiple calls in the same kernel do not alias each other.
#
# IMPORTANT: every thread must participate in the `atomic_add` call unconditionally (guarding with
# `if predicate: atomic_add(...)` is NOT safe). On Metal, `workgroupBarrier` does not propagate atomic writes from
# divergent branches to threads that did not enter the branch -- non-participating SIMD groups never see the updated
# slot. By having every thread call `atomic_add(counter, select(...))` the control flow stays uniform, the barrier
# synchronises correctly, and all threads read the final count.
#
# We also use `atomic_add` rather than `atomic_or` because Metal / MoltenVK silently no-ops `OpAtomicOr` on threadgroup
# memory in some configurations.
@_func
def _block_reduce_count_nonzero_emulated(predicate: _i32) -> _i32:
    counter = SharedArray((1,), _i32)
    if thread_idx() == 0:
        counter[0] = 0
    sync()
    _ops.atomic_add(counter[0], _ops.select(predicate != 0, 1, 0))
    sync()
    return counter[0]


@_func
def _block_reduce_any_nonzero_emulated(predicate: _i32) -> _i32:
    counter = SharedArray((1,), _i32)
    if thread_idx() == 0:
        counter[0] = 0
    sync()
    _ops.atomic_add(counter[0], _ops.select(predicate != 0, 1, 0))
    sync()
    return _ops.min(counter[0], 1)


@_func
def _block_reduce_all_nonzero_emulated(predicate: _i32) -> _i32:
    counter = SharedArray((1,), _i32)
    if thread_idx() == 0:
        counter[0] = 0
    sync()
    _ops.atomic_add(counter[0], _ops.select(predicate == 0, 1, 0))
    sync()
    return 1 - _ops.min(counter[0], 1)
