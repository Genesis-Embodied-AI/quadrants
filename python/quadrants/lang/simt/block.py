# type: ignore
# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false

import warnings

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang import ops as _ops
from quadrants.lang.expr import make_expr_group
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.simt import subgroup as _subgroup
from quadrants.lang.simt.subgroup import _bin_add, _bin_max, _bin_min
from quadrants.lang.util import quadrants_scope
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32 as _i32
from quadrants.types.primitive_types import u32 as _u32


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
# Two-stage block reduce: each warp reduces its lanes via `shuffle_down`, lane 0 of every warp publishes the warp
# aggregate to shared memory, a `block.sync()` retires the publish, and thread 0 sequentially folds the warp aggregates
# with `op`.  Cost: `log2_warp` shuffles + 1 shared-mem write/read per warp + 1 `block.sync` + (NUM_WARPS - 1) ops on
# thread 0.
#
# `_warp_reduce` mirrors `subgroup.reduce_add` / `_min` / `_max` but takes a generic template operator so the same
# kernel skeleton covers add / min / max / mul / bitwise / custom monoids.  We don't reuse `subgroup.reduce_add` etc.
# directly because we want one source of truth for the block path's per-warp step and a cheap way to plug in arbitrary
# operators (used internally by `reduce` / `reduce_all`).


@_func
def _warp_reduce(value, log2_size: template(), op: template()):
    """Tree-reduce ``value`` across ``2**log2_size`` consecutive lanes via ``shuffle_down`` under a generic ``op``.

    Result valid in lane 0 of each ``2**log2_size`` group; other lanes hold partial values.  ``log2_size`` is a
    compile-time template, so the body unrolls into ``log2_size`` shuffle+op pairs.
    """
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << (log2_size - 1 - i))
        value = op(value, _subgroup.shuffle_down(value, _u32(offset)))
    return value


@_func
def reduce(value, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope reduction under a generic associative ``op``.  Result is valid in **thread 0 only**; other threads
    retain partial values.  Use `reduce_all` if you need the result on every thread.

    Args:
        value: per-thread input.
        block_dim: threads per block (template, multiple of ``2**log2_warp``).
        log2_warp: ``log2(warp_size)``; 5 on CUDA / Metal / RDNA, 6 on CDNA AMDGPU.
        op: ``@qd.func`` taking two values and returning the same type as ``value``; callers can plug in custom
            associative monoids (bitwise ops, multiplicative, matrix-multiply, etc.) without re-implementing the
            per-warp + shared-mem skeleton.  See `reduce_add` for the standard sum specialization.
        dtype: scalar dtype for the inter-warp shared-memory staging slot (must match ``value``'s type).

    When the block is exactly one warp (``block_dim == 2**log2_warp``) the shared-memory path is short-circuited at
    trace time and the call costs only the per-warp tree.
    """
    WARP_SIZE = impl.static(1 << log2_warp)
    NUM_WARPS = impl.static(block_dim // WARP_SIZE)

    warp_agg = _warp_reduce(value, log2_warp, op)

    if impl.static(NUM_WARPS == 1):
        return warp_agg

    tid = thread_idx()
    warp_id = tid // WARP_SIZE
    # Use the **logical** lane (``tid & (WARP_SIZE-1)``) instead of ``invocation_id()``: on wave64 hardware
    # (CDNA AMDGPU) the hardware-lane index runs 0..63 inside one wave but logical warp 1 starts at lane 32, so
    # ``invocation_id() == 0`` would skip every other logical warp's publish.  ``tid & (WARP_SIZE-1)`` matches the
    # standard ``threadIdx.x & 31`` recipe on CUDA wave32 and stays correct when the hardware wave is wider than the
    # logical warp.
    lane_id = tid & impl.static(WARP_SIZE - 1)

    shared = SharedArray(impl.static((NUM_WARPS,)), dtype)
    if lane_id == 0:
        shared[warp_id] = warp_agg
    sync()

    result = warp_agg
    if tid == 0:
        result = shared[0]
        for w in impl.static(range(1, NUM_WARPS)):
            result = op(result, shared[impl.static(w)])
    return result


@_func
def reduce_all(value, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope reduction under a generic associative ``op``, broadcast to every thread.  Costs one extra
    ``block.sync()`` plus a one-slot shared-memory broadcast vs. `reduce`.  See `reduce` for the operator contract.
    """
    result = reduce(value, block_dim, log2_warp, op, dtype)
    bcast = SharedArray((1,), dtype)
    if thread_idx() == 0:
        bcast[0] = result
    sync()
    return bcast[0]


@_func
def reduce_add(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope sum reduction.  Result valid in **thread 0 only**.  See `reduce` for the argument contract."""
    return reduce(value, block_dim, log2_warp, _bin_add, dtype)


@_func
def reduce_min(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope min reduction.  Result valid in **thread 0 only**.  See `reduce` for the argument contract."""
    return reduce(value, block_dim, log2_warp, _bin_min, dtype)


@_func
def reduce_max(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope max reduction.  Result valid in **thread 0 only**.  See `reduce` for the argument contract."""
    return reduce(value, block_dim, log2_warp, _bin_max, dtype)


@_func
def reduce_all_add(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope sum reduction with the result broadcast to every thread.  See `reduce_add` for the cheaper
    thread-0-only variant and `reduce` for the argument contract.
    """
    return reduce_all(value, block_dim, log2_warp, _bin_add, dtype)


@_func
def reduce_all_min(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope min reduction broadcast to every thread.  See `reduce_all_add`."""
    return reduce_all(value, block_dim, log2_warp, _bin_min, dtype)


@_func
def reduce_all_max(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope max reduction broadcast to every thread.  See `reduce_all_add`."""
    return reduce_all(value, block_dim, log2_warp, _bin_max, dtype)


# --- Block scans -----------------------------------------------------------------------
#
# Two-stage block scan.  Each warp does a Hillis-Steele scan via
# `subgroup.{_inclusive_scan, _exclusive_scan}`, the last lane of every warp publishes the
# warp aggregate to shared memory, then every thread sequentially folds the warp prefixes
# and applies its own warp's prefix to its scan value.  All threads receive a valid result;
# cost: one warp scan + 1 shared-mem write/read per warp + 1 `block.sync()` + (NUM_WARPS - 1)
# ops on every thread (the cross-warp prefix is computed redundantly to avoid a second
# barrier).
#
# Inclusive: warp aggregate at the last lane is just the inclusive value, written directly.
# Exclusive: warp aggregate = `op(exclusive[last_lane], value[last_lane])`, since the
# exclusive scan does not include the last lane's input — we recover the inclusive total
# with one extra `op` on the publish path.


@_func
def inclusive_scan(value, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope inclusive scan under a generic associative ``op``.  Every thread receives a valid result.

    Args:
        value: per-thread input.
        block_dim: threads per block (template, multiple of ``2**log2_warp``).
        log2_warp: ``log2(warp_size)``; 5 on CUDA / Metal / RDNA, 6 on CDNA AMDGPU.
        op: ``@qd.func`` taking two values and returning the same type as ``value``; callers can plug in custom
            associative monoids without re-implementing the per-warp + shared-mem skeleton.  See `inclusive_add`
            for the standard sum specialization.
        dtype: scalar dtype for the inter-warp shared-memory staging slot; must match ``value``'s type.

    When the block is exactly one warp the cross-warp shared-memory path is short-circuited at trace time and the
    call costs only the per-warp Hillis-Steele tree.
    """
    WARP_SIZE = impl.static(1 << log2_warp)
    NUM_WARPS = impl.static(block_dim // WARP_SIZE)

    inclusive = _subgroup._inclusive_scan(value, op, log2_warp)

    if impl.static(NUM_WARPS == 1):
        return inclusive

    tid = thread_idx()
    warp_id = tid // WARP_SIZE
    # Logical lane within the 32-lane warp (see ``reduce`` for the wave32-vs-wave64 rationale).
    lane_id = tid & impl.static(WARP_SIZE - 1)

    shared = SharedArray(impl.static((NUM_WARPS,)), dtype)
    if lane_id == impl.static(WARP_SIZE - 1):
        shared[warp_id] = inclusive
    sync()

    # Sequential exclusive prefix scan over warp aggregates; each thread captures its own warp's prefix.  Warp 0's
    # prefix is unused (its inclusive value is already the prefix sum from the start of the block), so we never read
    # `warp_prefix` on warp 0; the placeholder there exists only to give the variable a definite type.
    block_aggregate = shared[0]
    warp_prefix = block_aggregate
    for w in impl.static(range(1, NUM_WARPS)):
        if warp_id == impl.static(w):
            warp_prefix = block_aggregate
        addend = shared[impl.static(w)]
        block_aggregate = op(block_aggregate, addend)

    if warp_id != 0:
        inclusive = op(warp_prefix, inclusive)
    return inclusive


@_func
def exclusive_scan(value, block_dim: template(), log2_warp: template(), op: template(), identity, dtype: template()):
    """Block-scope exclusive scan under a generic associative ``op`` with explicit ``identity``.  Every thread receives
    a valid result; thread 0 holds ``identity`` and thread ``i > 0`` holds ``op(v[0], ..., v[i-1])``.

    See `inclusive_scan` for the per-arg contract; in addition this op takes an explicit ``identity`` because exclusive
    scan needs a definite value for thread 0 (and for the sentinel paths in `exclusive_min` / `exclusive_max`).  See
    `exclusive_add` for the additive specialization which derives a zero identity automatically.
    """
    WARP_SIZE = impl.static(1 << log2_warp)
    NUM_WARPS = impl.static(block_dim // WARP_SIZE)

    exclusive = _subgroup._exclusive_scan(value, op, identity, log2_warp)

    if impl.static(NUM_WARPS == 1):
        return exclusive

    tid = thread_idx()
    warp_id = tid // WARP_SIZE
    # Logical lane within the 32-lane warp (see ``reduce`` for the wave32-vs-wave64 rationale).
    lane_id = tid & impl.static(WARP_SIZE - 1)

    shared = SharedArray(impl.static((NUM_WARPS,)), dtype)
    if lane_id == impl.static(WARP_SIZE - 1):
        # Warp aggregate = inclusive at last lane = exclusive[last] + value[last] under `op`.
        shared[warp_id] = op(exclusive, value)
    sync()

    block_aggregate = shared[0]
    warp_prefix = identity  # warp 0's prefix is the identity; subsequent warps overwrite this in their own iteration
    for w in impl.static(range(1, NUM_WARPS)):
        if warp_id == impl.static(w):
            warp_prefix = block_aggregate
        addend = shared[impl.static(w)]
        block_aggregate = op(block_aggregate, addend)

    if warp_id != 0:
        exclusive = op(warp_prefix, exclusive)
    return exclusive


@_func
def inclusive_add(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope inclusive prefix sum.  After the call, thread ``i`` holds ``v[0] + v[1] + ... + v[i]``.  See
    `inclusive_scan` for the argument contract.
    """
    return inclusive_scan(value, block_dim, log2_warp, _bin_add, dtype)


@_func
def inclusive_min(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope inclusive prefix min.  See `inclusive_scan` for the argument contract."""
    return inclusive_scan(value, block_dim, log2_warp, _bin_min, dtype)


@_func
def inclusive_max(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope inclusive prefix max.  See `inclusive_scan` for the argument contract."""
    return inclusive_scan(value, block_dim, log2_warp, _bin_max, dtype)


@_func
def exclusive_add(value, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope exclusive prefix sum.  After the call, thread ``i > 0`` holds ``v[0] + v[1] + ... + v[i-1]`` and
    thread 0 holds the additive identity (zero, in ``value``'s dtype, derived as ``value - value``).  See
    `exclusive_scan` for the argument contract.
    """
    return exclusive_scan(value, block_dim, log2_warp, _bin_add, value - value, dtype)


@_func
def exclusive_min(value, block_dim: template(), log2_warp: template(), identity, dtype: template()):
    """Block-scope exclusive prefix min.  Thread 0 holds ``identity``: the caller must supply a value that is ``>=``
    every legal element of the input (typically ``+∞`` for floats, the dtype's maximum for integers).  See
    `subgroup.exclusive_min` for why this op alone takes an explicit identity.
    """
    return exclusive_scan(value, block_dim, log2_warp, _bin_min, identity, dtype)


@_func
def exclusive_max(value, block_dim: template(), log2_warp: template(), identity, dtype: template()):
    """Block-scope exclusive prefix max.  Thread 0 holds ``identity``: the caller must supply a value that is ``<=``
    every legal element of the input (typically ``-∞`` for floats, the dtype's minimum for integers).  See
    `exclusive_min`.
    """
    return exclusive_scan(value, block_dim, log2_warp, _bin_max, identity, dtype)


# --- Block radix rank ------------------------------------------------------------------
#
# Block-level radix ranking via the atomic-OR match-and-count strategy.  Each thread holds a single ``u32`` key; the
# function returns the key's stable rank within the block under the digit `(key >> bit_start) & ((1 << num_bits) - 1)`,
# and writes the per-digit count and exclusive-prefix arrays to caller-supplied shared-memory outparams.
#
# The algorithm runs in six steps:
#
# 1. ComputeHistogramsWarp: each warp builds a private digit histogram in shared memory via ``atomic_add``.
# 2. ComputeOffsetsWarpUpsweep: every thread sums per-warp histograms column-wise to produce a block-wide bin count
#    for digit ``= tid``, while rewriting the warp histogram entries into per-warp running exclusive prefixes.
# 3. ExclusiveSum on the per-thread bin counts — uses the block exclusive scan defined above.
# 4. ComputeOffsetsWarpDownsweep: add the block-wide exclusive prefix into every warp's offset entry.
# 5. ComputeRanksItem (atomic-OR match): per-warp match via ``atomic_or`` on a per-digit lane-mask, then leader
#    (highest set lane) does a single ``atomic_add`` on the warp offset and broadcasts via ``subgroup.shuffle``; each
#    thread's rank is ``warp_offset + popc(bin_mask & lanemask_le) - 1``.
# 6. Write bin count + exclusive prefix to the outparam shared arrays.
#
# Shared-memory layout (all i32, total ``2 * BLOCK_WARPS * RADIX_DIGITS`` ints, 4096 ints = 16 KiB at the default
# 8-warp / 256-digit configuration):
#
#     warp_offsets / warp_histograms : [0, BLOCK_WARPS * RADIX_DIGITS)        (union backing)
#     match_masks                    : [BLOCK_WARPS * RADIX_DIGITS, 2 * ...)
#
# Warp-scope barriers use ``subgroup.sync()`` (lowers to ``__syncwarp`` on CUDA,
# ``OpControlBarrier(ScopeSubgroup, ...)`` on SPIR-V, ``s_barrier`` on AMDGPU).  ``LaneMaskLe()`` (the PTX intrinsic
# that gives a lane its less-than-or-equal lane mask) is replaced by ``subgroup.lanemask_le(lane)`` from the portable
# subgroup primitives.


@_func
def _warp_sync_fence():
    """Warp-scope barrier + memory fence — CUDA ``__syncwarp`` semantics across every backend.

    Why both ops: on CUDA, `subgroup.sync()` already lowers to `__syncwarp` which folds in a memory fence, so the
    extra `subgroup.mem_fence()` is redundant (a `__threadfence_block`).  On SPIR-V, however, the codegen emits
    `subgroupBarrier` as `OpControlBarrier(ScopeSubgroup, ScopeSubgroup, 0)` — i.e. with **no** memory semantics — so
    a bare `subgroup.sync()` does *not* publish prior shared-memory writes to other lanes.  The radix rank algorithm
    relies on the `__syncwarp` invariant that, after the barrier, every lane sees every other lane's prior
    `atomic_or` / `atomic_add` to shared memory; pairing the barrier with `subgroup.mem_fence()` (which emits a real
    `OpMemoryBarrier(ScopeSubgroup, AcquireRelease | UniformMemory | WorkgroupMemory)`) restores that invariant.
    """
    _subgroup.sync()
    _subgroup.mem_fence()


@_func
def radix_rank_match_atomic_or(
    key,
    block_dim: template(),
    log2_warp: template(),
    radix_bits: template(),
    bit_start: template(),
    num_bits: template(),
    bins,
    excl_prefix,
):
    """Block-level radix rank via the atomic-OR match-and-count strategy.

    Returns the calling thread's stable rank within the block under digit ``(key >> bit_start) & ((1 << num_bits) - 1)``.

    Args:
        key: ``u32`` key, one per thread.
        block_dim: threads per block (template).  Must equal ``RADIX_DIGITS = 1 << radix_bits``: each digit gets exactly
            one thread for the per-thread bin/excl_prefix output.
        log2_warp: ``log2(warp_size)`` (template).  Must currently be 5 (warp size 32) — the atomic-OR match path is
            built around 32-lane ``i32`` ballot masks, so wave64 callers should pass 5 and arrange to launch with a
            32-thread subgroup, or wait until a wave64 path lands.
        radix_bits: number of bits in the digit (template).  Typical onesweep value is 8, giving 256 digits.
        bit_start: starting bit of the digit (template).  Used as ``key >> bit_start``.
        num_bits: actual digit width in bits (template), with ``num_bits <= radix_bits``.  Bits ``[bit_start, bit_start +
            num_bits)`` of ``key`` are extracted.
        bins: ``block.SharedArray((1 << radix_bits,), qd.i32)`` outparam.  After the call, ``bins[d]`` holds the count
            of keys whose digit equals ``d``.  Caller is responsible for allocating this array exactly once per kernel.
        excl_prefix: ``block.SharedArray((1 << radix_bits,), qd.i32)`` outparam.  After the call, ``excl_prefix[d]`` holds
            the exclusive prefix sum of ``bins`` up to digit ``d``.  Caller allocates as for ``bins``.

    Pre/post: caller must guarantee uniform control flow on entry; the function inserts the necessary ``block.sync()``
    and ``subgroup.sync()`` retires.  After the call, ``bins`` and ``excl_prefix`` are visible to every thread without a
    further ``block.sync()`` (we sync internally before exit).

    Cost: ``~items_per_thread`` atomic_or + atomic_add per pass on shared memory + 2 ``block.sync()`` + 1 block exclusive
    scan + ``BLOCK_WARPS`` ops per thread for the column-sum upsweep.  The shared-memory footprint is
    ``2 * BLOCK_WARPS * RADIX_DIGITS`` i32 ints (16 KiB at the default ``log2_warp=5, radix_bits=8`` configuration).
    """
    WARP_THREADS = impl.static(1 << log2_warp)
    RADIX_DIGITS = impl.static(1 << radix_bits)
    BLOCK_WARPS = impl.static(block_dim // WARP_THREADS)
    NUM_BITS_MASK = impl.static((1 << num_bits) - 1)
    MM_OFF = impl.static(BLOCK_WARPS * RADIX_DIGITS)
    BINS_PER_LANE = impl.static(RADIX_DIGITS // WARP_THREADS)

    # ``TempStorage``: union of warp_offsets / warp_histograms (same backing) + match_masks.  All i32.
    smem = SharedArray(impl.static((2 * BLOCK_WARPS * RADIX_DIGITS,)), _i32)

    tid = thread_idx()
    warp_idx = tid // WARP_THREADS
    lane = _ops.cast(_subgroup.invocation_id(), _i32)

    # Step 1: zero per-warp histograms and match_masks.
    for b in impl.static(range(BINS_PER_LANE)):
        bin_idx = lane + impl.static(b * WARP_THREADS)
        smem[warp_idx * RADIX_DIGITS + bin_idx] = _i32(0)
        smem[MM_OFF + warp_idx * RADIX_DIGITS + bin_idx] = _i32(0)
    _warp_sync_fence()

    # Each thread atomic-adds 1 to its warp's bin for ``digit``.
    digit = _ops.cast(_ops.bit_and(_ops.bit_shr(key, _u32(bit_start)), _u32(NUM_BITS_MASK)), _i32)
    _ops.atomic_add(smem[warp_idx * RADIX_DIGITS + digit], _i32(1))

    sync()  # Publish per-warp histograms before column-sum.

    # Step 2: per-thread column sum across warps for digit == tid.  Each thread collects the running exclusive prefix
    # into ``bin_count`` while overwriting the warp histogram entries with their per-warp exclusive prefix.  After the
    # loop, ``bin_count`` is the block-wide total for digit == tid.
    bin_count = _i32(0)
    for j_warp in impl.static(range(BLOCK_WARPS)):
        warp_count = smem[impl.static(j_warp * RADIX_DIGITS) + tid]
        smem[impl.static(j_warp * RADIX_DIGITS) + tid] = bin_count
        bin_count = bin_count + warp_count

    # Step 3: block-wide exclusive sum on the per-thread bin counts.
    exclusive_digit_prefix = exclusive_add(bin_count, block_dim, log2_warp, _i32)

    # Step 4: ComputeOffsetsWarpDownsweep — fold the block-wide exclusive prefix into every warp's offset.
    for j_warp in impl.static(range(BLOCK_WARPS)):
        smem[impl.static(j_warp * RADIX_DIGITS) + tid] = (
            smem[impl.static(j_warp * RADIX_DIGITS) + tid] + exclusive_digit_prefix
        )

    sync()  # Publish warp offsets before the per-key match phase.

    # Step 5: per-key atomic-OR match.  ``items_per_thread == 1``, so this runs once per thread.
    lane_mask = _i32(1) << lane
    lane_mask_le_v = _subgroup.lanemask_le(_subgroup.invocation_id())

    match_idx = MM_OFF + warp_idx * RADIX_DIGITS + digit

    # Every thread ORs its lane_mask into the per-digit match mask of its warp.  Threads with the same digit collide
    # on the same shared-memory cell and produce a bitmask of "lanes in this warp that share this digit".
    _ops.atomic_or(smem[match_idx], lane_mask)
    _warp_sync_fence()

    # Read the bin_mask back and find the leader (highest matching lane) + intra-warp rank.  ``clz`` here MUST run on
    # the u32 (FindUMsb on SPIR-V): casting to i32 first triggers SPIR-V's FindSMsb, which for negative i32 (top bit
    # set) returns the most-significant 0-bit instead of MSB-of-1, giving a leader that's one less than the actual
    # highest matching lane.  Concretely, with lane 31 holding the only key for its digit, bin_mask = 0x80000000;
    # FindSMsb on -2147483648 returns 30 (highest 0-bit), so 31 - 30 = 1 elects lane 1 instead of lane 31, and lane
    # 31's shuffle reads from lane 1 (= 0) — observed as last-lane ranks off by one on Vulkan / Metal.  Now that the
    # subgroup layer dispatches FindUMsb for unsigned ``clz``, passing the u32 directly emits the right intrinsic on
    # every backend.
    bin_mask = _ops.cast(smem[match_idx], _u32)
    leader = _i32(31) - _ops.cast(_ops.clz(bin_mask), _i32)
    popc = _ops.popcnt(_ops.bit_and(bin_mask, lane_mask_le_v))

    # Leader claims `popc` slots from this warp's slice of the warp_offsets entry.
    warp_offset = _i32(0)
    if lane == leader:
        warp_offset = _ops.atomic_add(smem[warp_idx * RADIX_DIGITS + digit], _ops.cast(popc, _i32))

    # Leader broadcasts its claimed offset to every lane in the warp.
    warp_offset = _subgroup.shuffle(warp_offset, _ops.cast(leader, _u32))

    # Leader resets the match mask so subsequent passes (or items_per_thread > 1) start clean.
    if lane == leader:
        smem[match_idx] = _i32(0)
    _warp_sync_fence()

    rank = warp_offset + _ops.cast(popc, _i32) - _i32(1)

    # Step 6: publish bins + exclusive_digit_prefix to the caller-supplied outparams.  ``block_dim == RADIX_DIGITS`` so
    # every thread writes exactly one digit.  Followed by a ``block.sync()`` so the caller can read these arrays
    # without having to add their own retiring barrier.
    bins[tid] = bin_count
    excl_prefix[tid] = exclusive_digit_prefix
    sync()

    return rank


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
