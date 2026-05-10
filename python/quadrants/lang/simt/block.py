# type: ignore
# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.expr import make_expr_group
from quadrants.lang.kernel_impl import func
from quadrants.lang.simt.subgroup import (
    _bin_add,
    _bin_max,
    _bin_min,
    invocation_id,
    shuffle_down,
)
from quadrants.lang.simt.subgroup import _exclusive_scan as _subgroup_exclusive_scan
from quadrants.lang.simt.subgroup import _inclusive_scan as _subgroup_inclusive_scan
from quadrants.lang.util import quadrants_scope
from quadrants.types.annotations import template
from quadrants.types.primitive_types import u32


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
        return impl.call_internal("block_barrier_and_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_all_nonzero is not supported for arch {arch}")


def sync_any_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_or_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_any_nonzero is not supported for arch {arch}")


def sync_count_nonzero(predicate):
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier_count_i32", predicate, with_runtime_context=False)
    raise ValueError(f"qd.block.sync_count_nonzero is not supported for arch {arch}")


def mem_sync():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda:
        return impl.call_internal("block_barrier", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("workgroupMemoryBarrier", with_runtime_context=False)
    raise ValueError(f"qd.block.mem_sync is not supported for arch {arch}")


def thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch_uses_spv(arch):
        return impl.call_internal("localInvocationId", with_runtime_context=False)
    raise ValueError(f"qd.block.thread_idx is not supported for arch {arch}")


def global_thread_idx():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or _qd_core.amdgpu:
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
# CUB-style two-stage block reduce (`BLOCK_REDUCE_WARP_REDUCTIONS`): each warp reduces its lanes via `shuffle_down`,
# lane 0 of every warp publishes the warp aggregate to shared memory, a `block.sync()` retires the publish, and thread 0
# sequentially folds the warp aggregates with `op`.  Cost: `log2_warp` shuffles + 1 shared-mem write/read per warp + 1
# `block.sync` + (NUM_WARPS - 1) ops on thread 0.
#
# `_warp_reduce` mirrors `subgroup.reduce_add` / `_min` / `_max` but takes a generic template operator so the same
# kernel skeleton covers add / min / max / mul / bitwise / custom monoids.  We don't reuse `subgroup.reduce_add` etc.
# directly because we want one source of truth for the block path's per-warp step and a cheap way to plug in arbitrary
# operators (used internally by `_reduce`, `_reduce_all`, and downstream block scans).


@func
def _warp_reduce(value, log2_size: template(), op: template()):
    """Tree-reduce ``value`` across ``2**log2_size`` consecutive lanes via ``shuffle_down`` under a generic ``op``.

    Result valid in lane 0 of each ``2**log2_size`` group; other lanes hold partial values.  ``log2_size`` is a
    compile-time template, so the body unrolls into ``log2_size`` shuffle+op pairs.
    """
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << (log2_size - 1 - i))
        value = op(value, shuffle_down(value, u32(offset)))
    return value


@func
def _reduce(value, tid, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope reduction under a generic associative ``op``.  Result is valid in **thread 0 only**; other threads
    retain partial values.  Use ``_reduce_all`` if you need the result on every thread.

    ``tid`` is the calling thread's block-local index (``i % block_dim`` from a ``loop_config(block_dim=...)`` kernel,
    or ``qd.simt.block.thread_idx()`` on backends that expose it).  ``block_dim`` must be a multiple of
    ``2**log2_warp``; ``log2_warp`` should be 5 on CUDA / Metal / RDNA and 6 on CDNA AMDGPU.  ``dtype`` sizes the
    inter-warp shared-memory staging slot.

    When the block is exactly one warp (``block_dim == 2**log2_warp``) the shared-memory path is short-circuited at
    trace time and the call costs only the per-warp tree.
    """
    WARP_SIZE = impl.static(1 << log2_warp)
    NUM_WARPS = impl.static(block_dim // WARP_SIZE)

    warp_agg = _warp_reduce(value, log2_warp, op)

    if impl.static(NUM_WARPS == 1):
        return warp_agg

    warp_id = tid // WARP_SIZE
    lane_id = invocation_id()

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


@func
def _reduce_all(value, tid, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block reduction whose result is broadcast to every thread.  Costs one extra ``block.sync()`` plus a one-slot
    shared-memory broadcast vs. ``_reduce``.
    """
    result = _reduce(value, tid, block_dim, log2_warp, op, dtype)
    bcast = SharedArray((1,), dtype)
    if tid == 0:
        bcast[0] = result
    sync()
    return bcast[0]


@func
def reduce(value, tid, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope reduction under a generic associative ``op``.  Result valid in **thread 0 only**.  See `reduce_add`
    for the argument contract; this generic form additionally takes an ``op: template()`` ``@qd.func`` taking two args
    and returning the same type as ``value``, so callers can plug in custom monoids (bitwise ops, multiplicative,
    matrix-multiply, etc.) without re-implementing the warp-reduce + shared-mem skeleton.
    """
    return _reduce(value, tid, block_dim, log2_warp, op, dtype)


@func
def reduce_all(value, tid, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope reduction under a generic associative ``op``, broadcast to every thread.  See `reduce_all_add` for
    the cost profile and `reduce` for the operator contract.
    """
    return _reduce_all(value, tid, block_dim, log2_warp, op, dtype)


@func
def reduce_add(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope sum reduction.  Result valid in **thread 0 only**.

    Args:
        value: per-thread input.
        tid: calling thread's block-local index.
        block_dim: threads per block (template, multiple of ``2**log2_warp``).
        log2_warp: ``log2(warp_size)``; 5 on CUDA / Metal / RDNA, 6 on CDNA AMDGPU.
        dtype: scalar dtype for the inter-warp shared-memory staging slot (must match ``value``'s type).
    """
    return _reduce(value, tid, block_dim, log2_warp, _bin_add, dtype)


@func
def reduce_min(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope min reduction.  Result valid in **thread 0 only**.  See `reduce_add` for the argument contract."""
    return _reduce(value, tid, block_dim, log2_warp, _bin_min, dtype)


@func
def reduce_max(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope max reduction.  Result valid in **thread 0 only**.  See `reduce_add` for the argument contract."""
    return _reduce(value, tid, block_dim, log2_warp, _bin_max, dtype)


@func
def reduce_all_add(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope sum reduction with the result broadcast to every thread.  See `reduce_add` for the cheaper
    thread-0-only variant and for the argument contract.
    """
    return _reduce_all(value, tid, block_dim, log2_warp, _bin_add, dtype)


@func
def reduce_all_min(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope min reduction broadcast to every thread.  See `reduce_all_add`."""
    return _reduce_all(value, tid, block_dim, log2_warp, _bin_min, dtype)


@func
def reduce_all_max(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope max reduction broadcast to every thread.  See `reduce_all_add`."""
    return _reduce_all(value, tid, block_dim, log2_warp, _bin_max, dtype)


# --- Block scans -----------------------------------------------------------------------
#
# CUB-style two-stage block scan (`BLOCK_SCAN_WARP_SCANS`).  Each warp does a Hillis-Steele
# scan via `subgroup.{_inclusive_scan, _exclusive_scan}`, the last lane of every warp
# publishes the warp aggregate to shared memory, then every thread sequentially folds the
# warp prefixes and applies its own warp's prefix to its scan value.  All threads receive a
# valid result; cost: one warp scan + 1 shared-mem write/read per warp + 1 `block.sync()` +
# (NUM_WARPS - 1) ops on every thread (the cross-warp prefix is computed redundantly to
# avoid a second barrier — same trade-off CUB makes).
#
# Inclusive: warp aggregate at the last lane is just the inclusive value, written directly.
# Exclusive: warp aggregate = `op(exclusive[last_lane], value[last_lane])`, since the
# exclusive scan does not include the last lane's input — we recover the inclusive total
# with one extra `op` on the publish path.


@func
def _inclusive_block(value, tid, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope inclusive scan under a generic associative ``op``.  Every thread receives a valid result.

    See `inclusive_add` for the argument contract.  When the block is exactly one warp the cross-warp
    shared-memory path is short-circuited at trace time and the call costs only the per-warp Hillis-Steele tree.
    """
    WARP_SIZE = impl.static(1 << log2_warp)
    NUM_WARPS = impl.static(block_dim // WARP_SIZE)

    inclusive = _subgroup_inclusive_scan(value, op, log2_warp)

    if impl.static(NUM_WARPS == 1):
        return inclusive

    warp_id = tid // WARP_SIZE
    lane_id = invocation_id()

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


@func
def _exclusive_block(
    value, tid, block_dim: template(), log2_warp: template(), op: template(), identity, dtype: template()
):
    """Block-scope exclusive scan under a generic associative ``op``.  Every thread receives a valid result; thread 0
    holds ``identity``.  See `exclusive_add` for the argument contract.
    """
    WARP_SIZE = impl.static(1 << log2_warp)
    NUM_WARPS = impl.static(block_dim // WARP_SIZE)

    exclusive = _subgroup_exclusive_scan(value, op, identity, log2_warp)

    if impl.static(NUM_WARPS == 1):
        return exclusive

    warp_id = tid // WARP_SIZE
    lane_id = invocation_id()

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


@func
def inclusive_scan(value, tid, block_dim: template(), log2_warp: template(), op: template(), dtype: template()):
    """Block-scope inclusive scan under a generic associative ``op``.  See `inclusive_add` for the argument contract;
    this generic form additionally takes an ``op: template()`` ``@qd.func`` so callers can plug in custom monoids
    without re-implementing the per-warp + shared-mem skeleton.
    """
    return _inclusive_block(value, tid, block_dim, log2_warp, op, dtype)


@func
def exclusive_scan(
    value, tid, block_dim: template(), log2_warp: template(), op: template(), identity, dtype: template()
):
    """Block-scope exclusive scan under a generic associative ``op`` with explicit ``identity``.  See `exclusive_add`
    for the argument contract.  Thread 0 holds ``identity``; subsequent threads hold ``op(v[0], ..., v[i-1])``.
    """
    return _exclusive_block(value, tid, block_dim, log2_warp, op, identity, dtype)


@func
def inclusive_add(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope inclusive prefix sum.  After the call, thread ``i`` holds ``v[0] + v[1] + ... + v[i]``.

    Args:
        value: per-thread input.
        tid: calling thread's block-local index (``i % block_dim``, or `block.thread_idx()`).
        block_dim: threads per block (template, multiple of ``2**log2_warp``).
        log2_warp: ``log2(warp_size)``; 5 on CUDA / Metal / RDNA, 6 on CDNA AMDGPU.
        dtype: scalar dtype for the inter-warp shared-memory staging slot; must match ``value``'s type.
    """
    return _inclusive_block(value, tid, block_dim, log2_warp, _bin_add, dtype)


@func
def inclusive_min(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope inclusive prefix min.  See `inclusive_add` for the argument contract."""
    return _inclusive_block(value, tid, block_dim, log2_warp, _bin_min, dtype)


@func
def inclusive_max(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope inclusive prefix max.  See `inclusive_add` for the argument contract."""
    return _inclusive_block(value, tid, block_dim, log2_warp, _bin_max, dtype)


@func
def exclusive_add(value, tid, block_dim: template(), log2_warp: template(), dtype: template()):
    """Block-scope exclusive prefix sum.  After the call, thread ``i > 0`` holds ``v[0] + v[1] + ... + v[i-1]`` and
    thread 0 holds the additive identity (zero, in ``value``'s dtype, derived as ``value - value``).  See `inclusive_add`
    for the argument contract.
    """
    return _exclusive_block(value, tid, block_dim, log2_warp, _bin_add, value - value, dtype)


@func
def exclusive_min(value, tid, block_dim: template(), log2_warp: template(), identity, dtype: template()):
    """Block-scope exclusive prefix min.  Thread 0 holds ``identity``: the caller must supply a value that is ``>=``
    every legal element of the input (typically ``+∞`` for floats, the dtype's maximum for integers).  See
    `subgroup.exclusive_min` for why this op alone takes an explicit identity.
    """
    return _exclusive_block(value, tid, block_dim, log2_warp, _bin_min, identity, dtype)


@func
def exclusive_max(value, tid, block_dim: template(), log2_warp: template(), identity, dtype: template()):
    """Block-scope exclusive prefix max.  Thread 0 holds ``identity``: the caller must supply a value that is ``<=``
    every legal element of the input (typically ``-∞`` for floats, the dtype's minimum for integers).  See
    `exclusive_min`.
    """
    return _exclusive_block(value, tid, block_dim, log2_warp, _bin_max, identity, dtype)
