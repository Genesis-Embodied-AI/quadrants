# pyright: reportInvalidTypeForm=false

import warnings

from quadrants.lang import impl
from quadrants.lang.kernel_impl import func
from quadrants.types.annotations import template
from quadrants.types.primitive_types import u32


def sync():
    return impl.call_internal("subgroupBarrier", with_runtime_context=False)


def mem_fence():
    return impl.call_internal("subgroupMemoryBarrier", with_runtime_context=False)


_barrier_deprecation_warned = False
_memory_barrier_deprecation_warned = False


def barrier():
    global _barrier_deprecation_warned
    if not _barrier_deprecation_warned:
        _barrier_deprecation_warned = True
        warnings.warn(
            "qd.simt.subgroup.barrier() is deprecated; use qd.simt.subgroup.sync() instead "
            "(matching block.sync()).",
            DeprecationWarning,
            stacklevel=2,
        )
    return sync()


def memory_barrier():
    global _memory_barrier_deprecation_warned
    if not _memory_barrier_deprecation_warned:
        _memory_barrier_deprecation_warned = True
        warnings.warn(
            "qd.simt.subgroup.memory_barrier() is deprecated; use qd.simt.subgroup.mem_fence() instead "
            "(matching the planned block.mem_fence() / grid.mem_fence()).",
            DeprecationWarning,
            stacklevel=2,
        )
    return mem_fence()


def elect():
    return impl.call_internal("subgroupElect", with_runtime_context=False)


def all_true(cond):
    # TODO
    pass


def any_true(cond):
    # TODO
    pass


def all_equal(value):
    # TODO
    pass


def broadcast_first(value):
    # TODO
    pass


def broadcast(value, index):
    return impl.call_internal("subgroupBroadcast", value, index, with_runtime_context=False)


def group_size():
    return impl.call_internal("subgroupSize", with_runtime_context=False)


def invocation_id():
    return impl.call_internal("subgroupInvocationId", with_runtime_context=False)


@func
def reduce_add(value, log2_size: template()):
    """Sum ``value`` across ``2**log2_size`` consecutive lanes via a ``shuffle_down`` tree.

    The result is valid in lane 0 of each ``2**log2_size`` group; other lanes hold partial sums.
    Caller must ensure ``2**log2_size`` does not exceed the active subgroup size on the target
    (32 on CUDA/Metal, 32 on RDNA, 64 on CDNA).

    ``log2_size`` is a compile-time template; the body is fully unrolled into ``log2_size``
    shuffle+add operations in the calling kernel's IR.
    """
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << (log2_size - 1 - i))
        value = value + shuffle_down(value, u32(offset))
    return value


@func
def reduce_all_add(value, log2_size: template()):
    """Sum ``value`` across ``2**log2_size`` consecutive lanes via a butterfly XOR.

    The result is broadcast to all ``2**log2_size`` lanes.  Caller must ensure ``2**log2_size``
    does not exceed the active subgroup size on the target.

    ``log2_size`` is a compile-time template; the body is fully unrolled into ``log2_size``
    shuffle+add operations in the calling kernel's IR.
    """
    lane = invocation_id()
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        value = value + shuffle(value, u32(lane ^ mask))
    return value


# reduce_mul / reduce_min / reduce_max / reduce_and / reduce_or / reduce_xor (no-arg, SPIR-V-only)
# have been removed.  Build sized portable replacements on top of `shuffle_down` / `shuffle`
# following the same pattern as `reduce_add` / `reduce_all_add` above when needed.


def inclusive_add(value):
    return impl.call_internal("subgroupInclusiveAdd", value, with_runtime_context=False)


def inclusive_mul(value):
    return impl.call_internal("subgroupInclusiveMul", value, with_runtime_context=False)


def inclusive_min(value):
    return impl.call_internal("subgroupInclusiveMin", value, with_runtime_context=False)


def inclusive_max(value):
    return impl.call_internal("subgroupInclusiveMax", value, with_runtime_context=False)


def inclusive_and(value):
    return impl.call_internal("subgroupInclusiveAnd", value, with_runtime_context=False)


def inclusive_or(value):
    return impl.call_internal("subgroupInclusiveOr", value, with_runtime_context=False)


def inclusive_xor(value):
    return impl.call_internal("subgroupInclusiveXor", value, with_runtime_context=False)


def exclusive_add(value):
    # TODO
    pass


def exclusive_mul(value):
    # TODO
    pass


def exclusive_min(value):
    # TODO
    pass


def exclusive_max(value):
    # TODO
    pass


def exclusive_and(value):
    # TODO
    pass


def exclusive_or(value):
    # TODO
    pass


def exclusive_xor(value):
    # TODO
    pass


def shuffle(value, index):
    return impl.call_internal("subgroupShuffle", value, index, with_runtime_context=False)


def shuffle_xor(value, mask):
    # TODO
    pass


def shuffle_up(value, offset):
    return impl.call_internal("subgroupShuffleUp", value, offset, with_runtime_context=False)


def shuffle_down(value, offset):
    return impl.call_internal("subgroupShuffleDown", value, offset, with_runtime_context=False)


__all__ = [
    "sync",
    "mem_fence",
    "barrier",
    "memory_barrier",
    "elect",
    "all_true",
    "any_true",
    "all_equal",
    "broadcast_first",
    "reduce_add",
    "reduce_all_add",
    "inclusive_add",
    "inclusive_mul",
    "inclusive_min",
    "inclusive_max",
    "inclusive_and",
    "inclusive_or",
    "inclusive_xor",
    "exclusive_add",
    "exclusive_mul",
    "exclusive_min",
    "exclusive_max",
    "exclusive_and",
    "exclusive_or",
    "exclusive_xor",
    "shuffle",
    "shuffle_xor",
    "shuffle_up",
    "shuffle_down",
]
