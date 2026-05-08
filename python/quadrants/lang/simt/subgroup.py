# pyright: reportInvalidTypeForm=false

import warnings

from quadrants.lang import impl
from quadrants.lang.kernel_impl import func
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32


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


@func
def elect():
    """Return ``1`` on lane ``0`` of every subgroup and ``0`` on every other lane.

    Implemented portably as ``invocation_id() == 0``: every backend that lowers
    ``invocation_id()`` therefore lowers ``elect()`` at zero extra cost (it inlines at trace
    time into a single compare + zext).

    Note that this narrows SPIR-V's ``OpGroupNonUniformElect`` semantics, which may pick any
    *active* lane as the elected one.  Under the documented uniform-CF + all-lanes-active
    contract for ``qd.simt.subgroup`` this distinction is invisible (lane 0 is always
    active and is a legal choice), and pinning down the elected lane keeps the behaviour
    consistent across backends.
    """
    return i32(invocation_id() == 0)


def all_true(cond):
    # TODO
    pass


def any_true(cond):
    # TODO
    pass


def all_equal(value):
    # TODO
    pass


def broadcast(value, index):
    return impl.call_internal("subgroupBroadcast", value, index, with_runtime_context=False)


@func
def broadcast_first(value):
    """Broadcast lane 0's ``value`` to every lane in the subgroup.

    Equivalent to ``broadcast(value, qd.u32(0))``; ``0`` is trivially dynamically uniform, so the
    SPIR-V ``OpGroupNonUniformBroadcast`` requirement is satisfied. Decorated with ``@qd.func`` and
    inlined into the calling kernel.
    """
    return broadcast(value, u32(0))


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


# --- Inclusive scans -------------------------------------------------------------------
#
# All seven inclusive scans share the same Hillis-Steele tree over `shuffle_up`; only
# the binary operator differs.  Each operator is a tiny ``@func`` so it can be passed
# as a ``template()`` callable to the shared `_inclusive_scan` helper, which inlines it
# into the per-lane reduce step.  ``log2_size`` is a compile-time constant so the loop
# fully unrolls into ``log2_size`` shuffle+op pairs in the calling kernel's IR.


@func
def _bin_add(a, b):
    return a + b


@func
def _bin_mul(a, b):
    return a * b


@func
def _bin_min(a, b):
    return min(a, b)


@func
def _bin_max(a, b):
    return max(a, b)


@func
def _bin_and(a, b):
    return a & b


@func
def _bin_or(a, b):
    return a | b


@func
def _bin_xor(a, b):
    return a ^ b


@func
def _inclusive_scan(value, op: template(), log2_size: template()):
    """Hillis-Steele inclusive scan of ``value`` under binary ``op``, over ``2**log2_size``
    consecutive lanes.  See `inclusive_add` for the contract; the only thing that changes
    between the seven `inclusive_*` ops is which ``_bin_*`` is passed here.

    The shuffle is in uniform CF (every lane participates); only the per-lane reduce step
    is conditional, matching the contract for ``shuffle_up``.  Cross-group ``shuffle_up``
    partners are masked out by ``lane_in_group >= offset``, so groups smaller than the
    full subgroup compose correctly when ``log2_size < log2(group_size)``.
    """
    lane_in_group = invocation_id() & impl.static((1 << log2_size) - 1)
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << i)
        partner = shuffle_up(value, u32(offset))
        if lane_in_group >= offset:
            value = op(value, partner)
    return value


@func
def inclusive_add(value, log2_size: template()):
    """Inclusive prefix sum across ``2**log2_size`` consecutive lanes.

    Lane ``i`` within each group of ``2**log2_size`` lanes returns
    ``v[group_start] + v[group_start + 1] + ... + v[i]``.  Caller must ensure
    ``2**log2_size`` does not exceed the active subgroup size on the target
    (32 on CUDA / Metal / RDNA, 64 on CDNA).
    """
    return _inclusive_scan(value, _bin_add, log2_size)


@func
def inclusive_mul(value, log2_size: template()):
    """Inclusive prefix product across ``2**log2_size`` consecutive lanes.  See
    `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_mul, log2_size)


@func
def inclusive_min(value, log2_size: template()):
    """Inclusive prefix min across ``2**log2_size`` consecutive lanes.  See
    `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_min, log2_size)


@func
def inclusive_max(value, log2_size: template()):
    """Inclusive prefix max across ``2**log2_size`` consecutive lanes.  See
    `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_max, log2_size)


@func
def inclusive_and(value, log2_size: template()):
    """Inclusive prefix bitwise-AND across ``2**log2_size`` consecutive lanes.  Integer
    dtypes only.  See `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_and, log2_size)


@func
def inclusive_or(value, log2_size: template()):
    """Inclusive prefix bitwise-OR across ``2**log2_size`` consecutive lanes.  Integer
    dtypes only.  See `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_or, log2_size)


@func
def inclusive_xor(value, log2_size: template()):
    """Inclusive prefix bitwise-XOR across ``2**log2_size`` consecutive lanes.  Integer
    dtypes only.  See `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_xor, log2_size)


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


@func
def shuffle_xor(value, mask):
    """Lane ``i`` reads ``value`` from lane ``i ^ mask``.

    Implemented portably as ``shuffle(value, lane ^ mask)``: every backend that lowers
    ``shuffle`` therefore lowers ``shuffle_xor``. ``mask`` is a ``u32`` and must place the
    XOR partner inside the active subgroup; results outside that range are
    implementation-defined (same caveat as ``shuffle``). Decorated with ``@qd.func`` and inlined
    into the calling kernel.
    """
    lane = invocation_id()
    return shuffle(value, u32(lane) ^ mask)


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
