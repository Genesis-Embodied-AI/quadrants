# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false

import warnings

from quadrants._lib import core as _qd_core
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
            "qd.simt.subgroup.barrier() is deprecated; use qd.simt.subgroup.sync() instead " "(matching block.sync()).",
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

    Implemented portably as ``invocation_id() == 0``: every backend that lowers ``invocation_id()`` therefore lowers
    ``elect()`` at zero extra cost (it inlines at trace time into a single compare + zext).

    Note that this narrows SPIR-V's ``OpGroupNonUniformElect`` semantics, which may pick any *active* lane as the
    elected one.  Under the documented uniform-CF + all-lanes-active contract for ``qd.simt.subgroup`` this distinction
    is invisible (lane 0 is always active and is a legal choice), and pinning down the elected lane keeps the behaviour
    consistent across backends.
    """
    return i32(invocation_id() == 0)


# --- Voting / predicate ops ------------------------------------------------------------
#
# All three are group-scoped over ``2**log2_size`` consecutive lanes, mirror the API of ``reduce_all_add`` /
# ``inclusive_*`` / ``exclusive_*``, and broadcast the result to every lane in the group as an ``i32`` (``0`` or ``1``).
#
# Backend strategy
# ----------------
# * On CUDA, when ``log2_size == 5`` (full warp), ``all_true`` / ``any_true`` lower to ``__all_sync(0xFFFFFFFF, p)`` /
#   ``__any_sync(0xFFFFFFFF, p)`` (one ``vote.all`` / ``vote.any`` instruction).  This shortcut is selected at trace
#   time via ``static()`` on ``impl.current_cfg().arch`` and on the compile-time ``log2_size`` template, so it
#   collapses to a single intrinsic call in the IR with no overhead vs. handwritten CUDA.
# * Every other backend, and CUDA for partial-warp groups, uses a portable ``shuffle_xor`` butterfly: ``log2_size``
#   shuffles + ``log2_size`` ANDs / ORs, fully unrolled into the calling kernel's IR.  Same shape as ``reduce_all_add``.
# * ``all_equal`` is always implemented as ``all_true(value == broadcast_group_lane_0)``, so it inherits the CUDA
#   shortcut transitively.  We don't reach for ``__match_all_sync`` because (a) it requires sm_70+, (b) it does
#   bit-equality on floats, contradicting the SPIR-V ``OpGroupNonUniformAllEqual`` semantics this op advertises
#   (``NaN != NaN``, ``+0.0 == -0.0``), and (c) the broadcast-then-AND form is only one shuffle slower while staying on
#   the same code path everywhere.


@func
def all_true(predicate, log2_size: template()):
    """AND-reduce ``predicate != 0`` across ``2**log2_size`` consecutive lanes.  Returns ``1`` (``i32``) on every lane
    of the group iff every lane in the group has a non-zero ``predicate``, else ``0``.

    Caller must ensure ``2**log2_size`` does not exceed the active subgroup size on the target (32 on CUDA / Metal, 64
    on AMDGPU — wave64 is forced on every AMDGPU target).  ``log2_size`` is a compile-time template; the body is fully
    unrolled.
    """
    p = i32(predicate != 0)
    if impl.static(impl.current_cfg().arch == _qd_core.cuda and log2_size == 5):
        return impl.call_internal("cuda_all_sync_i32", u32(0xFFFFFFFF), p, with_runtime_context=False)
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        p = p & shuffle_xor(p, u32(mask))
    return p


@func
def any_true(predicate, log2_size: template()):
    """OR-reduce ``predicate != 0`` across ``2**log2_size`` consecutive lanes.  Returns ``1`` (``i32``) on every lane
    of the group iff at least one lane in the group has a non-zero ``predicate``, else ``0``.

    See `all_true` for the size contract.
    """
    p = i32(predicate != 0)
    if impl.static(impl.current_cfg().arch == _qd_core.cuda and log2_size == 5):
        return impl.call_internal("cuda_any_sync_i32", u32(0xFFFFFFFF), p, with_runtime_context=False)
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        p = p | shuffle_xor(p, u32(mask))
    return p


@func
def all_equal(value, log2_size: template()):
    """Return ``1`` (``i32``) on every lane in each ``2**log2_size`` group iff every lane in the group has the same
    ``value``, else ``0``.

    Equality is the backend's native ``==`` on ``value``'s dtype: for floats this means ``NaN != NaN`` (a group with
    any ``NaN`` returns ``0``) and ``+0.0 == -0.0``, matching SPIR-V ``OpGroupNonUniformAllEqual``.  Callers wanting
    bit-equality on floats should bit-cast to the same-width integer dtype before calling.

    Implementation: each lane reads the value at the start of its group via ``shuffle``, then ``all_true`` AND-reduces
    the per-lane equality bit.  Cost: one ``shuffle`` plus one ``all_true`` (one ``vote.all`` on CUDA at
    ``log2_size == 5``, otherwise a ``log2_size``-deep ``shuffle_xor`` butterfly).
    """
    lane = invocation_id()
    group_base = u32(lane) & u32(~((1 << log2_size) - 1) & 0xFFFFFFFF)
    base = shuffle(value, group_base)
    return all_true(i32(value == base), log2_size)


def broadcast(value, index):
    return impl.call_internal("subgroupBroadcast", value, index, with_runtime_context=False)


@func
def broadcast_first(value):
    """Broadcast lane 0's ``value`` to every lane in the subgroup.

    Equivalent to ``broadcast(value, qd.u32(0))``; ``0`` is trivially dynamically uniform, so the SPIR-V
    ``OpGroupNonUniformBroadcast`` requirement is satisfied. Decorated with ``@qd.func`` and inlined into the calling
    kernel.
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
    (32 on CUDA / Metal, 64 on AMDGPU — wave64 is forced on every AMDGPU target).

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
# All seven inclusive scans share the same Hillis-Steele tree over `shuffle_up`; only the binary operator differs.
# Each operator is a tiny ``@func`` so it can be passed as a ``template()`` callable to the shared `_inclusive_scan`
# helper, which inlines it into the per-lane reduce step.  ``log2_size`` is a compile-time constant so the loop fully
# unrolls into ``log2_size`` shuffle+op pairs in the calling kernel's IR.


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
    """Hillis-Steele inclusive scan of ``value`` under binary ``op``, over ``2**log2_size`` consecutive lanes.  See
    `inclusive_add` for the contract; the only thing that changes between the seven `inclusive_*` ops is which
    ``_bin_*`` is passed here.

    The shuffle is in uniform CF (every lane participates); only the per-lane reduce step is conditional, matching the
    contract for ``shuffle_up``.  Cross-group ``shuffle_up`` partners are masked out by ``lane_in_group >= offset``, so
    groups smaller than the full subgroup compose correctly when ``log2_size < log2(group_size)``.

    Note: ``qd.select`` cannot be used here instead of ``if`` because ``OpSelect`` on MoltenVK / Metal miscompiles when
    one operand is an f32 produced by a shuffle intrinsic — the select silently returns the false-branch value
    regardless of the condition.  The ``if`` form works correctly on its own.
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

    Lane ``i`` within each group of ``2**log2_size`` lanes returns ``v[group_start] + v[group_start + 1] + ... + v[i]``.
    Caller must ensure ``2**log2_size`` does not exceed the active subgroup size on the target (32 on CUDA / Metal, 64
    on AMDGPU — wave64 is forced on every AMDGPU target).
    """
    return _inclusive_scan(value, _bin_add, log2_size)


@func
def inclusive_mul(value, log2_size: template()):
    """Inclusive prefix product across ``2**log2_size`` consecutive lanes.

    See `inclusive_add` for the size contract.
    """
    return _inclusive_scan(value, _bin_mul, log2_size)


@func
def inclusive_min(value, log2_size: template()):
    """Inclusive prefix min across ``2**log2_size`` consecutive lanes.  See `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_min, log2_size)


@func
def inclusive_max(value, log2_size: template()):
    """Inclusive prefix max across ``2**log2_size`` consecutive lanes.  See `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_max, log2_size)


@func
def inclusive_and(value, log2_size: template()):
    """Inclusive prefix bitwise-AND across ``2**log2_size`` consecutive lanes.  Integer dtypes only.  See
    `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_and, log2_size)


@func
def inclusive_or(value, log2_size: template()):
    """Inclusive prefix bitwise-OR across ``2**log2_size`` consecutive lanes.  Integer dtypes only.  See
    `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_or, log2_size)


@func
def inclusive_xor(value, log2_size: template()):
    """Inclusive prefix bitwise-XOR across ``2**log2_size`` consecutive lanes.  Integer dtypes only.  See
    `inclusive_add` for the size contract."""
    return _inclusive_scan(value, _bin_xor, log2_size)


# --- Exclusive scans -------------------------------------------------------------------
#
# Each `exclusive_*` shifts the *input* right by one lane via `shuffle_up(value, 1)`, seeds lane 0 of every group with
# the operator's identity, and then runs the inclusive scan on the shifted data.  Doing the shuffle first (rather than
# running the inclusive scan and shuffling the result) avoids the MoltenVK / Metal miscompile where the SPIR-V
# compiler misoptimizes the register holding the inclusive-scan result when its only consumer is a shuffle intrinsic.
# Lane 0's result must be set explicitly because `shuffle_up` with offset 1 returns an implementation-defined value at
# lane 0 (`OpGroupNonUniformShuffleUp` calls it undefined outright).  See `_exclusive_scan` for the shared body.
#
# Identity per op (in `value`'s dtype, expressed via dtype-preserving arithmetic so the wrapper does not need to
# inspect the dtype):
#
#   add: ``value - value``                  (zero)
#   mul: ``value - value + 1``              (one; the literal +1 takes value's dtype)
#   or:  ``value ^ value``                  (zero; bitwise xor of value with itself)
#   xor: ``value ^ value``                  (zero)
#   and: ``~(value ^ value)``               (all bits set; bitwise not of zero)
#
# For min and max there is no portable type-extreme that can be derived from `value` alone, so those two ops take an
# explicit ``identity`` argument: pass +∞ for `exclusive_min`, −∞ for `exclusive_max` (or whatever sentinel makes sense
# for the caller's dtype and value range).


@func
def _exclusive_scan(value, op: template(), identity, log2_size: template()):
    """Generic exclusive scan over ``2**log2_size`` consecutive lanes.

    Shift the input right by one lane within each group (filling lane 0 with ``identity``), then run the inclusive
    scan on the shifted data.  This avoids issuing a ``shuffle_up`` on the inclusive-scan result, which miscompiles on
    MoltenVK / Metal (the SPIR-V compiler misoptimizes the register holding the inclusive result when it is only
    consumed by a shuffle intrinsic).
    """
    lane_in_group = invocation_id() & impl.static((1 << log2_size) - 1)
    prev = shuffle_up(value, u32(1))
    if lane_in_group == 0:
        prev = identity
    return _inclusive_scan(prev, op, log2_size)


@func
def exclusive_add(value, log2_size: template()):
    """Exclusive prefix sum across ``2**log2_size`` consecutive lanes.

    Lane ``i`` (with ``i > 0``) within each group of ``2**log2_size`` lanes returns ``v[group_start] +
    v[group_start + 1] + ... + v[i - 1]``.  Lane 0 of each group returns the additive identity (zero, in ``value``'s
    dtype).
    """
    return _exclusive_scan(value, _bin_add, value - value, log2_size)


@func
def exclusive_mul(value, log2_size: template()):
    """Exclusive prefix product across ``2**log2_size`` consecutive lanes.  Lane 0 of each group returns the
    multiplicative identity (one, in ``value``'s dtype)."""
    return _exclusive_scan(value, _bin_mul, value - value + 1, log2_size)


@func
def exclusive_min(value, log2_size: template(), identity):
    """Exclusive prefix min across ``2**log2_size`` consecutive lanes.

    Lane 0 of each group returns ``identity``: the caller must supply a value that is ``>=`` every legal element of
    the input (typically ``+∞`` for floats, the dtype's maximum for integers).  See the module-level note for why this
    op alone takes an explicit identity.
    """
    return _exclusive_scan(value, _bin_min, identity, log2_size)


@func
def exclusive_max(value, log2_size: template(), identity):
    """Exclusive prefix max across ``2**log2_size`` consecutive lanes.

    Lane 0 of each group returns ``identity``: the caller must supply a value that is ``<=`` every legal element of
    the input (typically ``-∞`` for floats, the dtype's minimum for integers).  See the module-level note for why this
    op alone takes an explicit identity.
    """
    return _exclusive_scan(value, _bin_max, identity, log2_size)


@func
def exclusive_and(value, log2_size: template()):
    """Exclusive prefix bitwise-AND.  Integer dtypes only.  Lane 0 of each group returns all-bits-set in ``value``'s
    dtype."""
    return _exclusive_scan(value, _bin_and, ~(value ^ value), log2_size)


@func
def exclusive_or(value, log2_size: template()):
    """Exclusive prefix bitwise-OR.  Integer dtypes only.  Lane 0 of each group returns zero in ``value``'s dtype."""
    return _exclusive_scan(value, _bin_or, value ^ value, log2_size)


@func
def exclusive_xor(value, log2_size: template()):
    """Exclusive prefix bitwise-XOR.  Integer dtypes only.  Lane 0 of each group returns zero in ``value``'s dtype."""
    return _exclusive_scan(value, _bin_xor, value ^ value, log2_size)


def shuffle(value, index):
    return impl.call_internal("subgroupShuffle", value, index, with_runtime_context=False)


@func
def shuffle_xor(value, mask):
    """Lane ``i`` reads ``value`` from lane ``i ^ mask``.

    Implemented portably as ``shuffle(value, lane ^ mask)``: every backend that lowers ``shuffle`` therefore lowers
    ``shuffle_xor``. ``mask`` is a ``u32`` and must place the XOR partner inside the active subgroup; results outside
    that range are implementation-defined (same caveat as ``shuffle``). Decorated with ``@qd.func`` and inlined into
    the calling kernel.
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
