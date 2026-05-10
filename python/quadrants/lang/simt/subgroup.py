# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false

import warnings

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.kernel_impl import func
from quadrants.lang.ops import clz
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32, u64


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


@func
def ballot_first_n(predicate, n: template()):
    """Return a ``u32`` bitmask whose bit ``i`` is set iff ``i < n`` AND lane ``i``'s ``predicate`` is non-zero.

    ``n`` is a ``qd.template()`` compile-time constant in ``[1, 32]``; bits ``>= n`` of the result are always zero.
    Pass ``n = 32`` for "ballot all 32 representable lanes" (the most common case, used by ``segmented_reduce_*`` and
    every other consumer of the u32 ballot in this module).

    Backend lowering:

    * CUDA: ``__ballot_sync(0xFFFFFFFF, predicate)`` — the warp is always 32 lanes, so the ``u32`` result naturally
      packs every lane.
    * AMDGPU: ``llvm.amdgcn.ballot.i64`` followed by ``trunc to i32``; bits ``[32, 64)`` of the i64 (lanes 32..63 on
      wave64) are always discarded, matching the ``ballot_first_n(p, n <= 32)`` contract.  This is a workaround for
      an LLVM AMDGPU isel bug — ``ballot.i32`` is documented as well-defined on wave64 (PR
      https://github.com/llvm/llvm-project/pull/71556) but in practice still fails ``Cannot select`` on gfx942 in
      LLVM 20 / 22 for non-constant predicates.  See ``codegen_amdgpu.cpp`` for the full bug + workaround comment.
    * SPIR-V: ``OpGroupNonUniformBallot`` (returns a uvec4); we extract component 0 = lanes 0..31's ballot.

    For ``n < 32`` we mask the predicate by ``lane < n`` before issuing the ballot, so bits ``[n, 32)`` of the result
    are forced to zero regardless of what those lanes' actual predicates are.  At ``n == 32`` the masking is provably
    a no-op on every backend (lanes ``>= 32`` are either non-existent on wave32 or already not represented in the u32
    result on wave64), so we shortcut and emit the ballot directly with no extra arithmetic.

    Caller contract: uniform CF + all lanes active.  If you need a mask covering more than 32 lanes (for wave64
    callers who want the full subgroup), use `ballot_full_subgroup` and check the high 32 bits.
    """
    if impl.static(n == 32):
        return impl.call_internal("subgroupBallotU32", predicate, with_runtime_context=False)
    lane = invocation_id()
    masked = predicate * i32(u32(lane) < u32(impl.static(n)))
    return impl.call_internal("subgroupBallotU32", masked, with_runtime_context=False)


def ballot_full_subgroup(predicate):
    """Return a ``u64`` bitmask covering the entire subgroup; bit ``i`` is set iff lane ``i``'s ``predicate`` is
    non-zero.  On wave32 backends (CUDA, RDNA wave32, most Vulkan / Metal) the high 32 bits of the result are always
    zero, since lanes ``>= 32`` do not exist; on wave64 backends (AMDGPU CDNA, GFX9, RDNA explicit-wave64) all 64 bits
    are meaningful.  Use this when you need a subgroup-wide population count, prefix-mask, or compaction that has to
    cover more than 32 lanes; use `ballot_first_n` when you only care about the first 32 lanes.

    Backend lowering:

    * CUDA: ``__ballot_sync(0xFFFFFFFF, predicate)`` zero-extended to ``u64`` — the warp is 32 lanes so the high half
      is always zero.
    * AMDGPU: ``llvm.amdgcn.ballot.i64`` — returns the full 64-bit ballot on wave64; on wave32 the AMDGPU backend
      lowers it to the wave32 ballot zero-extended to 64 bits, so the API stays uniform across wavefront modes.  The
      i64 form selects cleanly in current LLVM (unlike the i32 form's isel bug noted in ``ballot_first_n``); see
      https://github.com/llvm/llvm-project/pull/71556 and ``codegen_amdgpu.cpp`` for the full background.
    * SPIR-V: extract components 0 and 1 of the ``OpGroupNonUniformBallot`` uvec4 (lanes 0..31 and 32..63
      respectively) and pack them: ``u64(hi) << 32 | u64(lo)``.

    Caller contract: uniform CF + all lanes active.
    """
    return impl.call_internal("subgroupBallotU64", predicate, with_runtime_context=False)


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

    Caller must ensure ``2**log2_size`` does not exceed the active subgroup size on the target (32 on CUDA / Metal /
    RDNA, 64 on CDNA).  ``log2_size`` is a compile-time template; the body is fully unrolled.
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


# --- Lane masks ------------------------------------------------------------------------
#
# Five trivial ``u32`` constants parametrised by a lane id, mirroring CUDA's ``__lanemask_{lt,le,eq,gt,ge}``.  They are
# portable arithmetic wrappers (no backend intrinsic): every backend that lowers ``shl`` / ``sub`` / ``not`` lowers
# them.  Pass ``invocation_id()`` to get the current lane's mask, or any other integer expression in ``[0, 31]`` to
# query an arbitrary lane's mask.
#
# Caller contract: ``lane_id`` must be in ``[0, 31]`` (matching the ``u32`` return type, which represents 32 lanes).
# Passing ``lane_id >= 32`` triggers an undefined-behaviour shift on most backends.  AMDGPU CDNA wave64 callers can
# build a 64-bit mask from two ``u32`` ballots if they need lanes 32..63.


@func
def lanemask_lt(lane_id):
    """Bitmask of lanes strictly below ``lane_id`` — bit ``i`` is set iff ``i < lane_id``.

    Pass ``invocation_id()`` for the classic CUDA ``__lanemask_lt()`` (current lane's mask).  Equivalent to
    ``(u32(1) << u32(lane_id)) - u32(1)``; inlined at trace time into 1 shift + 1 subtract.

    See the module-level note for the ``lane_id ∈ [0, 31]`` contract and the AMDGPU CDNA wave64 caveat.
    """
    return (u32(1) << u32(lane_id)) - u32(1)


@func
def lanemask_le(lane_id):
    """Bitmask of lanes ``<= lane_id`` — bit ``i`` is set iff ``i <= lane_id``.

    Equivalent to ``lanemask_lt(lane_id) | lanemask_eq(lane_id)``.  See `lanemask_lt` for the contract.
    """
    one_at_lane = u32(1) << u32(lane_id)
    return one_at_lane | (one_at_lane - u32(1))


@func
def lanemask_eq(lane_id):
    """Bitmask with exactly one bit set at ``lane_id`` — equivalent to ``u32(1) << u32(lane_id)``.

    See `lanemask_lt` for the contract.
    """
    return u32(1) << u32(lane_id)


@func
def lanemask_gt(lane_id):
    """Bitmask of lanes strictly above ``lane_id`` — bit ``i`` is set iff ``i > lane_id``.

    Equivalent to ``~lanemask_le(lane_id)``.  See `lanemask_lt` for the contract.
    """
    one_at_lane = u32(1) << u32(lane_id)
    return ~(one_at_lane | (one_at_lane - u32(1)))


@func
def lanemask_ge(lane_id):
    """Bitmask of lanes ``>= lane_id`` — bit ``i`` is set iff ``i >= lane_id``.

    Equivalent to ``~lanemask_lt(lane_id)``.  See `lanemask_lt` for the contract.
    """
    return ~((u32(1) << u32(lane_id)) - u32(1))


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


@func
def reduce_min(value, log2_size: template()):
    """Min of ``value`` across ``2**log2_size`` consecutive lanes via a ``shuffle_down`` tree.

    The result is valid in lane 0 of each ``2**log2_size`` group; other lanes hold partial mins.  Caller must ensure
    ``2**log2_size`` does not exceed the active subgroup size on the target (32 on CUDA / Metal / RDNA, 64 on CDNA).

    ``log2_size`` is a compile-time template; the body is fully unrolled into ``log2_size`` shuffle+min operations in
    the calling kernel's IR.

    Float NaN handling is implementation-defined: ``qd.min`` lowers to a backend-specific intrinsic (``fminnm`` on PTX,
    ``llvm.minnum`` on AMDGPU, ``OpFMin`` on SPIR-V) and these differ on whether NaN propagates or is suppressed.
    Avoid NaN inputs if you need a portable result.
    """
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << (log2_size - 1 - i))
        value = min(value, shuffle_down(value, u32(offset)))
    return value


@func
def reduce_max(value, log2_size: template()):
    """Max of ``value`` across ``2**log2_size`` consecutive lanes via a ``shuffle_down`` tree.

    See `reduce_min` for the size contract, the unrolling shape, and the NaN caveat (with ``qd.max`` in place of
    ``qd.min``).  The result is valid in lane 0 of each group; other lanes hold partial maxes.
    """
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << (log2_size - 1 - i))
        value = max(value, shuffle_down(value, u32(offset)))
    return value


@func
def reduce_all_min(value, log2_size: template()):
    """Min of ``value`` across ``2**log2_size`` consecutive lanes via a butterfly XOR.

    The result is broadcast to all ``2**log2_size`` lanes.  Same size contract, unrolling shape, and NaN caveat as
    `reduce_min`.  Use this when every lane needs the reduction (e.g. to subtract the min, or to branch on it
    uniformly): same shuffle count as `reduce_min`, no extra broadcast needed.
    """
    lane = invocation_id()
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        value = min(value, shuffle(value, u32(lane ^ mask)))
    return value


@func
def reduce_all_max(value, log2_size: template()):
    """Max of ``value`` across ``2**log2_size`` consecutive lanes via a butterfly XOR.

    The result is broadcast to all ``2**log2_size`` lanes.  See `reduce_all_min` (with ``qd.max``
    in place of ``qd.min``).
    """
    lane = invocation_id()
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        value = max(value, shuffle(value, u32(lane ^ mask)))
    return value


# reduce_mul / reduce_and / reduce_or / reduce_xor (no-arg, SPIR-V-only) have been removed.  Build
# sized portable replacements on top of `shuffle_down` / `shuffle` following the same pattern as
# `reduce_add` / `reduce_all_add` above when needed.


# --- Segmented reduce ------------------------------------------------------------------
#
# `segmented_reduce_{add,min,max}(value, head_flag, log2_size)` runs a per-lane inclusive scan under ``+`` / ``min`` /
# ``max`` where every lane with ``head_flag != 0`` resets the running aggregate: lane ``i`` ends up holding the scan of
# ``value[head_below..i+1]``, where ``head_below`` is the largest lane ``<= i`` (within the lane's ``2**log2_size``
# group) whose ``head_flag`` is non-zero.  If no such lane exists the algorithm treats the group's first lane as an
# implicit head, so a segment that runs from ``group_base`` to the lane is still aggregated correctly.
#
# Implementation: one ``ballot_full_subgroup`` to materialise a u64 of head positions across the whole subgroup, then
# a Hillis-Steele inclusive scan bounded by ``distance >= offset`` (where ``distance = lane - segment_head``).
# ``segment_head`` comes from ``31 - clz(effective_mask & ((1 << (lane + 1)) - 1))`` with an OR-injected virtual head
# at ``group_base`` to guarantee a non-zero ``lower``.  We work in half-local 32-lane coordinates so the bit-mask
# arithmetic stays in u32 even on wave64; ``2**log2_size <= 32`` guarantees segments never cross a half boundary, so
# half-local distance equals absolute distance and the downstream ``shuffle_up`` partners stay in-half.  Cost: 1
# ballot + 1 clz + ``log2_size`` shuffles + ``log2_size`` ops — the same shape as `inclusive_add` / `inclusive_min` /
# `inclusive_max`, plus a single-instruction setup.
#
# No identity argument is required (unlike `exclusive_min` / `exclusive_max`) because the per-lane ``distance >=
# offset`` guard ensures the scan never reaches across a segment boundary, so a partner from another segment is never
# combined with the local value.


@func
def _segment_head_distance(head_flag, log2_size: template()):
    """Compute ``lane - segment_head`` — how many lanes the current lane sits past its segment head, scoped to
    ``2**log2_size`` lanes.  Returns ``0`` at the segment head, ``1, 2, ...`` for later lanes within the segment.

    Shared by `segmented_reduce_add` / `_min` / `_max`; see the module-level note for the algorithm.

    Wave32 vs wave64: the documented contract is ``2**log2_size <= 32``, so segments never cross a 32-lane boundary.
    But on wave64 lanes 32..63 still execute this helper and need correct results within their own group.  We pull
    a u64 ``ballot_full_subgroup`` mask, shift the relevant 32-lane half down to bits 0..31, and run the rest of the
    bit-mask algorithm in half-local lane coordinates (``lane_in_half = lane - half_base``).  Half-local ``distance``
    equals absolute ``distance`` because both ``lane`` and the recovered ``segment_head`` are offset by the same
    ``half_base``, so `segmented_reduce_*`'s downstream ``distance >= offset`` guard still works in absolute terms.
    On wave32 this collapses to the original code (high 32 bits of the u64 are always zero, ``half_base == 0`` always).
    """
    # u64 mask covering the entire subgroup.  Wave32: high 32 bits are zero by definition.  Wave64: all 64 bits are
    # meaningful and we need both halves to handle lanes 32..63 correctly.
    full_mask = ballot_full_subgroup(i32(head_flag != 0))
    lane = invocation_id()
    # Which 32-lane half this lane belongs to: 0 on wave32 (always), 0 or 32 on wave64.  ``& ~31`` rounds down to a
    # multiple of 32.
    half_base = u32(lane) & u32(~31 & 0xFFFFFFFF)
    # Slide the relevant 32 bits down to bits 0..31 of a u32.  ``head_mask >> half_base`` discards the other half;
    # the truncating cast to u32 then drops everything above bit 31.  On wave32 ``half_base`` is always 0 and the
    # high 32 bits of ``full_mask`` are zero, so this is a no-op shift + truncate (LLVM folds it).
    head_mask = u32(full_mask >> u64(half_base))
    # Half-local lane index — always in ``[0, 32)``, so all the u32 shifts below are well-defined.
    lane_in_half = i32(lane) - i32(half_base)
    group_base = u32(lane_in_half) & u32(impl.static(~((1 << log2_size) - 1) & 0xFFFFFFFF))
    bits_in_group = u32(impl.static((2 ** (1 << log2_size) - 1) & 0xFFFFFFFF))
    group_mask = bits_in_group << group_base
    effective_mask = (head_mask & group_mask) | (u32(1) << group_base)
    inclusive_mask = u32(0xFFFFFFFF) >> u32(i32(31) - lane_in_half)
    lower = effective_mask & inclusive_mask
    # `clz` follows the input type on most backends (CUDA explicitly normalizes to i32; AMDGPU LLVM `ctlz` returns
    # input type; SPIR-V FindUMsb returns the input type).  Wrap the result in `i32(...)` so the subsequent arithmetic
    # is uniformly signed-32-bit and SPIR-V's strict-type `sub` is happy.
    segment_head = i32(31) - i32(clz(lower))
    # ``segment_head`` is in half-local coords; ``lane_in_half - segment_head`` is the same as absolute
    # ``lane - segment_head_abs`` (both shifted by ``half_base``), so the distance returned here is consistent
    # with what ``segmented_reduce_*`` callers expect for the ``distance >= offset`` shuffle-up bound.
    return lane_in_half - segment_head


@func
def segmented_reduce_add(value, head_flag, log2_size: template()):
    """Per-lane inclusive sum that resets at every non-zero ``head_flag``, scoped to ``2**log2_size`` lanes.

    Lane ``i`` returns ``sum(value[head_below..i+1])``, where ``head_below`` is the largest lane index ``<= i`` (within
    the lane's ``2**log2_size`` group) whose ``head_flag`` is non-zero.  If no such lane exists inside the group, the
    group's first lane is treated as an implicit head, so the result is the inclusive sum from ``group_base`` to ``i``.

    Caller contract:

    * ``2**log2_size`` must not exceed 32: segments are scoped to a single 32-lane half of the subgroup so the
      bit-mask bookkeeping stays in u32 internally.  On wave64 the high 32 lanes are handled correctly via a separate
      half-local ballot — see ``_segment_head_distance`` for the wave64 details.  ``log2_size`` is a ``qd.template()``
      compile-time constant; the body is fully unrolled into ``log2_size`` ``shuffle_up + add`` pairs.
    * ``head_flag`` is any integer scalar; the lowering tests ``head_flag != 0``, so non-binary truthy values work.
    * Same uniform-CF + all-lanes-active contract as the rest of ``qd.simt.subgroup``.
    """
    distance = _segment_head_distance(head_flag, log2_size)
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << i)
        partner = shuffle_up(value, u32(offset))
        if distance >= offset:
            value = value + partner
    return value


@func
def segmented_reduce_min(value, head_flag, log2_size: template()):
    """Per-lane inclusive min that resets at every non-zero ``head_flag``, scoped to ``2**log2_size`` lanes.

    Lane ``i`` returns ``min(value[head_below..i+1])``; see `segmented_reduce_add` for the head-flag semantics, the
    ``2**log2_size <= 32`` cap, and the truthy-predicate / uniform-CF contract.

    Float NaN handling is implementation-defined: ``qd.min`` lowers to a backend-specific intrinsic (``fminnm`` on
    PTX, ``llvm.minnum`` on AMDGPU, ``OpFMin`` on SPIR-V) and these differ on whether NaN propagates or is suppressed.
    Avoid NaN inputs if you need a portable result.
    """
    distance = _segment_head_distance(head_flag, log2_size)
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << i)
        partner = shuffle_up(value, u32(offset))
        if distance >= offset:
            value = min(value, partner)
    return value


@func
def segmented_reduce_max(value, head_flag, log2_size: template()):
    """Per-lane inclusive max that resets at every non-zero ``head_flag``, scoped to ``2**log2_size`` lanes.

    See `segmented_reduce_min` (with ``qd.max`` in place of ``qd.min``).  Same head-flag semantics, ``2**log2_size <=
    32`` cap, truthy-predicate / uniform-CF contract, and NaN caveat.
    """
    distance = _segment_head_distance(head_flag, log2_size)
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << i)
        partner = shuffle_up(value, u32(offset))
        if distance >= offset:
            value = max(value, partner)
    return value


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
    one operand is an f32 produced by a shuffle intrinsic—the select silently returns the false-branch value regardless
    of the condition.  The ``if`` form works correctly for the inclusive scan on its own; callers that issue further
    subgroup ops after this scan (e.g. `_exclusive_scan`) must insert a ``sync()`` barrier to force reconvergence
    before the next shuffle.
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
    Caller must ensure ``2**log2_size`` does not exceed the active subgroup size on the target (32 on CUDA / Metal /
    RDNA, 64 on CDNA).
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
# Each `exclusive_*` runs the inclusive scan, then shifts the result up by one lane via `shuffle_up(inc, 1)` and
# replaces lane 0 of every group with the operator's identity.  Lane 0's result must be set explicitly because
# `shuffle_up` with offset 1 returns an implementation-defined value at lane 0 (and `OpGroupNonUniformShuffleUp` calls
# it undefined outright).  See `_exclusive_scan` for the shared body.
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
    "ballot_first_n",
    "ballot_full_subgroup",
    "all_true",
    "any_true",
    "all_equal",
    "broadcast_first",
    "lanemask_lt",
    "lanemask_le",
    "lanemask_eq",
    "lanemask_gt",
    "lanemask_ge",
    "reduce_add",
    "reduce_all_add",
    "reduce_min",
    "reduce_max",
    "reduce_all_min",
    "reduce_all_max",
    "segmented_reduce_add",
    "segmented_reduce_min",
    "segmented_reduce_max",
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
