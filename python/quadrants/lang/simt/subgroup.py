# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false

import warnings

import numpy as np

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.expr import make_constant_expr
from quadrants.lang.kernel_impl import func
from quadrants.lang.ops import clz
from quadrants.lang.util import to_numpy_type
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
    ``elect()`` at zero extra cost (it inlines at compile time into a single compare + zext).

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
    callers who want the full subgroup), use `ballot` and check the high 32 bits.
    """
    if impl.static(n == 32):
        return impl.call_internal("subgroupBallotU32", predicate, with_runtime_context=False)
    lane = invocation_id()
    masked = predicate * i32(u32(lane) < u32(impl.static(n)))
    return impl.call_internal("subgroupBallotU32", masked, with_runtime_context=False)


def ballot(predicate):
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


_ballot_full_subgroup_deprecation_warned = False


def ballot_full_subgroup(predicate):
    """Deprecated alias for :func:`ballot`.

    Emits a ``DeprecationWarning`` on first use and forwards to :func:`ballot`.  Will be removed in a future
    release; rename call sites to ``ballot(predicate)``.  Full-subgroup ops are unsuffixed throughout this module
    (``reduce_add`` / ``all_true`` / ``inclusive_max`` / etc.); tiled variants take a ``_tiled`` suffix and an extra
    ``log2_size`` template parameter.
    """
    global _ballot_full_subgroup_deprecation_warned
    if not _ballot_full_subgroup_deprecation_warned:
        _ballot_full_subgroup_deprecation_warned = True
        warnings.warn(
            "qd.simt.subgroup.ballot_full_subgroup() is deprecated; use qd.simt.subgroup.ballot() instead "
            "(full-subgroup ops are unsuffixed; tiled forms use _tiled, e.g. reduce_add_tiled).",
            DeprecationWarning,
            stacklevel=2,
        )
    return ballot(predicate)


# --- Voting / predicate ops ------------------------------------------------------------
#
# All three are group-scoped over ``2**log2_size`` consecutive lanes, mirror the API of ``reduce_all_add_tiled`` /
# ``inclusive_*`` / ``exclusive_*``, and broadcast the result to every lane in the group as an ``i32`` (``0`` or ``1``).
#
# Backend strategy
# ----------------
# * On CUDA, when ``log2_size == 5`` (full warp), ``all_true_tiled`` / ``any_true_tiled`` lower to
#   ``__all_sync(0xFFFFFFFF, p)`` / ``__any_sync(0xFFFFFFFF, p)`` (one ``vote.all`` / ``vote.any`` instruction).  This
#   shortcut is selected at compile time via ``static()`` on ``impl.current_cfg().arch`` and on the compile-time
#   ``log2_size`` template, so it collapses to a single intrinsic call in the IR with no overhead vs. handwritten CUDA.
# * Every other backend, and CUDA for partial-warp groups, uses a portable ``shuffle_xor`` butterfly: ``log2_size``
#   shuffles + ``log2_size`` ANDs / ORs, fully unrolled into the calling kernel's IR.  Same shape as
#   ``reduce_all_add_tiled``.
# * ``all_equal_tiled`` is always implemented as ``all_true_tiled(value == broadcast_group_lane_0)``, so it inherits
#   the CUDA shortcut transitively.  We don't reach for ``__match_all_sync`` because (a) it requires sm_70+, (b) it does
#   bit-equality on floats, contradicting the SPIR-V ``OpGroupNonUniformAllEqual`` semantics this op advertises
#   (``NaN != NaN``, ``+0.0 == -0.0``), and (c) the broadcast-then-AND form is only one shuffle slower while staying on
#   the same code path everywhere.


# --- Full-subgroup wrappers ------------------------------------------------------------
#
# Every wrapper below is a plain Python function (not ``@qd.func``) so that ``log2_group_size()`` is evaluated in the
# kernel's Python tracing pass and feeds into the underlying ``*_tiled`` op's ``log2_size: template()`` argument as a
# compile-time ``int``.  Each wrapper compiles down to exactly the same IR as a hand-written call site that hard-codes
# ``log2_size=5`` (CUDA / Metal / Vulkan-wave32) or ``log2_size=6`` (AMDGPU wave64), so there is no runtime overhead vs
# calling the underlying ``*_tiled`` op directly.  These are the default form for "operate over the entire subgroup"
# portably without branching on ``group_size()`` -- the common case for whole-warp reductions, broadcasts, and votes.
# Reach for the ``*_tiled`` form when you want multiple independent tiles per subgroup (e.g.
# ``reduce_add_tiled(v, 4)`` to fold every 16 lanes into one).


@func
def all_true_tiled(predicate, log2_size: template()):
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
def any_true_tiled(predicate, log2_size: template()):
    """OR-reduce ``predicate != 0`` across ``2**log2_size`` consecutive lanes.  Returns ``1`` (``i32``) on every lane
    of the group iff at least one lane in the group has a non-zero ``predicate``, else ``0``.

    See `all_true_tiled` for the size contract.
    """
    p = i32(predicate != 0)
    if impl.static(impl.current_cfg().arch == _qd_core.cuda and log2_size == 5):
        return impl.call_internal("cuda_any_sync_i32", u32(0xFFFFFFFF), p, with_runtime_context=False)
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        p = p | shuffle_xor(p, u32(mask))
    return p


@func
def all_equal_tiled(value, log2_size: template()):
    """Return ``1`` (``i32``) on every lane in each ``2**log2_size`` group iff every lane in the group has the same
    ``value``, else ``0``.

    Equality is the backend's native ``==`` on ``value``'s dtype: for floats this means ``NaN != NaN`` (a group with
    any ``NaN`` returns ``0``) and ``+0.0 == -0.0``, matching SPIR-V ``OpGroupNonUniformAllEqual``.  Callers wanting
    bit-equality on floats should bit-cast to the same-width integer dtype before calling.

    Implementation: each lane reads the value at the start of its group via ``shuffle``, then ``all_true_tiled``
    AND-reduces the per-lane equality bit.  Cost: one ``shuffle`` plus one ``all_true_tiled`` (one ``vote.all`` on
    CUDA at ``log2_size == 5``, otherwise a ``log2_size``-deep ``shuffle_xor`` butterfly).
    """
    lane = invocation_id()
    group_base = u32(lane) & u32(~((1 << log2_size) - 1) & 0xFFFFFFFF)
    base = shuffle(value, group_base)
    return all_true_tiled(i32(value == base), log2_size)


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
    ``(u32(1) << u32(lane_id)) - u32(1)``; inlined at compile time into 1 shift + 1 subtract.

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


def group_size() -> int:
    """Active subgroup size for the current launch, as a Python ``int``.

    Resolves once at compile time by querying the live ``Program`` --- 32 on CUDA, 64 on AMDGPU (every AMDGPU target is
    pinned to ``+wavefrontsize64``), and the device-probed value on the SPIR-V backends (read from
    ``VkPhysicalDeviceSubgroupProperties::subgroupSize`` on Vulkan, fixed at 32 on Metal). Because the return type is a
    plain ``int``, the value can be used as a ``qd.template()`` argument inside ``@qd.kernel`` / ``@qd.func`` bodies ---
    this is how the full-subgroup reductions (e.g. ``reduce_add(v)``) pick up the right ``log2_size`` per backend
    without the caller having to plumb it manually.

    For use inside ``@qd.kernel`` / ``@qd.func`` bodies: the value is folded into the kernel IR as a constant on every
    backend, including SPIR-V (so MoltenVK / desktop Vulkan see a literal subgroup size rather than a runtime
    ``OpLoad`` of ``BuiltInSubgroupSize``). Calling it from plain host Python after ``qd.init()`` is also legal and
    returns the same number, which is handy for setting up grid dimensions on the host side.
    """
    return impl.get_runtime().prog.subgroup_size()


def log2_group_size() -> int:
    """``log2(group_size())`` as a Python ``int``, asserting the subgroup size is a power of two.

    Equivalent to ``int(math.log2(group_size()))`` but emits a clearer error if the device ever reports a
    non-power-of-two subgroup width (no current SPIR-V driver does, but the spec allows it). Like ``group_size()`` this
    is a compile-time constant on every backend --- callers feed it straight into ``qd.template()`` to pick the right
    ``log2_size`` for a full-subgroup reduction (e.g. ``reduce_add_tiled(v, qd.simt.subgroup.log2_group_size())``).
    """
    size = group_size()
    assert size > 0 and (size & (size - 1)) == 0, f"subgroup size {size} is not a power of two"
    return size.bit_length() - 1


def invocation_id():
    return impl.call_internal("subgroupInvocationId", with_runtime_context=False)


@func
def _reduce_tiled(value, op: template(), log2_size: template()):
    """Tree-reduce ``value`` across ``2**log2_size`` consecutive lanes via ``shuffle_down`` under a caller-supplied
    binary ``op``.  Mirrors the operator-specialized public ``reduce_add_tiled`` / ``reduce_min_tiled`` /
    ``reduce_max_tiled`` but takes a template operator so cross-module callers (currently ``block.reduce`` and the
    typed ``block.reduce_{add,min,max}``) can compose the per-subgroup step with custom monoids without reimplementing
    the shuffle tree.

    Result is valid in lane 0 of each ``2**log2_size`` group; other lanes hold partial values.  ``log2_size`` is a
    compile-time template, so the body unrolls into ``log2_size`` shuffle+op pairs.  Caller must ensure
    ``2**log2_size`` does not exceed the active subgroup size on the target.

    Underscore-prefixed because the generic-op contract is fragile (``op`` must be associative and side-effect-free)
    and we don't want to invite ad-hoc subgroup-scope reductions from arbitrary kernels; the typed
    ``reduce_{add,min,max}_tiled`` cover the common cases.
    """
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << (log2_size - 1 - i))
        value = op(value, shuffle_down(value, u32(offset)))
    return value


@func
def reduce_add_tiled(value, log2_size: template()):
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
def reduce_all_add_tiled(value, log2_size: template()):
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
def reduce_min_tiled(value, log2_size: template()):
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
def reduce_max_tiled(value, log2_size: template()):
    """Max of ``value`` across ``2**log2_size`` consecutive lanes via a ``shuffle_down`` tree.

    See `reduce_min_tiled` for the size contract, the unrolling shape, and the NaN caveat (with ``qd.max`` in place of
    ``qd.min``).  The result is valid in lane 0 of each group; other lanes hold partial maxes.
    """
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << (log2_size - 1 - i))
        value = max(value, shuffle_down(value, u32(offset)))
    return value


@func
def reduce_all_min_tiled(value, log2_size: template()):
    """Min of ``value`` across ``2**log2_size`` consecutive lanes via a butterfly XOR.

    The result is broadcast to all ``2**log2_size`` lanes.  Same size contract, unrolling shape, and NaN caveat as
    `reduce_min_tiled`.  Use this when every lane needs the reduction (e.g. to subtract the min, or to branch on it
    uniformly): same shuffle count as `reduce_min_tiled`, no extra broadcast needed.
    """
    lane = invocation_id()
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        value = min(value, shuffle(value, u32(lane ^ mask)))
    return value


@func
def reduce_all_max_tiled(value, log2_size: template()):
    """Max of ``value`` across ``2**log2_size`` consecutive lanes via a butterfly XOR.

    The result is broadcast to all ``2**log2_size`` lanes.  See `reduce_all_min_tiled` (with ``qd.max``
    in place of ``qd.min``).
    """
    lane = invocation_id()
    for i in impl.static(range(log2_size)):
        mask = impl.static(1 << i)
        value = max(value, shuffle(value, u32(lane ^ mask)))
    return value


# reduce_mul / reduce_and / reduce_or / reduce_xor (no-arg, SPIR-V-only) have been removed.  Build
# sized portable replacements on top of `shuffle_down` / `shuffle` following the same pattern as
# `reduce_add_tiled` / `reduce_all_add_tiled` above when needed.


# --- Segmented reduce ------------------------------------------------------------------
#
# `segmented_reduce_{add,min,max}(value, head_flag, log2_size)` runs a per-lane inclusive scan under ``+`` / ``min`` /
# ``max`` where every lane with ``head_flag != 0`` resets the running aggregate: lane ``i`` ends up holding the scan of
# ``value[head_below..i+1]``, where ``head_below`` is the largest lane ``<= i`` (within the lane's ``2**log2_size``
# group) whose ``head_flag`` is non-zero.  If no such lane exists the algorithm treats the group's first lane as an
# implicit head, so a segment that runs from ``group_base`` to the lane is still aggregated correctly.
#
# Implementation: one ``ballot`` to materialise a u64 of head positions across the whole subgroup, then
# a Hillis-Steele inclusive scan bounded by ``distance >= offset`` (where ``distance = lane - segment_head``).
# ``segment_head`` comes from ``31 - clz(effective_mask & ((1 << (lane + 1)) - 1))`` with an OR-injected virtual head
# at ``group_base`` to guarantee a non-zero ``lower``.  We work in half-local 32-lane coordinates so the bit-mask
# arithmetic stays in u32 even on wave64; ``2**log2_size <= 32`` guarantees segments never cross a half boundary, so
# half-local distance equals absolute distance and the downstream ``shuffle_up`` partners stay in-half.  Cost: 1
# ballot + 1 clz + ``log2_size`` shuffles + ``log2_size`` ops — the same shape as `inclusive_add_tiled` /
# `inclusive_min_tiled` / `inclusive_max_tiled`, plus a single-instruction setup.
#
# No identity element is involved at all -- the per-lane ``distance >= offset`` guard ensures the scan never reaches
# across a segment boundary, so a partner from another segment is never combined with the local value (i.e. the
# implementation doesn't need a "what to combine with at the segment head" sentinel the way ``exclusive_min`` /
# ``exclusive_max`` do for lane 0 within each tile).


@func
def _segment_head_distance_tiled(head_flag, log2_size: template()):
    """Compute ``lane - segment_head`` — how many lanes the current lane sits past its segment head, scoped to
    ``2**log2_size`` lanes.  Returns ``0`` at the segment head, ``1, 2, ...`` for later lanes within the segment.

    Shared by `segmented_reduce_add_tiled` / `_min` / `_max`; see the module-level note for the algorithm.

    Two compile-time-selected paths:

    * **``log2_size <= 5`` (segments fit in a single 32-lane half)** — u32-bitmask path.  Pulls a u64
      ``ballot``, shifts the relevant 32-lane half down to bits 0..31, then runs the bit-mask algorithm in
      half-local lane coordinates.  Half-local ``distance`` equals absolute ``distance`` because both ``lane`` and the
      recovered ``segment_head`` are offset by the same ``half_base``, so the downstream ``distance >= offset`` guard
      still works in absolute terms.  On wave32 this collapses to the no-op (``half_base == 0`` always); on wave64
      lanes 32..63 see their own half-mask via the shift.  This is the only path on CUDA / Metal / Vulkan-wave32 (and
      compiles to identical IR to the historical wave32-only impl), so backends with ``group_size() == 32`` see no perf
      regression from supporting wave64 here.

    * **``log2_size == 6`` (full-wave64 segments, only reachable on AMDGPU)** — u64-bitmask path.  Segments span all 64
      lanes so we need the full ``ballot`` u64 mask and a ``clz(u64)``.  Costs one extra ``u64`` shift +
      ``u64`` clz vs the u32 path but avoids the half-local split and stays in absolute lane coordinates throughout.
      Gated by ``impl.static(log2_size <= 5)`` so this entire path is dead-code-eliminated at compile time on every
      ``log2_size <= 5`` call site, including on AMDGPU itself when callers stay under the wave-half boundary.
    """
    # u64 mask covering the entire subgroup.  Wave32: high 32 bits are zero by definition.  Wave64: all 64 bits are
    # meaningful and we need both halves to handle lanes 32..63 correctly.
    full_mask = ballot(i32(head_flag != 0))
    lane = invocation_id()
    if impl.static(log2_size <= 5):
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
        # input type; SPIR-V FindUMsb returns the input type).  Wrap the result in `i32(...)` so the subsequent
        # arithmetic is uniformly signed-32-bit and SPIR-V's strict-type `sub` is happy.
        segment_head = i32(31) - i32(clz(lower))
        # ``segment_head`` is in half-local coords; ``lane_in_half - segment_head`` is the same as absolute
        # ``lane - segment_head_abs`` (both shifted by ``half_base``), so the distance returned here is consistent
        # with what ``segmented_reduce_*`` callers expect for the ``distance >= offset`` shuffle-up bound.
        return lane_in_half - segment_head
    # ``log2_size == 6`` (full wave64). Skip the half-local split and work in absolute lane coordinates with a u64
    # bitmask.  The implicit-head bit is injected at the segment's group base (always 0 on the only
    # wave64-and-log2_size-6 configuration), and we use ``clz(u64)`` for the segment head.  Quadrants pins AMDGPU to
    # wave64 so this branch is only reachable when ``group_size() == 64``; on every other backend ``log2_size == 6``
    # violates the documented caller contract (``2**log2_size <= group_size()``) — checked by the ``static_assert`` in
    # ``segmented_reduce_*``.
    bits_in_group = u64(impl.static(((1 << (1 << log2_size)) - 1) & 0xFFFFFFFFFFFFFFFF))
    group_base = u64(lane) & u64(impl.static(~((1 << log2_size) - 1) & 0xFFFFFFFFFFFFFFFF))
    group_mask = bits_in_group << group_base
    effective_mask = (full_mask & group_mask) | (u64(1) << group_base)
    # ``lane + 1 <= 64`` on wave64; ``63 - lane`` is in ``[0, 63]`` so the u64 shift is well-defined and produces
    # all-1s in the bottom ``lane + 1`` bits.
    inclusive_mask = u64(0xFFFFFFFFFFFFFFFF) >> u64(i32(63) - i32(lane))
    lower = effective_mask & inclusive_mask
    segment_head = i32(63) - i32(clz(lower))
    return i32(lane) - segment_head


@func
def segmented_reduce_add_tiled(value, head_flag, log2_size: template()):
    """Per-lane inclusive sum that resets at every non-zero ``head_flag``, scoped to ``2**log2_size`` lanes.

    Lane ``i`` returns ``sum(value[head_below..i+1])``, where ``head_below`` is the largest lane index ``<= i`` (within
    the lane's ``2**log2_size`` group) whose ``head_flag`` is non-zero.  If no such lane exists inside the group, the
    group's first lane is treated as an implicit head, so the result is the inclusive sum from ``group_base`` to ``i``.

    Caller contract:

    * ``2**log2_size`` must not exceed the active subgroup size: up to 32 on CUDA / Metal / Vulkan-wave32, up to 64 on
      AMDGPU (wave64).  ``log2_size`` is a ``qd.template()`` compile-time constant; the body is fully unrolled into
      ``log2_size`` ``shuffle_up + add`` pairs.  ``_segment_head_distance_tiled`` selects between a u32-bitmask path
      (``log2_size <= 5``, identical IR to the historical wave32-only impl) and a u64-bitmask path
      (``log2_size == 6``, AMDGPU-only) at compile time, so CUDA / SPIR-V callers see zero overhead from the wave64
      support.
    * ``head_flag`` is any integer scalar; the lowering tests ``head_flag != 0``, so non-binary truthy values work.
    * Same uniform-CF + all-lanes-active contract as the rest of ``qd.simt.subgroup``.
    """
    impl.static_assert(log2_size <= 6, "segmented_reduce_add_tiled requires log2_size <= 6")
    distance = _segment_head_distance_tiled(head_flag, log2_size)
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << i)
        partner = shuffle_up(value, u32(offset))
        if distance >= offset:
            value = value + partner
    return value


@func
def segmented_reduce_min_tiled(value, head_flag, log2_size: template()):
    """Per-lane inclusive min that resets at every non-zero ``head_flag``, scoped to ``2**log2_size`` lanes.

    Lane ``i`` returns ``min(value[head_below..i+1])``; see `segmented_reduce_add_tiled` for the head-flag semantics,
    the ``2**log2_size <= group_size()`` cap, and the truthy-predicate / uniform-CF contract.

    Float NaN handling is implementation-defined: ``qd.min`` lowers to a backend-specific intrinsic (``fminnm`` on
    PTX, ``llvm.minnum`` on AMDGPU, ``OpFMin`` on SPIR-V) and these differ on whether NaN propagates or is suppressed.
    Avoid NaN inputs if you need a portable result.
    """
    impl.static_assert(log2_size <= 6, "segmented_reduce_min_tiled requires log2_size <= 6")
    distance = _segment_head_distance_tiled(head_flag, log2_size)
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << i)
        partner = shuffle_up(value, u32(offset))
        if distance >= offset:
            value = min(value, partner)
    return value


@func
def segmented_reduce_max_tiled(value, head_flag, log2_size: template()):
    """Per-lane inclusive max that resets at every non-zero ``head_flag``, scoped to ``2**log2_size`` lanes.

    See `segmented_reduce_min_tiled` (with ``qd.max`` in place of ``qd.min``).  Same head-flag semantics,
    ``2**log2_size <= group_size()`` cap, truthy-predicate / uniform-CF contract, and NaN caveat.
    """
    impl.static_assert(log2_size <= 6, "segmented_reduce_max_tiled requires log2_size <= 6")
    distance = _segment_head_distance_tiled(head_flag, log2_size)
    for i in impl.static(range(log2_size)):
        offset = impl.static(1 << i)
        partner = shuffle_up(value, u32(offset))
        if distance >= offset:
            value = max(value, partner)
    return value


# --- Inclusive scans -------------------------------------------------------------------
#
# All seven inclusive scans share the same Hillis-Steele tree over `shuffle_up`; only the binary operator differs.
# Each operator is a tiny ``@func`` so it can be passed as a ``template()`` callable to the shared
# `_inclusive_scan_tiled` helper, which inlines it into the per-lane reduce step.  ``log2_size`` is a compile-time
# constant so the loop fully unrolls into ``log2_size`` shuffle+op pairs in the calling kernel's IR.


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
def _inclusive_scan_tiled(value, op: template(), log2_size: template()):
    """Hillis-Steele inclusive scan of ``value`` under binary ``op``, over ``2**log2_size`` consecutive lanes.  See
    `inclusive_add_tiled` for the contract; the only thing that changes between the seven `inclusive_*` ops is which
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
def inclusive_add_tiled(value, log2_size: template()):
    """Inclusive prefix sum across ``2**log2_size`` consecutive lanes.

    Lane ``i`` within each group of ``2**log2_size`` lanes returns ``v[group_start] + v[group_start + 1] + ... + v[i]``.
    Caller must ensure ``2**log2_size`` does not exceed the active subgroup size on the target (32 on CUDA / Metal, 64
    on AMDGPU — wave64 is forced on every AMDGPU target).
    """
    return _inclusive_scan_tiled(value, _bin_add, log2_size)


@func
def inclusive_mul_tiled(value, log2_size: template()):
    """Inclusive prefix product across ``2**log2_size`` consecutive lanes.

    See `inclusive_add_tiled` for the size contract.
    """
    return _inclusive_scan_tiled(value, _bin_mul, log2_size)


@func
def inclusive_min_tiled(value, log2_size: template()):
    """Inclusive prefix min across ``2**log2_size`` consecutive lanes.  See `inclusive_add_tiled` for the size
    contract."""
    return _inclusive_scan_tiled(value, _bin_min, log2_size)


@func
def inclusive_max_tiled(value, log2_size: template()):
    """Inclusive prefix max across ``2**log2_size`` consecutive lanes.  See `inclusive_add_tiled` for the size
    contract."""
    return _inclusive_scan_tiled(value, _bin_max, log2_size)


@func
def inclusive_and_tiled(value, log2_size: template()):
    """Inclusive prefix bitwise-AND across ``2**log2_size`` consecutive lanes.  Integer dtypes only.  See
    `inclusive_add_tiled` for the size contract."""
    return _inclusive_scan_tiled(value, _bin_and, log2_size)


@func
def inclusive_or_tiled(value, log2_size: template()):
    """Inclusive prefix bitwise-OR across ``2**log2_size`` consecutive lanes.  Integer dtypes only.  See
    `inclusive_add_tiled` for the size contract."""
    return _inclusive_scan_tiled(value, _bin_or, log2_size)


@func
def inclusive_xor_tiled(value, log2_size: template()):
    """Inclusive prefix bitwise-XOR across ``2**log2_size`` consecutive lanes.  Integer dtypes only.  See
    `inclusive_add_tiled` for the size contract."""
    return _inclusive_scan_tiled(value, _bin_xor, log2_size)


# --- Exclusive scans -------------------------------------------------------------------
#
# Each `exclusive_*` shifts the *input* right by one lane via `shuffle_up(value, 1)`, seeds lane 0 of every group with
# the operator's identity, and then runs the inclusive scan on the shifted data.  Doing the shuffle first (rather than
# running the inclusive scan and shuffling the result) avoids the MoltenVK / Metal miscompile where the SPIR-V
# compiler misoptimizes the register holding the inclusive-scan result when its only consumer is a shuffle intrinsic.
# Lane 0's result must be set explicitly because `shuffle_up` with offset 1 returns an implementation-defined value at
# lane 0 (`OpGroupNonUniformShuffleUp` calls it undefined outright).  See `_exclusive_scan_tiled` for the shared body.
#
# Identity per op (in `value`'s dtype):
#
#   add: ``value - value``                  (zero; built from arithmetic on ``value`` to inherit its dtype)
#   mul: ``value - value + 1``              (one; the literal +1 takes value's dtype)
#   or:  ``value ^ value``                  (zero; bitwise xor of value with itself)
#   xor: ``value ^ value``                  (zero)
#   and: ``~(value ^ value)``               (all bits set; bitwise not of zero)
#   min: dtype-max constant                 (+inf for float dtypes; ``np.iinfo(dtype).max`` for integer dtypes)
#   max: dtype-min constant                 (-inf for float dtypes; ``np.iinfo(dtype).min`` for integer dtypes)
#
# For add / mul / and / or / xor the identity falls out of pure arithmetic on ``value`` itself, so the body stays
# inside a ``@func`` and the identity is built from typed Exprs.  For min and max there is no such trick (you can't
# manufacture ``+inf`` or ``INT_MAX`` from arithmetic on a single value of unknown dtype), so ``exclusive_min_tiled``
# / ``exclusive_max_tiled`` are plain Python wrappers that introspect ``value``'s dtype at compile time and emit a
# typed-constant identity Expr.


@func
def _exclusive_scan_tiled(value, op: template(), identity, log2_size: template()):
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
    return _inclusive_scan_tiled(prev, op, log2_size)


@func
def exclusive_add_tiled(value, log2_size: template()):
    """Exclusive prefix sum across ``2**log2_size`` consecutive lanes.

    Lane ``i`` (with ``i > 0``) within each group of ``2**log2_size`` lanes returns ``v[group_start] +
    v[group_start + 1] + ... + v[i - 1]``.  Lane 0 of each group returns the additive identity (zero, in ``value``'s
    dtype).
    """
    return _exclusive_scan_tiled(value, _bin_add, value - value, log2_size)


@func
def exclusive_mul_tiled(value, log2_size: template()):
    """Exclusive prefix product across ``2**log2_size`` consecutive lanes.  Lane 0 of each group returns the
    multiplicative identity (one, in ``value``'s dtype)."""
    return _exclusive_scan_tiled(value, _bin_mul, value - value + 1, log2_size)


def _typed_min_identity(value):
    """Return a typed-constant Expr equal to the largest value representable in ``value``'s dtype.

    Suitable as the identity for an ``exclusive_min`` scan -- lane 0's "predecessor" must be guaranteed ``>=`` every
    real element, and ``+inf`` / ``INT_MAX`` / ``UINT_MAX`` are the tightest such sentinels per dtype.
    """
    dtype = value.ptr.get_rvalue_type()
    if _qd_core.is_real(dtype):
        return make_constant_expr(float("inf"), dtype)
    npty = to_numpy_type(dtype)
    if npty is np.bool_:
        return make_constant_expr(1, dtype)
    assert issubclass(npty, np.integer)
    return make_constant_expr(int(np.iinfo(npty).max), dtype)


def _typed_max_identity(value):
    """Return a typed-constant Expr equal to the smallest value representable in ``value``'s dtype.

    Suitable as the identity for an ``exclusive_max`` scan -- ``-inf`` for floats, ``INT_MIN`` for signed ints, ``0``
    for unsigned ints and bool.
    """
    dtype = value.ptr.get_rvalue_type()
    if _qd_core.is_real(dtype):
        return make_constant_expr(float("-inf"), dtype)
    npty = to_numpy_type(dtype)
    if npty is np.bool_:
        return make_constant_expr(0, dtype)
    assert issubclass(npty, np.integer)
    return make_constant_expr(int(np.iinfo(npty).min), dtype)


def exclusive_min_tiled(value, log2_size):
    """Exclusive prefix min across ``2**log2_size`` consecutive lanes.

    Lane 0 of each group returns the dtype-typed identity: ``+inf`` for real dtypes, the dtype's maximum
    (``np.iinfo(dtype).max``) for integer dtypes.  See the module-level note for why this op (and ``exclusive_max``)
    is a plain Python wrapper rather than ``@func``.
    """
    return _exclusive_scan_tiled(value, _bin_min, _typed_min_identity(value), log2_size)


def exclusive_max_tiled(value, log2_size):
    """Exclusive prefix max across ``2**log2_size`` consecutive lanes.

    Lane 0 of each group returns the dtype-typed identity: ``-inf`` for real dtypes, the dtype's minimum
    (``np.iinfo(dtype).min``) for integer dtypes (``0`` for unsigned).
    """
    return _exclusive_scan_tiled(value, _bin_max, _typed_max_identity(value), log2_size)


@func
def exclusive_and_tiled(value, log2_size: template()):
    """Exclusive prefix bitwise-AND.  Integer dtypes only.  Lane 0 of each group returns all-bits-set in ``value``'s
    dtype."""
    return _exclusive_scan_tiled(value, _bin_and, ~(value ^ value), log2_size)


@func
def exclusive_or_tiled(value, log2_size: template()):
    """Exclusive prefix bitwise-OR.  Integer dtypes only.  Lane 0 of each group returns zero in ``value``'s dtype."""
    return _exclusive_scan_tiled(value, _bin_or, value ^ value, log2_size)


@func
def exclusive_xor_tiled(value, log2_size: template()):
    """Exclusive prefix bitwise-XOR.  Integer dtypes only.  Lane 0 of each group returns zero in ``value``'s dtype."""
    return _exclusive_scan_tiled(value, _bin_xor, value ^ value, log2_size)


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


# Full-subgroup convenience wrappers.  See the "Full-subgroup wrappers" header for the design rationale; each one is
# a one-liner that picks ``log2_size = log2_group_size()`` so the call covers every lane in the active subgroup.
# Kept as plain Python functions so the ``log2_group_size()`` int is resolved at compile time and flows into the
# underlying ``*_tiled`` op's ``template()`` argument.


def all_true(predicate):
    """``all_true_tiled`` across the entire subgroup -- see ``all_true_tiled(..., log2_size=log2_group_size())``."""
    return all_true_tiled(predicate, log2_group_size())


def any_true(predicate):
    """``any_true_tiled`` across the entire subgroup -- see ``any_true_tiled(..., log2_size=log2_group_size())``."""
    return any_true_tiled(predicate, log2_group_size())


def all_equal(value):
    """``all_equal_tiled`` across the entire subgroup -- see ``all_equal_tiled(..., log2_size=log2_group_size())``."""
    return all_equal_tiled(value, log2_group_size())


def reduce_add(value):
    """``reduce_add_tiled`` over the entire subgroup -- lane 0 ends up with the sum across all ``group_size()``
    lanes."""
    return reduce_add_tiled(value, log2_group_size())


def reduce_all_add(value):
    """``reduce_all_add_tiled`` over the entire subgroup -- every lane ends up with the sum across all lanes."""
    return reduce_all_add_tiled(value, log2_group_size())


def reduce_min(value):
    """``reduce_min_tiled`` over the entire subgroup -- lane 0 ends up with the min across all lanes."""
    return reduce_min_tiled(value, log2_group_size())


def reduce_max(value):
    """``reduce_max_tiled`` over the entire subgroup -- lane 0 ends up with the max across all lanes."""
    return reduce_max_tiled(value, log2_group_size())


def reduce_all_min(value):
    """``reduce_all_min_tiled`` over the entire subgroup -- every lane ends up with the min across all lanes."""
    return reduce_all_min_tiled(value, log2_group_size())


def reduce_all_max(value):
    """``reduce_all_max_tiled`` over the entire subgroup -- every lane ends up with the max across all lanes."""
    return reduce_all_max_tiled(value, log2_group_size())


def inclusive_add(value):
    """``inclusive_add_tiled`` over the entire subgroup."""
    return inclusive_add_tiled(value, log2_group_size())


def inclusive_mul(value):
    """``inclusive_mul_tiled`` over the entire subgroup."""
    return inclusive_mul_tiled(value, log2_group_size())


def inclusive_min(value):
    """``inclusive_min_tiled`` over the entire subgroup."""
    return inclusive_min_tiled(value, log2_group_size())


def inclusive_max(value):
    """``inclusive_max_tiled`` over the entire subgroup."""
    return inclusive_max_tiled(value, log2_group_size())


def inclusive_and(value):
    """``inclusive_and_tiled`` over the entire subgroup.  Integer dtypes only."""
    return inclusive_and_tiled(value, log2_group_size())


def inclusive_or(value):
    """``inclusive_or_tiled`` over the entire subgroup.  Integer dtypes only."""
    return inclusive_or_tiled(value, log2_group_size())


def inclusive_xor(value):
    """``inclusive_xor_tiled`` over the entire subgroup.  Integer dtypes only."""
    return inclusive_xor_tiled(value, log2_group_size())


def exclusive_add(value):
    """``exclusive_add_tiled`` over the entire subgroup."""
    return exclusive_add_tiled(value, log2_group_size())


def exclusive_mul(value):
    """``exclusive_mul_tiled`` over the entire subgroup."""
    return exclusive_mul_tiled(value, log2_group_size())


def exclusive_min(value):
    """``exclusive_min_tiled`` over the entire subgroup -- lane 0 returns the dtype's max (``+inf`` for floats)."""
    return exclusive_min_tiled(value, log2_group_size())


def exclusive_max(value):
    """``exclusive_max_tiled`` over the entire subgroup -- lane 0 returns the dtype's min (``-inf`` for floats, ``0``
    for unsigned ints)."""
    return exclusive_max_tiled(value, log2_group_size())


def exclusive_and(value):
    """``exclusive_and_tiled`` over the entire subgroup.  Integer dtypes only."""
    return exclusive_and_tiled(value, log2_group_size())


def exclusive_or(value):
    """``exclusive_or_tiled`` over the entire subgroup.  Integer dtypes only."""
    return exclusive_or_tiled(value, log2_group_size())


def exclusive_xor(value):
    """``exclusive_xor_tiled`` over the entire subgroup.  Integer dtypes only."""
    return exclusive_xor_tiled(value, log2_group_size())


def segmented_reduce_add(value, head_flag):
    """``segmented_reduce_add_tiled`` over the entire subgroup.

    ``log2_size = log2_group_size()`` --- 5 on wave32 backends, 6 on AMDGPU.
    """
    return segmented_reduce_add_tiled(value, head_flag, log2_group_size())


def segmented_reduce_min(value, head_flag):
    """``segmented_reduce_min_tiled`` over the entire subgroup."""
    return segmented_reduce_min_tiled(value, head_flag, log2_group_size())


def segmented_reduce_max(value, head_flag):
    """``segmented_reduce_max_tiled`` over the entire subgroup."""
    return segmented_reduce_max_tiled(value, head_flag, log2_group_size())


__all__ = [
    "sync",
    "mem_fence",
    "barrier",
    "memory_barrier",
    "elect",
    "ballot_first_n",
    "ballot",
    "ballot_full_subgroup",
    "all_true_tiled",
    "any_true_tiled",
    "all_equal_tiled",
    "broadcast_first",
    "lanemask_lt",
    "lanemask_le",
    "lanemask_eq",
    "lanemask_gt",
    "lanemask_ge",
    "group_size",
    "log2_group_size",
    "invocation_id",
    "reduce_add_tiled",
    "reduce_all_add_tiled",
    "reduce_min_tiled",
    "reduce_max_tiled",
    "reduce_all_min_tiled",
    "reduce_all_max_tiled",
    "segmented_reduce_add_tiled",
    "segmented_reduce_min_tiled",
    "segmented_reduce_max_tiled",
    "inclusive_add_tiled",
    "inclusive_mul_tiled",
    "inclusive_min_tiled",
    "inclusive_max_tiled",
    "inclusive_and_tiled",
    "inclusive_or_tiled",
    "inclusive_xor_tiled",
    "exclusive_add_tiled",
    "exclusive_mul_tiled",
    "exclusive_min_tiled",
    "exclusive_max_tiled",
    "exclusive_and_tiled",
    "exclusive_or_tiled",
    "exclusive_xor_tiled",
    "shuffle",
    "shuffle_xor",
    "shuffle_up",
    "shuffle_down",
    "all_true",
    "any_true",
    "all_equal",
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
]
