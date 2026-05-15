# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false

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


# Reduce / segmented reduce / inclusive / exclusive scan implementations and full-subgroup wrappers live in
# ``quadrants.lang.simt.reductions`` and are re-exported below; the public API
# (``qd.simt.subgroup.reduce_add(v)`` etc.) is unchanged.


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


# Pull in the reduce / scan / segmented-reduce APIs that live in ``quadrants.lang.simt.reductions``.  The import
# has to happen *after* the primitives ``reductions`` depends on (``ballot``, ``invocation_id``, ``shuffle``,
# ``shuffle_up``, ``shuffle_down``, ``log2_group_size``) are defined, so this back-import sits at the bottom of
# the module body.  Side effect: every name in ``reductions.__all__`` becomes accessible as
# ``qd.simt.subgroup.X``, preserving the historical public API.
#
# WARNING: reordering definitions above this line -- in particular moving any of ``ballot`` / ``invocation_id`` /
# ``shuffle`` / ``shuffle_up`` / ``shuffle_down`` / ``log2_group_size`` below this back-import -- will silently
# break the circular import (``reductions`` would see only a partially-populated ``subgroup`` module).  Keep this
# line at the bottom of the module body, after every name ``reductions`` consumes.
from quadrants.lang.simt.reductions import *  # noqa: E402, F401  pylint: disable=wrong-import-position,wildcard-import

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
