# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false
"""Subgroup-scoped reductions and scans.

Split out from `quadrants.lang.simt.subgroup` to keep that module focused on data movement (shuffle / broadcast /
ballot / lanemask / vote primitives) and to keep the "Reductions and scans" user-guide section as a single
implementation unit.  Public API is unchanged: everything defined here is re-exported by
`quadrants.lang.simt.subgroup`, so `qd.simt.subgroup.reduce_add(v)` etc. continue to work.

The four families live here for distinct reasons:

* `reduce_*_tiled`              -- ``shuffle_down`` / ``shuffle`` butterfly (fan to lane-0 / broadcast variants).
* `segmented_reduce_*_tiled`    -- ``shuffle_up`` Hillis-Steele scan guarded by a bitmask-derived distance to head.
* `inclusive_*_tiled`           -- unguarded ``shuffle_up`` Hillis-Steele scan over a template binary op.
* `exclusive_*_tiled`           -- ``shuffle_up`` shift to seed lane-0 with the op's identity, then inclusive scan.

The four share idiom (shuffle-based reductions over ``2**log2_size`` consecutive lanes) but not code; only the
exclusive scans actually invoke a helper defined for another family (``_exclusive_scan_tiled`` -> inclusive scan).
"""

import numpy as np

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.lang.expr import make_constant_expr
from quadrants.lang.kernel_impl import func
from quadrants.lang.ops import clz
from quadrants.lang.simt.subgroup import (
    ballot,
    invocation_id,
    log2_group_size,
    shuffle,
    shuffle_down,
    shuffle_up,
)
from quadrants.lang.util import to_numpy_type
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32, u64

# --- Reductions ------------------------------------------------------------------------


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
    (32 on CUDA / Metal, 64 on AMDGPU -- wave64 is forced on every AMDGPU target).

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
# ballot + 1 clz + ``log2_size`` shuffles + ``log2_size`` ops - the same shape as `inclusive_add_tiled` /
# `inclusive_min_tiled` / `inclusive_max_tiled`, plus a single-instruction setup.
#
# No identity element is involved at all -- the per-lane ``distance >= offset`` guard ensures the scan never reaches
# across a segment boundary, so a partner from another segment is never combined with the local value (i.e. the
# implementation doesn't need a "what to combine with at the segment head" sentinel the way ``exclusive_min`` /
# ``exclusive_max`` do for lane 0 within each tile).


@func
def _segment_head_distance_tiled(head_flag, log2_size: template()):
    """Compute ``lane - segment_head`` - how many lanes the current lane sits past its segment head, scoped to
    ``2**log2_size`` lanes.  Returns ``0`` at the segment head, ``1, 2, ...`` for later lanes within the segment.

    Shared by `segmented_reduce_add_tiled` / `_min` / `_max`; see the module-level note for the algorithm.

    Two compile-time-selected paths:

    * **``log2_size <= 5`` (segments fit in a single 32-lane half)** - u32-bitmask path.  Pulls a u64
      ``ballot``, shifts the relevant 32-lane half down to bits 0..31, then runs the bit-mask algorithm in
      half-local lane coordinates.  Half-local ``distance`` equals absolute ``distance`` because both ``lane`` and the
      recovered ``segment_head`` are offset by the same ``half_base``, so the downstream ``distance >= offset`` guard
      still works in absolute terms.  On wave32 this collapses to the no-op (``half_base == 0`` always); on wave64
      lanes 32..63 see their own half-mask via the shift.  This is the only path on CUDA / Metal / Vulkan-wave32 (and
      compiles to identical IR to the historical wave32-only impl), so backends with ``group_size() == 32`` see no perf
      regression from supporting wave64 here.

    * **``log2_size == 6`` (full-wave64 segments, only reachable on AMDGPU)** - u64-bitmask path.  Segments span all 64
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
        # Half-local lane index - always in ``[0, 32)``, so all the u32 shifts below are well-defined.
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
    # violates the documented caller contract (``2**log2_size <= group_size()``) - checked by the ``static_assert`` in
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
    one operand is an f32 produced by a shuffle intrinsic - the select silently returns the false-branch value
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
    on AMDGPU -- wave64 is forced on every AMDGPU target).
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


# --- Full-subgroup wrappers ------------------------------------------------------------
#
# Plain Python one-liners (not ``@qd.func``) so the ``log2_group_size()`` int is resolved at compile time and flows
# into the underlying ``*_tiled`` op's ``template()`` argument.  Same rationale as the wrappers in
# ``quadrants.lang.simt.subgroup`` (``all_true`` / ``any_true`` / ``all_equal``).


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
    """``exclusive_min_tiled`` over the entire subgroup."""
    return exclusive_min_tiled(value, log2_group_size())


def exclusive_max(value):
    """``exclusive_max_tiled`` over the entire subgroup."""
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
