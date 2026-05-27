# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false
"""Subgroup-scoped sorting primitives.

Split out from `quadrants.lang.simt.subgroup` for the same reason `reductions` was: it keeps that module focused on
data movement (shuffle / broadcast / ballot / lanemask / vote primitives) and groups all sort-shaped helpers under one
implementation unit.  Public API is unchanged: everything defined here is re-exported by
`quadrants.lang.simt.subgroup`, so `qd.simt.subgroup.bitonic_sort_kv(k, v)` etc. work as if they lived in that module.

Only one family lives here today:

* ``bitonic_sort_kv_tiled`` -- in-register stable bitonic key/value sort across ``2**log2_size`` consecutive lanes, one
  ``(key, value)`` pair per lane.  Pure ``shuffle`` (no shared memory, no barriers within the sort), fully unrolled at
  compile time into the standard ``log2_size * (log2_size + 1) / 2`` compare-exchange stages.

The full-subgroup form ``bitonic_sort_kv(key, value)`` is a one-liner that picks ``log2_size = log2_group_size()``,
following the same convention as ``reduce_add`` / ``reduce_all_add`` / etc.
"""

from quadrants.lang import impl
from quadrants.lang.kernel_impl import func
from quadrants.lang.simt.subgroup import (
    invocation_id,
    log2_group_size,
    shuffle,
)
from quadrants.types.annotations import template
from quadrants.types.primitive_types import u32

# --- Bitonic key/value sort ------------------------------------------------------------
#
# Classic 1-D bitonic sorting network laid out across ``2**log2_size`` consecutive lanes, one ``(key, value)`` pair per
# lane.  For each outer ``k_log2`` in ``[1, log2_size]`` and inner ``j_log2`` in ``[k_log2 - 1, 0]``, lane ``i``
# exchanges with lane ``i ^ (1 << j_log2)`` via two ``shuffle`` ops.  Whether ``i`` keeps the min or max of the pair
# is decided by two bits of the lane id: ``(i & (1 << j_log2)) == 0`` says "I'm the low partner of this swap" and
# ``(i & (1 << k_log2)) == 0`` says "this is an ascending half of the bitonic sequence".  When those bits agree the
# lane keeps the min, otherwise the max.
#
# Stability is built in by extending the key compare to a lex compare on ``(key, value)``: ties between equal keys
# break on ascending value, so the relative order of equal-keyed pairs is preserved.  Sentinel-padding the high lanes
# with a key that compares greater than every real key (e.g. ``+inf`` for floats, ``INT_MAX`` for ints) is the
# documented way to sort fewer than ``2**log2_size`` real elements -- the sort moves the sentinels to the high end of
# the group, leaving the meaningful data contiguous starting at lane 0.


@func
def bitonic_sort_kv_tiled(key, value, log2_size: template()):
    """In-register stable bitonic key/value sort across ``2**log2_size`` consecutive lanes.

    Sorts ``(key, value)`` pairs ascending by ``key``; ties on ``key`` are broken by ascending ``value`` (stable).  Each
    lane holds one pair on entry; on exit, lane ``i`` (within its ``2**log2_size``-aligned tile) holds the ``i``-th
    smallest pair under that lex order.  Lanes in different tiles sort independently of each other; the result is
    broadcast-to-tile in the sense of [Tiled variants](#tiled-variants) -- every lane in a tile has a valid sorted
    pair, just not the same one.

    Caller contract:

    * ``2**log2_size`` must not exceed the active subgroup size on the target: up to 32 on CUDA / Metal /
      Vulkan-wave32, up to 64 on AMDGPU (every AMDGPU target is pinned to wave64).  ``log2_size = 0`` is the degenerate
      "every lane is its own tile of 1" case -- the function returns ``(key, value)`` unchanged.
    * ``log2_size`` is a ``qd.template()`` compile-time constant; the body is fully unrolled into exactly
      ``log2_size * (log2_size + 1) / 2`` compare-exchange stages, each one two ``shuffle`` ops + a lex compare + a
      predicated assignment.  For ``log2_size = 5`` (32 lanes) this is the standard 15-stage bitonic schedule.
    * Same uniform-CF + all-lanes-active contract as the rest of ``qd.simt.subgroup``: every lane in the subgroup must
      execute every call to ``bitonic_sort_kv_tiled`` together.
    * ``key`` and ``value`` are scalar values held one-per-lane.  ``key`` must support ``<`` and ``==``; ``value``
      must support ``<`` and ``==`` as well (the stability tiebreak compares values).  The supported scalar dtypes are
      the union of what ``subgroup.shuffle`` accepts for each (i32, u32, f32, f64, i64, u64).  ``key`` and ``value``
      do not have to share a dtype.

    Short-input pattern (sorting fewer than ``2**log2_size`` real elements):

    1. Load real data into the low ``n`` lanes (``lane < n``).
    2. Initialise the high lanes (``lane >= n``) with a sentinel key that compares greater than every real key
       (``+inf`` for floats, ``INT_MAX`` / ``UINT_MAX`` for ints) and any safe value (often ``-1`` for an "invalid
       index" marker).
    3. Call ``bitonic_sort_kv_tiled(key, value, log2_size)``.
    4. Read back only the low ``n`` lanes; the sentinels have drifted to the high end of the tile.

    Returns ``(key, value)`` -- assign with ``key, value = bitonic_sort_kv_tiled(key, value, log2_size)``.

    Float NaN handling on the key (or value) is implementation-defined: comparisons with NaN return false on most
    backends, so NaN-keyed lanes drift to an arbitrary position within the sorted tile and the result loses its
    "sorted" guarantee.  Bit-cast the key to a same-width integer dtype if you need a portable NaN-respecting order.

    Cost is exactly ``log2_size * (log2_size + 1)`` ``shuffle`` ops (two per compare-exchange stage, one for the
    partner's key and one for the partner's value) plus the same number of lex compares + predicated assignments, all
    fully unrolled into the calling kernel's IR.  No barriers, no shared memory, no launch overhead.  The shuffle
    pattern is the partner-XOR butterfly used by every other ``_tiled`` op in this module, so on AMDGPU the wave64
    cross-half handling (the ``permlane64`` + dual ``ds_bpermute`` lowering described in `AMDGPU wave64 cross-half
    lowering`) is inherited transparently.
    """
    lane = invocation_id()
    for k_log2 in impl.static(range(1, log2_size + 1)):
        k_mask = impl.static(1 << k_log2)
        for j_log2 in impl.static(range(k_log2 - 1, -1, -1)):
            j_mask = impl.static(1 << j_log2)
            partner = u32(lane) ^ u32(j_mask)
            their_key = shuffle(key, partner)
            their_value = shuffle(value, partner)
            i_am_low = (lane & j_mask) == 0
            ascending = (lane & k_mask) == 0
            take_min = i_am_low == ascending
            their_lt_mine = (their_key < key) or (their_key == key and their_value < value)
            if take_min:
                if their_lt_mine:
                    key = their_key
                    value = their_value
            else:
                if not their_lt_mine and (their_key != key or their_value != value):
                    key = their_key
                    value = their_value
    return key, value


def bitonic_sort_kv(key, value):
    """``bitonic_sort_kv_tiled`` across the entire subgroup -- the lex-smallest ``2**log2_group_size()`` pairs end up
    in ascending lane order.

    Plain Python wrapper (not ``@qd.func``): the ``log2_group_size()`` int is resolved at compile time and flows into
    the underlying ``@qd.func``'s ``template()`` parameter, so the generated IR is identical to a hand-written
    ``bitonic_sort_kv_tiled(key, value, 5)`` on wave32 backends or ``bitonic_sort_kv_tiled(key, value, 6)`` on AMDGPU
    wave64.  See `bitonic_sort_kv_tiled` for the full contract, the short-input pattern, and the NaN caveat.
    """
    return bitonic_sort_kv_tiled(key, value, log2_group_size())


__all__ = [
    "bitonic_sort_kv_tiled",
    "bitonic_sort_kv",
]
