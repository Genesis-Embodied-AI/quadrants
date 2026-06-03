# pyright: reportInvalidTypeForm=false, reportOperatorIssue=false, reportArgumentType=false
"""Subgroup-scoped sorting primitives.

Split out from `quadrants.lang.simt.subgroup` for the same reason `reductions` was: it keeps that module focused on
data movement (shuffle / broadcast / ballot / lanemask / vote primitives) and groups all sort-shaped helpers under one
implementation unit.  Public API is unchanged: everything defined here is re-exported by
`quadrants.lang.simt.subgroup`, so `qd.simt.subgroup.bitonic_sort_kv(k, v)` etc. work as if they lived in that module.

Only one family lives here today:

* ``bitonic_sort_kv_tiled`` -- in-register ascending lex sort on ``(key, value)`` pairs across ``2**log2_size``
  consecutive lanes, one ``(key, value)`` pair per lane.  Pure ``shuffle`` (no shared memory, no barriers within the
  sort), fully unrolled at compile time into the standard ``log2_size * (log2_size + 1) / 2`` compare-exchange stages.

Note this is *not* a stable sort in the textbook sense (preserve input lane order for equal keys): equal-keyed lanes
come back ordered by ``value``, not by their original lane id.  Callers that need lane-order stability should encode it
into ``value`` themselves (e.g. pack ``(payload, original_lane)``).

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
# The compare is a lex compare on ``(key, value)``: ties on ``key`` break on ascending ``value``.  This is *not* a
# stable sort in the textbook sense -- equal-keyed lanes come back in ascending-``value`` order, not in original-lane
# order.  Callers needing lane-order stability should pack the lane id into ``value`` themselves; for unique-``value``
# inputs (the Genesis contact-sort case) the distinction is invisible.  Sentinel-padding the high lanes with a key that
# compares greater than every real key (e.g. ``+inf`` for floats, ``INT_MAX`` for ints) is the documented way to sort
# fewer than ``2**log2_size`` real elements -- the sort moves the sentinels to the high end of the group, leaving the
# meaningful data contiguous starting at lane 0.


@func
def bitonic_sort_kv_tiled(key, value, log2_size: template()):
    """In-register ascending lex sort on ``(key, value)`` pairs across ``2**log2_size`` consecutive lanes.

    Sorts ``(key, value)`` pairs by ascending ``key``; ties on ``key`` break on ascending ``value`` (lex order on the
    ``(key, value)`` tuple).  Each lane holds one pair on entry; on exit, lane ``i`` (within its ``2**log2_size``-aligned
    tile) holds the ``i``-th smallest pair under that lex order.  Lanes in different tiles sort independently of each
    other; the result is broadcast-to-tile in the sense of [Tiled variants](#tiled-variants) -- every lane in a tile
    has a valid sorted pair, just not the same one.

    Not a textbook-stable sort: equal-keyed lanes come back in ascending-``value`` order, *not* in their original lane
    order.  If you need lane-order stability (i.e. ``value`` treated as opaque payload), encode the lane id into
    ``value`` yourself before calling -- e.g. pack ``(payload, original_lane)`` into a single value, or call from a
    context where ``value`` is unique by construction (the Genesis contact-sort case, where ``value`` is a contact
    index).

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
      must support ``<`` and ``==`` as well (the lex tiebreak compares values).  The supported scalar dtypes are the
      union of what ``subgroup.shuffle`` accepts for each (i32, u32, f32, f64, i64, u64).  ``key`` and ``value`` do not
      have to share a dtype.

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
    # ``tile_mask`` is the low ``log2_size`` bits, i.e. the lane-id-within-tile mask.  We AND ``k_mask`` with it when
    # reading the outer-stage direction bit so the sort treats every tile as an independent ascending sort: without
    # this, tile 1 (lanes ``[tile_size, 2*tile_size)``) would land "descending" at the final ``k_log2 == log2_size``
    # stage, because the unmasked ``(lane & tile_size) == 0`` test flips between adjacent tiles.  When
    # ``log2_size == log2_group_size()`` the mask is a no-op (``k_mask <= tile_size`` always, and ``k_mask < tile_size``
    # leaves the bit untouched; ``k_mask == tile_size`` zeroes it but every lane in the subgroup already had that bit
    # clear), so the full-subgroup path lowers to the same IR as without the AND.
    tile_mask = impl.static((1 << log2_size) - 1)
    for k_log2 in impl.static(range(1, log2_size + 1)):
        k_mask = impl.static(1 << k_log2)
        for j_log2 in impl.static(range(k_log2 - 1, -1, -1)):
            j_mask = impl.static(1 << j_log2)
            partner = u32(lane) ^ u32(j_mask)
            their_key = shuffle(key, partner)
            their_value = shuffle(value, partner)
            i_am_low = (lane & j_mask) == 0
            ascending = (lane & k_mask & tile_mask) == 0
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
