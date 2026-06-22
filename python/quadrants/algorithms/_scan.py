# type: ignore
"""Device-wide exclusive-scan primitives.

Provides the graph-composable ``qd.algorithms.exclusive_scan_{add,min,max}`` on top of the block-tier
``block.exclusive_scan`` primitive, plus :func:`exclusive_scan_scratch_slots` for sizing the caller-owned scratch. See
the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` (Blelloch 1990 / Harris-Sengupta-Owens 2007,
three-pass formulation).

Each ``exclusive_scan_{add,min,max}`` is a fixed-depth three-pass staircase of ``@qd.func`` phases - pass 1 tile-reduce
(:func:`_reduce_phase`), pass 2 in-place scan of the partials (:func:`_emit_scan_inplace`), pass 3 per-tile downsweep
(:func:`_scan_downsweep_phase`). Call it at the **top level** of your own ``@qd.kernel`` with the live count ``n`` as a
device ``Expr`` and the recursion depth as a compile-time ``log256_max_n``, so one captured graph replays for any count
``<= BLOCK_DIM ** log256_max_n``. The scan is out-of-place (``arr`` -> ``out``); the per-tile partials stage through a
**caller-owned** scratch buffer (``u32`` for 4-byte element dtypes, ``u64`` for 8-byte ones; ``0`` slots for
``n <= BLOCK_DIM``) sized via :func:`exclusive_scan_scratch_slots`.

The ``PrefixSumExecutor`` class in ``_algorithms.py`` predates this work and is kept for backward compat. ``sort``
reuses the private ``u32`` / add staircase (:func:`_emit_exclusive_scan_add`, at the bottom of this module) for its
digit-histogram scan.
"""

from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.reductions import (
    _bin_add,
    _typed_max_identity,
    _typed_min_identity,
)
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32, u64

from ._reduce import (
    _OP_ADD,
    _OP_BINS,
    _OP_MAX,
    _OP_MIN,
    BLOCK_DIM,
    _at_least_one,
    _dtype_width_bytes,
    _level_partials_slots,
    _reduce_depth_for_n,
    _reduce_pass,
    _reduce_pass_u64,
    _reduce_phase,
    _scratch_dtype_for_width,
    _typed_zero_expr,
    _validate_log256_max_n,
)


@kernel
def _scan_block_inplace_u32(
    buf: template(),
    buf_off: i32,
    n_valid: i32,
    identity_bits: u32,
    op: template(),
    dtype: template(),
):
    """Single-block in-place exclusive scan of ``buf[buf_off : buf_off + n_valid]`` (4-byte dtype path).

    Used at the recursion base of the scan driver, when the buffer being scanned fits in a single block. ``buf`` is
    the shared ``Field(u32)`` scratch; the per-thread read / write go through ``qd.bit_cast`` to / from ``dtype``.

    Threads with ``i >= n_valid`` participate with ``identity`` (so the block-scope scan algorithm sees a clean
    monoid) but do not write back.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        identity = bit_cast(identity_bits, dtype)
        v = identity
        if i < n_valid:
            v = bit_cast(buf[buf_off + i], dtype)
        prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, dtype)
        if i < n_valid:
            buf[buf_off + i] = bit_cast(prefix, u32)


@kernel
def _scan_block_inplace_u64(
    buf: template(),
    buf_off: i32,
    n_valid: i32,
    identity_bits: u64,
    op: template(),
    dtype: template(),
):
    """8-byte sibling of :func:`_scan_block_inplace_u32`. Stages through the ``Field(u64)`` scratch."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        identity = bit_cast(identity_bits, dtype)
        v = identity
        if i < n_valid:
            v = bit_cast(buf[buf_off + i], dtype)
        prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, dtype)
        if i < n_valid:
            buf[buf_off + i] = bit_cast(prefix, u64)


@kernel
def _scan_pass3(
    src: template(),
    src_off: i32,
    prefixes: template(),
    prefixes_off: i32,
    dst: template(),
    dst_off: i32,
    n_valid: i32,
    total_threads: i32,
    identity_bits: u32,
    op: template(),
    dtype: template(),
    src_is_u32: template(),
    dst_is_u32: template(),
):
    """Pass-3 downsweep: per-block tile scan + apply block prefix from scratch.

    Reads ``src[src_off : src_off + n_valid]`` (template-switched between the dtype tensor path and the u32-scratch
    ``bit_cast`` path), computes per-thread tile prefixes via ``block.exclusive_scan``, looks up the block prefix at
    ``prefixes[prefixes_off + block_id]`` (always a u32 scratch slot holding the dtype value bit-cast to u32, written
    by Pass 2), and writes ``op(block_prefix, tile_prefix)`` to ``dst[dst_off + i]``.

    ``dst`` may alias ``src`` (in-place recursion case); the read-modify-write is per-thread and the
    block.exclusive_scan internally barriers, so threads in a block see consistent values and writes by other blocks
    land in disjoint tiles.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        identity = bit_cast(identity_bits, dtype)
        v = identity
        if i < n_valid:
            if static(src_is_u32):
                v = bit_cast(src[src_off + i], dtype)
            else:
                v = src[src_off + i]
        tile_prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, dtype)
        block_prefix = bit_cast(prefixes[prefixes_off + block_id], dtype)
        if i < n_valid:
            scanned = op(block_prefix, tile_prefix)
            if static(dst_is_u32):
                dst[dst_off + i] = bit_cast(scanned, u32)
            else:
                dst[dst_off + i] = scanned


@kernel
def _scan_pass3_u64(
    src: template(),
    src_off: i32,
    prefixes: template(),
    prefixes_off: i32,
    dst: template(),
    dst_off: i32,
    n_valid: i32,
    total_threads: i32,
    identity_bits: u64,
    op: template(),
    dtype: template(),
    src_is_u64: template(),
    dst_is_u64: template(),
):
    """8-byte sibling of :func:`_scan_pass3`. Stages through the ``Field(u64)`` scratch."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        identity = bit_cast(identity_bits, dtype)
        v = identity
        if i < n_valid:
            if static(src_is_u64):
                v = bit_cast(src[src_off + i], dtype)
            else:
                v = src[src_off + i]
        tile_prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, dtype)
        block_prefix = bit_cast(prefixes[prefixes_off + block_id], dtype)
        if i < n_valid:
            scanned = op(block_prefix, tile_prefix)
            if static(dst_is_u64):
                dst[dst_off + i] = bit_cast(scanned, u64)
            else:
                dst[dst_off + i] = scanned


def _scan_total_scratch_slots(n, partials_cursor):
    """Return the high-water-mark scratch slot index that ``_exclusive_scan_inplace_{u32,u64}`` will use to scan
    ``n`` elements with its partials starting at ``partials_cursor`` (i.e. the smallest required ``capacity`` such
    that ``capacity >= return_value`` is sufficient for the entire recursion).

    Mirrors the level-by-level allocation that the recursion does internally: at each level we bump ``partials_cursor``
    by ``B = ceil(n / BLOCK_DIM)`` and recurse on ``B``, until ``B <= BLOCK_DIM`` (base case, no further partials).
    Callers (e.g. ``sort``) should use this helper for their *up-front* scratch check so they refuse the call before any
    in-place mutation runs (see PR 693 review: a single-level estimate misses deeper recursion levels and lets
    ``_twiddle_pass`` corrupt the user's keys before the recursive ``RuntimeError`` fires).

    Delegates to :func:`_level_partials_slots`, so it is host- **and** kernel-callable (branch-free arithmetic over
    an unrolled fixed loop); ``n`` / ``partials_cursor`` may be Python ``int`` s or device-read ``Expr`` s. The check
    inside ``_exclusive_scan_inplace_*`` itself stays as a defensive backstop; this helper is the contract that the
    *outer* drivers should honour first.
    """
    return _level_partials_slots(n, partials_cursor)


def _exclusive_scan_inplace_u32(scratch, off: int, n: int, identity_bits: int, op, dtype, partials_cursor: int):
    """Exclusive-scan ``scratch[off : off + n]`` in place. Recursive.

    ``partials_cursor`` is the next free u32 slot to use for the partials of the recursive level; the driver bumps it
    down each level.
    """
    if n <= BLOCK_DIM:
        _scan_block_inplace_u32(scratch, off, n, identity_bits, op, dtype)
        return

    B = (n + BLOCK_DIM - 1) // BLOCK_DIM
    partials_off = partials_cursor
    # Capacity comes from the caller-owned buffer we were handed (``scratch.shape[0]``); the caller is expected to size
    # it via ``exclusive_scan_scratch_slots`` up front, so this is a backstop.
    if partials_off + B > scratch.shape[0]:
        raise RuntimeError(
            f"device exclusive scan ran out of scratch at recursion level "
            f"n={n}, B={B}, partials_off={partials_off}, capacity={scratch.shape[0]}. "
            f"Allocate a larger scratch sized via exclusive_scan_scratch_slots(N)."
        )

    _reduce_pass(
        scratch,
        scratch,
        off,
        partials_off,
        n,
        B * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        True,
        True,
    )

    _exclusive_scan_inplace_u32(scratch, partials_off, B, identity_bits, op, dtype, partials_off + B)

    _scan_pass3(
        scratch,
        off,
        scratch,
        partials_off,
        scratch,
        off,
        n,
        B * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        True,
        True,
    )


def _exclusive_scan_inplace_u64(scratch, off: int, n: int, identity_bits: int, op, dtype, partials_cursor: int):
    """8-byte sibling of :func:`_exclusive_scan_inplace_u32`. Stages through the ``Field(u64)`` scratch.

    Used internally by the 64-bit ``exclusive_scan_*`` path. Mirrors the 32-bit recursion shape: tile-reduce into
    ``scratch[partials_off : partials_off + B]``, recurse on those partials, then downsweep back over the original
    ``scratch[off : off + n]`` to apply per-tile prefixes.
    """
    if n <= BLOCK_DIM:
        _scan_block_inplace_u64(scratch, off, n, identity_bits, op, dtype)
        return

    B = (n + BLOCK_DIM - 1) // BLOCK_DIM
    partials_off = partials_cursor
    # Capacity from the handed-in buffer; see note in ``_exclusive_scan_inplace_u32``.
    if partials_off + B > scratch.shape[0]:
        raise RuntimeError(
            f"device exclusive scan ran out of u64 scratch at recursion level n={n}, B={B}, "
            f"partials_off={partials_off}, capacity={scratch.shape[0]}. "
            f"Allocate a larger scratch sized via exclusive_scan_scratch_slots(N)."
        )

    _reduce_pass_u64(
        scratch,
        scratch,
        off,
        partials_off,
        n,
        B * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        True,
        True,
    )

    _exclusive_scan_inplace_u64(scratch, partials_off, B, identity_bits, op, dtype, partials_off + B)

    _scan_pass3_u64(
        scratch,
        off,
        scratch,
        partials_off,
        scratch,
        off,
        n,
        B * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        True,
        True,
    )


def exclusive_scan_scratch_slots(n, log256_max_n: int = None) -> int:
    """Number of scratch slots ``exclusive_scan_{add,min,max}`` need to scan a length-``n`` input.

    The out-of-place staircase stages the per-tile partials in scratch: pass 1 writes ``b0 = ceil(n / BLOCK_DIM)``
    partials, then the in-place scan of those partials stacks ``ceil(b0 / BLOCK_DIM)`` more, ... for ``depth - 2``
    further levels (the final tile-scan writes straight back). The count is **dtype-width-independent** (slot count,
    not byte count): 4-byte scans stage through a ``u32`` scratch and 8-byte scans through a ``u64`` scratch, both of
    this many slots::

        slots = qd.algorithms.exclusive_scan_scratch_slots(N)
        scratch = qd.field(qd.u32, shape=slots)   # u64 for i64 / u64 / f64 inputs

    Two ways to call it: **explicit depth** ``exclusive_scan_scratch_slots(n, D)`` is host- **and** kernel-callable
    (branch-free arithmetic over the unrolled ``D`` loop); **auto depth** ``exclusive_scan_scratch_slots(n)`` derives
    the minimal ``D`` from ``n`` (host-only). Always returns **at least 1** so the result can size an allocation
    directly; the depth ``<= 1`` case (``n <= BLOCK_DIM``: a single tile scans straight to ``out`` with no scratch)
    returns ``1`` (the lone slot is never touched).
    """
    if log256_max_n is None:
        log256_max_n = _reduce_depth_for_n(n)
    _validate_log256_max_n(log256_max_n)
    if log256_max_n <= 1:
        return 1
    cursor = (n + (BLOCK_DIM - 1)) // BLOCK_DIM  # b0 partials from pass 1
    cur = cursor
    for _ in range(log256_max_n - 2):
        cur = (cur + (BLOCK_DIM - 1)) // BLOCK_DIM
        cursor = cursor + cur
    return _at_least_one(cursor)


# ---------------------------------------------------------------------------------------------------------------------
# Graph-composable exclusive scan: fixed-depth staircase of @qd.func phases (device-resident count, compile-time depth)
# ---------------------------------------------------------------------------------------------------------------------
#
# Mirrors the reduce design (see ``_reduce.py``): the three-pass Blelloch scan is emitted as a staircase of ``@qd.func``
# phases. ``exclusive_scan_{add,min,max}`` are the graph-composable entries: call them at the **top level** of your
# own ``@qd.kernel`` with a device-resident count ``n`` (i32 Expr) and a compile-time ``log256_max_n``, so one captured
# graph replays for any count ``<= BLOCK_DIM ** log256_max_n``. The scan is out-of-place (``arr`` -> ``out``);
# ``scratch`` stages the per-tile partials staircase (``bit_cast`` through ``u32`` / ``u64``). See
# ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.


def _scan_identity(dtype, op):
    """Typed monoid-identity ``Expr`` for ``(op, dtype)``: ``0`` for add, ``+extremum`` for min, ``-extremum`` for max.
    Trace-time helper (like :func:`_typed_min_identity`), used to seed out-of-range lanes and lane 0's predecessor in
    the block scans. Returns an ``Expr``, so its result is a valid unconditional first assignment inside a phase."""
    ident = _typed_zero_expr(dtype)
    if op == _OP_MIN:
        return _typed_min_identity(ident)
    if op == _OP_MAX:
        return _typed_max_identity(ident)
    return ident


@_func
def _scan_tile_phase(
    src: template(),
    dst: template(),
    src_off: i32,
    dst_off: i32,
    n: i32,
    dtype: template(),
    wide: template(),
    op: template(),
    op_bin: template(),
    src_wide: template(),
    dst_wide: template(),
):
    """Single-block exclusive scan of ``src[src_off:src_off+n]`` -> ``dst[dst_off:dst_off+n]`` (``n <= BLOCK_DIM``).

    The recursion base / single-tile case. ``src_wide`` / ``dst_wide`` switch between the ``bit_cast``-through-``wide``
    scratch path and the direct caller-tensor path. ``v`` is seeded with the monoid identity *before* the load branch
    (Quadrants requires a variable's first assignment at the outer scope)."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        ident = _scan_identity(dtype, op)
        v = ident
        if i < n:
            if static(src_wide):
                v = bit_cast(src[src_off + i], dtype)
            else:
                v = src[src_off + i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, op_bin, ident, dtype)
        if i < n:
            if static(dst_wide):
                dst[dst_off + i] = bit_cast(prefix, wide)
            else:
                dst[dst_off + i] = prefix


@_func
def _scan_downsweep_phase(
    src: template(),
    prefixes: template(),
    dst: template(),
    src_off: i32,
    prefixes_off: i32,
    dst_off: i32,
    n: i32,
    total_threads: i32,
    dtype: template(),
    wide: template(),
    op: template(),
    op_bin: template(),
    src_wide: template(),
    dst_wide: template(),
):
    """Per-tile exclusive scan of ``src[src_off:src_off+n]`` plus the scanned per-tile prefix
    ``prefixes[prefixes_off + block_id]``, written to ``dst[dst_off:dst_off+n]``.

    ``prefixes`` is always the ``wide`` scratch (the scanned partials from a lower level). ``src`` / ``dst`` switch
    between the caller tensors (``src_wide`` / ``dst_wide`` False) and the ``wide`` scratch (in-place partials scan).
    ``dst`` may alias ``src``; the per-thread read-modify-write and ``block.exclusive_scan``'s internal barrier keep a
    block's tile consistent, and blocks write disjoint tiles."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        # When this grid-strided loop wraps (total_threads > the codegen grid-stride cap), the block-collective
        # below reuses the same threadgroup-shared scratch every iteration. The collective only barriers between its
        # own shared write and read, not at the iteration boundary, so iteration k+1's shared writes would race
        # iteration k's shared reads (a WAR data race - UB on every backend, observed as corruption on Metal /
        # MoltenVK). This boundary barrier retires the previous iteration's shared reads before they are overwritten.
        _block.sync()
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        ident = _scan_identity(dtype, op)
        v = ident
        if i < n:
            if static(src_wide):
                v = bit_cast(src[src_off + i], dtype)
            else:
                v = src[src_off + i]
        tile_prefix = _block.exclusive_scan(v, BLOCK_DIM, op_bin, ident, dtype)
        block_prefix = bit_cast(prefixes[prefixes_off + block_id], dtype)
        if i < n:
            scanned = op_bin(block_prefix, tile_prefix)
            if static(dst_wide):
                dst[dst_off + i] = bit_cast(scanned, wide)
            else:
                dst[dst_off + i] = scanned


def _emit_scan_inplace(buf, off, n, levels_remaining, dtype, wide, op, op_bin):
    """Emit a fixed-depth in-place exclusive scan of ``buf[off:off+n]`` (``wide`` scratch) at kernel-compile time.

    The partials-scan half of the staircase (Blelloch pass 2): reduce ``buf[off:off+n]`` into ``buf[off+n:...]``,
    recursively scan those partials, then downsweep back. ``n`` / ``off`` flow as ``Expr`` s; ``levels_remaining`` is a
    Python int so the depth is a compile-time constant (the base is reached by exhausting it, not by inspecting ``n``).
    Reuses :func:`_reduce_phase` for the tile-reduce rung."""
    if levels_remaining == 0:
        _scan_tile_phase(buf, buf, off, off, n, dtype, wide, op, op_bin, True, True)
        return
    B = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    part_off = off + n
    _reduce_phase(buf, buf, off, part_off, n, B * BLOCK_DIM, dtype, wide, op, op_bin, True, True)
    _emit_scan_inplace(buf, part_off, B, levels_remaining - 1, dtype, wide, op, op_bin)
    _scan_downsweep_phase(buf, buf, buf, off, part_off, off, n, B * BLOCK_DIM, dtype, wide, op, op_bin, True, True)


def _emit_scan(arr, out, scratch, n, log256_max_n, dtype, wide, op):
    """Emit a fixed-depth (``log256_max_n``) out-of-place exclusive scan of ``arr[0:n]`` into ``out[0:n]`` at compile
    time.

    ``log256_max_n == 1`` (``n <= BLOCK_DIM``) is a single tile straight to ``out``; otherwise the three-pass scan:
    pass 1 tile-reduce ``arr`` -> ``scratch[0:b0]`` (:func:`_reduce_phase`), pass 2 in-place scan of those ``b0``
    partials (:func:`_emit_scan_inplace`, ``log256_max_n - 2`` further levels), pass 3 downsweep ``arr`` +
    ``scratch[0:b0]`` -> ``out``. An over-specified ``log256_max_n`` bottoms out at length-1 partials that scan as
    identity rungs."""
    _validate_log256_max_n(log256_max_n)
    op_bin = _OP_BINS[op]  # resolve the binary op at trace time so the @qd.func phases receive it as a template
    if log256_max_n == 1:
        _scan_tile_phase(arr, out, 0, 0, n, dtype, wide, op, op_bin, False, False)
        return
    b0 = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    _reduce_phase(arr, scratch, 0, 0, n, b0 * BLOCK_DIM, dtype, wide, op, op_bin, False, True)
    _emit_scan_inplace(scratch, 0, b0, log256_max_n - 2, dtype, wide, op, op_bin)
    _scan_downsweep_phase(arr, scratch, out, 0, 0, 0, n, b0 * BLOCK_DIM, dtype, wide, op, op_bin, False, False)


@_func(requires_top_level=True)
def exclusive_scan_add(
    arr: template(), out: template(), scratch: template(), n: i32, dtype: template(), log256_max_n: template()
):
    """Graph-composable ``out[i] = sum(arr[0:i])`` (exclusive prefix sum).

    **Experimental** - this API is new and may change in a future release.

    Call at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent); never nest it in ordinary
    runtime ``for`` / ``if`` / ``while`` control flow. ``n`` is a device ``Expr`` (the live count); ``dtype`` is the
    element dtype (pass it explicitly - an ndarray kernel arg has no in-kernel ``.dtype``); ``log256_max_n`` is the
    compile-time phase count - the scan handles any count ``<= BLOCK_DIM ** log256_max_n``. ``out`` must be distinct
    from ``arr``; size ``scratch`` via :func:`exclusive_scan_scratch_slots` ``(capacity_n, log256_max_n)``."""
    wide = _scratch_dtype_for_width(_dtype_width_bytes(dtype))  # compile-time (from the dtype template)
    _emit_scan(arr, out, scratch, n, log256_max_n, dtype, wide, _OP_ADD)


@_func(requires_top_level=True)
def exclusive_scan_min(
    arr: template(), out: template(), scratch: template(), n: i32, dtype: template(), log256_max_n: template()
):
    """Graph-composable ``out[i] = min(arr[0:i])`` (identity ``+extremum`` from ``dtype``). **Experimental** (new API,
    may change). See :func:`exclusive_scan_add` for the top-level-call contract and arg semantics."""
    wide = _scratch_dtype_for_width(_dtype_width_bytes(dtype))
    _emit_scan(arr, out, scratch, n, log256_max_n, dtype, wide, _OP_MIN)


@_func(requires_top_level=True)
def exclusive_scan_max(
    arr: template(), out: template(), scratch: template(), n: i32, dtype: template(), log256_max_n: template()
):
    """Graph-composable ``out[i] = max(arr[0:i])`` (identity ``-extremum`` from ``dtype``). **Experimental** (new API,
    may change). See :func:`exclusive_scan_add` for the top-level-call contract and arg semantics."""
    wide = _scratch_dtype_for_width(_dtype_width_bytes(dtype))
    _emit_scan(arr, out, scratch, n, log256_max_n, dtype, wide, _OP_MAX)


# ---------------------------------------------------------------------------------------------------------------------
# Internal: u32 / add fixed-depth exclusive-scan staircase (shared with the radix-sort histogram scan)
# ---------------------------------------------------------------------------------------------------------------------
#
# This is the ``u32`` / add specialization of the exclusive-scan staircase that ``sort`` reuses for its digit-histogram
# scan: a fixed-depth staircase of ``@qd.func`` phases (in place) emitted at kernel-compile time, so ``n`` flows as a
# device ``Expr`` while the recursion depth is a compile-time Python int (constant launch topology). Kept separate from
# the generic typed staircase above (``_emit_scan``) because the histogram scan is always ``u32`` / add and operates in
# place. See ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.


@_func
def _graph_scan_reduce(buf: template(), in_off: i32, out_off: i32, n: i32, total_threads: i32):
    """Tile-reduce ``buf[in_off:in_off+n]`` -> per-tile sums ``buf[out_off:out_off+ceil(n/BLOCK_DIM)]`` (u32 / add).

    One tile per block; out-of-range lanes contribute ``0``. ``@qd.func`` phase of :func:`_emit_exclusive_scan_add` -
    its single top-level ``for`` becomes its own offloaded GPU launch (and graph node) when inlined into a kernel.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        _block.sync()  # iteration-boundary barrier: see _scan_downsweep_phase (shared-scratch WAR hazard on wrap)
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        v = u32(0)
        if i < n:
            v = buf[in_off + i]
        agg = _block.reduce_add(v, BLOCK_DIM, u32)
        if tid == 0:
            buf[out_off + block_id] = agg


@_func
def _graph_scan_base(buf: template(), off: i32, n_valid: i32):
    """Single-block in-place exclusive scan of ``buf[off:off+n_valid]`` (``n_valid <= BLOCK_DIM``); recursion base of
    the staircase. u32 / add specialization of :func:`_scan_block_inplace_u32`."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        v = u32(0)
        if i < n_valid:
            v = buf[off + i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, _bin_add, u32(0), u32)
        if i < n_valid:
            buf[off + i] = prefix


@_func
def _graph_scan_downsweep(buf: template(), off: i32, part_off: i32, n: i32, total_threads: i32):
    """Downsweep: per-tile exclusive scan of ``buf[off:off+n]`` plus the scanned per-tile prefix at
    ``buf[part_off + block_id]``, written back in place (u32 / add, ``src == dst == buf``)."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        _block.sync()  # iteration-boundary barrier: see _scan_downsweep_phase (shared-scratch WAR hazard on wrap)
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        v = u32(0)
        if i < n:
            v = buf[off + i]
        tile_prefix = _block.exclusive_scan(v, BLOCK_DIM, _bin_add, u32(0), u32)
        block_prefix = buf[part_off + block_id]
        if i < n:
            buf[off + i] = block_prefix + tile_prefix


def _emit_exclusive_scan_add(buf, off, n, levels_remaining: int):
    """Emit a *fixed-depth* in-place ``u32`` / add exclusive scan of ``buf[off:off+n]`` at kernel-compile time.

    Plain-Python helper run during kernel tracing (the scan counterpart of radix sort's ``_emit_pass``): it makes the
    ``@qd.func`` calls (:func:`_graph_scan_reduce` / :func:`_graph_scan_base` / :func:`_graph_scan_downsweep`) that each
    become a top-level offloaded GPU launch - and a node in the enclosing captured graph. Call it at the **top level**
    of a kernel (a ``while qd.graph_do_while(...)`` body also counts as top level); never nest it in ordinary runtime
    ``for`` / ``if`` / ``while`` control flow, which would demote the phase loops and collapse the per-phase grid-wide
    barriers.

    ``off`` / ``n`` flow as Quadrants ``Expr`` s (runtime), so the offsets track the actual count and one captured
    graph serves every count ``<= 256 ** (levels_remaining + 1)``; ``levels_remaining`` is a Python int, so the
    recursion depth - and hence the launch topology - is a compile-time constant. The base case is reached by
    exhausting ``levels_remaining`` (not by inspecting ``n``), giving a constant depth. The per-tile partials are
    stacked in ``buf`` above ``n``.
    """
    if levels_remaining == 0:
        _graph_scan_base(buf, off, n)
        return
    B = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    part_off = off + n
    _graph_scan_reduce(buf, off, part_off, n, B * BLOCK_DIM)
    _emit_exclusive_scan_add(buf, part_off, B, levels_remaining - 1)
    _graph_scan_downsweep(buf, off, part_off, n, B * BLOCK_DIM)


__all__ = [
    "exclusive_scan_add",
    "exclusive_scan_max",
    "exclusive_scan_min",
    "exclusive_scan_scratch_slots",
]
