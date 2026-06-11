# type: ignore
"""Device-wide exclusive-scan primitives.

Implements ``qd.algorithms.exclusive_scan_{add,min,max}`` on top of the block-tier ``block.exclusive_scan``
primitive. See the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the algorithmic rationale
(Blelloch 1990 / Harris-Sengupta-Owens 2007, three-pass formulation).

Algorithm (three-pass, multi-level when needed):

- **Pass 1: per-block tile reduce.** Each block reads ``BLOCK_DIM`` input elements, reduces them via
  ``block.reduce(op, dtype)``, thread 0 writes the per-block aggregate into the shared ``u32`` scratch field
  (``qd.bit_cast`` on write).  Identical to ``_reduce_pass`` in ``_reduce.py``; we reuse that kernel.
- **Pass 2: exclusive-scan the partials.** Once the partials buffer is built, exclusive-scan it in place. For
  ``B <= BLOCK_DIM`` a single block does it in one kernel launch (``_scan_block_inplace_u32``). For ``B > BLOCK_DIM``
  the driver recurses: it runs another tile-reduce on the partials buffer to produce a smaller partials-of-partials
  buffer, recursively scans that, then runs a downsweep over the partials buffer to apply the per-tile prefixes.
- **Pass 3: per-block tile scan + block-prefix.** Each block re-reads its tile from the input source, computes
  per-thread tile prefixes via ``block.exclusive_scan(op, identity, dtype)``, fetches its block prefix from the
  scanned partials buffer, and writes ``out[i] = op(block_prefix, tile_prefix)``.

**Scratch.** ``exclusive_scan_*`` needs a **caller-owned** buffer sized via
:func:`exclusive_scan_scratch_slots` (the partials buffer, ``~N / BLOCK_DIM`` slots plus deeper recursion
levels; ``0`` for ``N <= BLOCK_DIM``, which runs as a single-tile launch with no scratch). The dtype is ``u32`` for
4-byte element types and ``u64`` for 8-byte ones. There is no module-level shared scratch - the caller always owns
the buffer; a too-small buffer raises :class:`InsufficientScratchError`. Total scratch usage at ``N = 1M`` and
``BLOCK_DIM = 256`` is ``B0 = 4096`` plus ``B1 = 16`` = 4112 slots (~16 KB at ``u32``).

The ``PrefixSumExecutor`` class in ``_algorithms.py`` predates this work; it is kept for backward compat. The new
functional API is preferred for new code - see ``docs/source/user_guide/algorithms.md``.
"""

from quadrants._tensor_wrapper import Tensor
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
    _dtype_width_bytes,
    _level_partials_slots,
    _reduce_depth_for_n,
    _reduce_pass,
    _reduce_pass_u64,
    _reduce_phase,
    _scratch_dtype_for_width,
    _typed_zero_expr,
    _validate_caller_scratch,
)
from ._reduce import (
    _SUPPORTED_DTYPES as _REDUCE_SUPPORTED_DTYPES,
)

_SUPPORTED_DTYPES = _REDUCE_SUPPORTED_DTYPES  # {i32, u32, f32, i64, u64, f64}


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

    Mirrors the level-by-level allocation that the recursion does internally: at each level we bump
    ``partials_cursor`` by ``B = ceil(n / BLOCK_DIM)`` and recurse on ``B``, until ``B <= BLOCK_DIM`` (base case, no
    further partials). Callers (e.g. ``radix_sort``) should use this helper for their *up-front* scratch
    check so they refuse the call before any in-place mutation runs (see PR 693 review: a single-level estimate
    misses deeper recursion levels and lets ``_twiddle_pass`` corrupt the user's keys before the recursive
    ``RuntimeError`` fires).

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
    # Capacity comes from the caller-owned buffer we were handed (``scratch.shape[0]``); the up-front
    # ``_validate_caller_scratch`` check should already have refused an undersized buffer, so this is a backstop.
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

    Used internally by the 64-bit ``exclusive_scan_*`` path. Mirrors the 32-bit recursion shape: tile-reduce
    into ``scratch[partials_off : partials_off + B]``, recurse on those partials, then downsweep back over the
    original ``scratch[off : off + n]`` to apply per-tile prefixes.
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
    """Number of scratch slots ``exclusive_scan_{add,min,max}`` / ``*_func`` need to scan a length-``n`` input.

    The out-of-place staircase stages the per-tile partials in scratch: pass 1 writes ``B0 = ceil(n / BLOCK_DIM)``
    partials, then the in-place scan of those partials stacks ``ceil(B0 / BLOCK_DIM)`` more, ... for ``depth - 2``
    further levels (the final tile-scan writes straight back). The count is **dtype-width-independent** (slot count,
    not byte count): 4-byte scans stage through a ``u32`` scratch and 8-byte scans through a ``u64`` scratch, both of
    this many slots::

        slots = qd.algorithms.exclusive_scan_scratch_slots(N)
        scratch = qd.field(qd.u32, shape=max(slots, 1))   # u64 for i64 / u64 / f64 inputs

    Two ways to call it: **explicit depth** ``exclusive_scan_scratch_slots(n, D)`` is host- **and** kernel-callable
    (branch-free arithmetic over the unrolled ``D`` loop); **auto depth** ``exclusive_scan_scratch_slots(n)`` derives
    the minimal ``D`` from ``n`` (host-only). Returns ``0`` for ``depth <= 1`` (``n <= BLOCK_DIM``: a single tile
    scans straight to ``out`` with no scratch).
    """
    if log256_max_n is None:
        log256_max_n = _reduce_depth_for_n(n)
    if log256_max_n <= 1:
        return 0
    cursor = (n + (BLOCK_DIM - 1)) // BLOCK_DIM  # B0 partials from pass 1
    cur = cursor
    for _ in range(log256_max_n - 2):
        cur = (cur + (BLOCK_DIM - 1)) // BLOCK_DIM
        cursor = cursor + cur
    return cursor


# ---------------------------------------------------------------------------------------------------------------------
# Graph-composable exclusive scan: fixed-depth staircase of @qd.func phases (device-resident count, compile-time depth)
# ---------------------------------------------------------------------------------------------------------------------
#
# Mirrors the reduce design (see ``_reduce.py``): the friendly host entries ``exclusive_scan_{add,min,max}`` validate +
# size on the host and launch ``_exclusive_scan_kernel`` (one launch; the three-pass Blelloch scan is emitted *inside*
# the kernel as a staircase of ``@qd.func`` phases). ``exclusive_scan_{add,min,max}_func`` are the graph-composable
# counterparts: call them at the **top level** of your own ``@qd.kernel`` with a device-resident count ``n`` (i32 Expr)
# and a compile-time ``LOG256_MAX_N``, so one captured graph replays for any count ``<= BLOCK_DIM ** LOG256_MAX_N``. The
# scan is out-of-place (``arr`` -> ``out``); ``scratch`` stages the per-tile partials staircase (``bit_cast`` through
# ``u32`` / ``u64``). Same op-tree as the host driver - see ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.


def _scan_identity(dtype, OP):
    """Typed monoid-identity ``Expr`` for ``(OP, dtype)``: ``0`` for add, ``+extremum`` for min, ``-extremum`` for max.
    Trace-time helper (like :func:`_typed_min_identity`), used to seed out-of-range lanes and lane 0's predecessor in
    the block scans. Returns an ``Expr``, so its result is a valid unconditional first assignment inside a phase."""
    ident = _typed_zero_expr(dtype)
    if OP == _OP_MIN:
        return _typed_min_identity(ident)
    if OP == _OP_MAX:
        return _typed_max_identity(ident)
    return ident


@_func
def _scan_tile_phase(
    src: template(),
    dst: template(),
    src_off: i32,
    dst_off: i32,
    n: i32,
    DTYPE: template(),
    WIDE: template(),
    OP: template(),
    OP_BIN: template(),
    SRC_WIDE: template(),
    DST_WIDE: template(),
):
    """Single-block exclusive scan of ``src[src_off:src_off+n]`` -> ``dst[dst_off:dst_off+n]`` (``n <= BLOCK_DIM``).

    The recursion base / single-tile case. ``SRC_WIDE`` / ``DST_WIDE`` switch between the ``bit_cast``-through-``WIDE``
    scratch path and the direct caller-tensor path. ``v`` is seeded with the monoid identity *before* the load branch
    (Quadrants requires a variable's first assignment at the outer scope)."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        ident = _scan_identity(DTYPE, OP)
        v = ident
        if i < n:
            if static(SRC_WIDE):
                v = bit_cast(src[src_off + i], DTYPE)
            else:
                v = src[src_off + i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, OP_BIN, ident, DTYPE)
        if i < n:
            if static(DST_WIDE):
                dst[dst_off + i] = bit_cast(prefix, WIDE)
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
    DTYPE: template(),
    WIDE: template(),
    OP: template(),
    OP_BIN: template(),
    SRC_WIDE: template(),
    DST_WIDE: template(),
):
    """Per-tile exclusive scan of ``src[src_off:src_off+n]`` plus the scanned per-tile prefix
    ``prefixes[prefixes_off + block_id]``, written to ``dst[dst_off:dst_off+n]``.

    ``prefixes`` is always the ``WIDE`` scratch (the scanned partials from a lower level). ``src`` / ``dst`` switch
    between the caller tensors (``SRC_WIDE`` / ``DST_WIDE`` False) and the ``WIDE`` scratch (in-place partials scan).
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
        ident = _scan_identity(DTYPE, OP)
        v = ident
        if i < n:
            if static(SRC_WIDE):
                v = bit_cast(src[src_off + i], DTYPE)
            else:
                v = src[src_off + i]
        tile_prefix = _block.exclusive_scan(v, BLOCK_DIM, OP_BIN, ident, DTYPE)
        block_prefix = bit_cast(prefixes[prefixes_off + block_id], DTYPE)
        if i < n:
            scanned = OP_BIN(block_prefix, tile_prefix)
            if static(DST_WIDE):
                dst[dst_off + i] = bit_cast(scanned, WIDE)
            else:
                dst[dst_off + i] = scanned


def _emit_scan_inplace(buf, off, n, levels_remaining, DTYPE, WIDE, OP, OP_BIN):
    """Emit a fixed-depth in-place exclusive scan of ``buf[off:off+n]`` (``WIDE`` scratch) at kernel-compile time.

    The partials-scan half of the staircase (Blelloch pass 2): reduce ``buf[off:off+n]`` into ``buf[off+n:...]``,
    recursively scan those partials, then downsweep back. ``n`` / ``off`` flow as ``Expr`` s; ``levels_remaining`` is a
    Python int so the depth is a compile-time constant (the base is reached by exhausting it, not by inspecting ``n``).
    Reuses :func:`_reduce_phase` for the tile-reduce rung."""
    if levels_remaining == 0:
        _scan_tile_phase(buf, buf, off, off, n, DTYPE, WIDE, OP, OP_BIN, True, True)
        return
    B = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    part_off = off + n
    _reduce_phase(buf, buf, off, part_off, n, B * BLOCK_DIM, DTYPE, WIDE, OP, OP_BIN, True, True)
    _emit_scan_inplace(buf, part_off, B, levels_remaining - 1, DTYPE, WIDE, OP, OP_BIN)
    _scan_downsweep_phase(buf, buf, buf, off, part_off, off, n, B * BLOCK_DIM, DTYPE, WIDE, OP, OP_BIN, True, True)


def _emit_scan(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, OP):
    """Emit a fixed-depth (``LOG256_MAX_N``) out-of-place exclusive scan of ``arr[0:n]`` into ``out[0:n]`` at compile
    time.

    ``LOG256_MAX_N == 1`` (``n <= BLOCK_DIM``) is a single tile straight to ``out``; otherwise the three-pass scan:
    pass 1 tile-reduce ``arr`` -> ``scratch[0:B0]`` (:func:`_reduce_phase`), pass 2 in-place scan of those ``B0``
    partials (:func:`_emit_scan_inplace`, ``LOG256_MAX_N - 2`` further levels), pass 3 downsweep ``arr`` +
    ``scratch[0:B0]`` -> ``out``. An over-specified ``LOG256_MAX_N`` bottoms out at length-1 partials that scan as
    identity rungs."""
    OP_BIN = _OP_BINS[OP]  # resolve the binary op at trace time so the @qd.func phases receive it as a template
    if LOG256_MAX_N == 1:
        _scan_tile_phase(arr, out, 0, 0, n, DTYPE, WIDE, OP, OP_BIN, False, False)
        return
    B0 = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    _reduce_phase(arr, scratch, 0, 0, n, B0 * BLOCK_DIM, DTYPE, WIDE, OP, OP_BIN, False, True)
    _emit_scan_inplace(scratch, 0, B0, LOG256_MAX_N - 2, DTYPE, WIDE, OP, OP_BIN)
    _scan_downsweep_phase(arr, scratch, out, 0, 0, 0, n, B0 * BLOCK_DIM, DTYPE, WIDE, OP, OP_BIN, False, False)


@_func
def exclusive_scan_add_func(
    arr: template(), out: template(), scratch: template(), n: i32, DTYPE: template(), LOG256_MAX_N: template()
):
    """Graph-composable ``out[i] = sum(arr[0:i])`` (exclusive prefix sum) - the @qd.func form of
    :func:`exclusive_scan_add`.

    Call at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent); never nest it in ordinary
    runtime ``for`` / ``if`` / ``while`` control flow. ``n`` is a device ``Expr`` (the live count); ``DTYPE`` is the
    element dtype (pass it explicitly - an ndarray kernel arg has no in-kernel ``.dtype``); ``LOG256_MAX_N`` is the
    compile-time phase count - the scan handles any count ``<= BLOCK_DIM ** LOG256_MAX_N``. ``out`` must be distinct
    from ``arr``; size ``scratch`` via :func:`exclusive_scan_scratch_slots` ``(capacity_n, LOG256_MAX_N)``."""
    WIDE = _scratch_dtype_for_width(_dtype_width_bytes(DTYPE))  # compile-time (from the DTYPE template)
    _emit_scan(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, _OP_ADD)


@_func
def exclusive_scan_min_func(
    arr: template(), out: template(), scratch: template(), n: i32, DTYPE: template(), LOG256_MAX_N: template()
):
    """Graph-composable ``out[i] = min(arr[0:i])`` (identity ``+extremum`` from ``DTYPE``) - the @qd.func form of
    :func:`exclusive_scan_min`. See :func:`exclusive_scan_add_func` for the top-level-call contract and arg
    semantics."""
    WIDE = _scratch_dtype_for_width(_dtype_width_bytes(DTYPE))
    _emit_scan(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, _OP_MIN)


@_func
def exclusive_scan_max_func(
    arr: template(), out: template(), scratch: template(), n: i32, DTYPE: template(), LOG256_MAX_N: template()
):
    """Graph-composable ``out[i] = max(arr[0:i])`` (identity ``-extremum`` from ``DTYPE``) - the @qd.func form of
    :func:`exclusive_scan_max`. See :func:`exclusive_scan_add_func` for the top-level-call contract and arg
    semantics."""
    WIDE = _scratch_dtype_for_width(_dtype_width_bytes(DTYPE))
    _emit_scan(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, _OP_MAX)


@kernel
def _exclusive_scan_kernel(
    arr: Tensor,
    out: Tensor,
    scratch: Tensor,
    n: i32,
    DTYPE: template(),
    OP: template(),
    LOG256_MAX_N: template(),
):
    """Host-launch wrapper for the scan staircase: a thin ``@qd.kernel`` dispatching to the matching
    ``exclusive_scan_{add,min,max}_func`` at top level. The buffers are ``qd.Tensor`` params so the host entry can pass
    either a ``qd.field`` or a ``qd.ndarray`` (the qipc path). ``n`` is a plain runtime count (the host knows ``N``).
    Private - the public host entries are :func:`exclusive_scan_add` / ``_min`` / ``_max``."""
    if static(OP == _OP_MIN):
        exclusive_scan_min_func(arr, out, scratch, n, DTYPE, LOG256_MAX_N)
    elif static(OP == _OP_MAX):
        exclusive_scan_max_func(arr, out, scratch, n, DTYPE, LOG256_MAX_N)
    else:
        exclusive_scan_add_func(arr, out, scratch, n, DTYPE, LOG256_MAX_N)


def _exclusive_scan_host(arr, *, out, scratch, OP, n=None):
    """Shared host entry for ``exclusive_scan_{add,min,max}``: validate, size, and launch
    :func:`_exclusive_scan_kernel`.

    Keeps the friendly host contract (dtype / shape / no-in-place / scratch validation up front, depth + ``N`` derived
    from ``arr.shape`` or the explicit ``n``) while the device work is the single-launch graph-composable staircase.
    ``N == 1`` is handled by the staircase itself (a single tile writes the identity to ``out[0]``)."""
    if not hasattr(arr, "shape") or len(arr.shape) != 1:
        raise TypeError(f"device exclusive scan expects a 1-D input tensor; got shape {getattr(arr, 'shape', None)}")
    if not hasattr(out, "shape") or out.shape != arr.shape:
        raise TypeError(f"device exclusive scan expects out.shape == arr.shape; got arr={arr.shape}, out={out.shape}")
    if arr.dtype != out.dtype:
        raise TypeError(f"device exclusive scan dtype mismatch: arr={arr.dtype}, out={out.dtype}")
    if arr is out:
        # See design doc: in-place scan is rejected (no benefit when the caller already allocates `out` once and
        # reuses it; protecting against same-buffer aliasing would just complicate the kernels).
        raise ValueError(
            "device exclusive scan does not support in-place operation; "
            "pass a distinct `out` buffer (the API is designed around "
            "caller-supplied out, see qipc_device_algos_design.md)"
        )
    dtype = arr.dtype
    if dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"device exclusive scan dtype {dtype} not supported (need one of "
            f"{[d for d in _SUPPORTED_DTYPES]}); see design doc dtype matrix"
        )
    width = _dtype_width_bytes(dtype)
    capacity = arr.shape[0]
    N = capacity if n is None else int(n)
    if N < 0 or N > capacity:
        raise ValueError(f"exclusive_scan n={N} out of range for input of length {capacity}")
    log256_max_n = _reduce_depth_for_n(N)
    required = exclusive_scan_scratch_slots(N, log256_max_n)
    _validate_caller_scratch("exclusive_scan", N, scratch, required, _scratch_dtype_for_width(width))
    _exclusive_scan_kernel(arr, out, scratch, N, dtype, OP, log256_max_n)


def exclusive_scan_add(arr, out, scratch, *, n=None):
    """Compute ``out[i] = sum(arr[0:i])`` (exclusive prefix sum) on the device.

    Args:
        arr: 1-D tensor of any supported scalar dtype - ``{i32, u32, f32, i64, u64, f64}``. Pass a ``qd.field``,
            ``qd.ndarray``, or ``qd.Tensor`` wrapper around either.
        out: 1-D tensor with the same dtype and shape as ``arr``. Must be a distinct buffer (no in-place scan).
        scratch: caller-owned 1-D workspace of :func:`exclusive_scan_scratch_slots` ``(N)`` slots, ``u32`` for
            4-byte ``arr`` dtypes and ``u64`` for 8-byte ones (unused, so any matching-width buffer is fine, when
            ``N <= BLOCK_DIM``). There is no module-level shared scratch; a too-small buffer raises
            :class:`InsufficientScratchError`.
        n: element count to scan. Defaults to ``arr.shape[0]`` (scan the whole buffer). Pass an explicit ``n`` to scan
            only the first ``n`` slots of oversized reusable buffers (the qipc ``padded_N`` idiom); ``out`` slots past
            ``n`` are left untouched. Must satisfy ``0 <= n <= arr.shape[0]``.

    A three-pass Blelloch-style scan built on ``block.exclusive_scan``, emitted as one kernel launch (a fixed-depth
    staircase that recurses on the partials buffer when ``N > BLOCK_DIM``). To compose the scan inside your own
    ``graph=True`` parent kernel, call :func:`exclusive_scan_add_func` directly.

    See the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the algorithmic background and
    the ``bit_cast``-into-scratch scheme.
    """
    _exclusive_scan_host(arr, out=out, scratch=scratch, OP=_OP_ADD, n=n)


def exclusive_scan_min(arr, out, scratch, *, n=None):
    """Compute ``out[i] = min(arr[0:i])`` (exclusive prefix min) on the device.

    Args:
        arr: see ``exclusive_scan_add`` (any of ``{i32, u32, f32, i64, u64, f64}``).
        out: see ``exclusive_scan_add``.
        scratch: see ``exclusive_scan_add``.
        n: see ``exclusive_scan_add`` (first-``n`` of an oversized buffer; defaults to the whole buffer).

    The monoid identity is derived from ``arr.dtype`` automatically (largest representable value: ``+inf`` for
    floats, ``INT{32,64}_MAX`` for signed ints, ``UINT{32,64}_MAX`` for unsigned). Mirrors the
    ``block.exclusive_min`` / ``subgroup.exclusive_min_tiled`` contract: the typed scan primitives do not take an
    identity argument because (op, dtype) fixes it. Call :func:`exclusive_scan_min_func` to compose inside your own
    ``graph=True`` kernel.
    """
    _exclusive_scan_host(arr, out=out, scratch=scratch, OP=_OP_MIN, n=n)


def exclusive_scan_max(arr, out, scratch, *, n=None):
    """Compute ``out[i] = max(arr[0:i])`` (exclusive prefix max) on the device. Mirror of
    :func:`exclusive_scan_min` with ``max`` and the dtype's *negative* extremum (``-inf`` for floats,
    ``INT{32,64}_MIN`` for signed ints, ``0`` for unsigned), again derived from ``arr.dtype`` automatically. Call
    :func:`exclusive_scan_max_func` to compose inside your own ``graph=True`` kernel. ``n`` selects the first-``n``
    sub-range of an oversized buffer (defaults to the whole buffer).
    """
    _exclusive_scan_host(arr, out=out, scratch=scratch, OP=_OP_MAX, n=n)


# ---------------------------------------------------------------------------------------------------------------------
# Internal: u32 / add fixed-depth exclusive-scan staircase (shared with the radix-sort histogram scan)
# ---------------------------------------------------------------------------------------------------------------------
#
# The host-launched ``exclusive_scan_{add,min,max}`` above branch on ``N = arr.shape[0]`` on the *host* and launch
# ``template()`` kernels, so they cannot be composed at the top level of a ``@qd.kernel(graph=True)`` parent driven by a
# device-resident count. This block is the graph-composable building block they lack: a fixed-depth staircase of
# ``@qd.func`` phases (``u32`` / add, in place) emitted at kernel-compile time, so ``n`` flows as a device ``Expr``
# while the recursion depth is a compile-time Python int (constant launch topology). ``radix_sort`` reuses it for its
# digit-histogram scan; it is kept private until a public graph-composable scan entry point lands. See
# ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.


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
    "exclusive_scan_add_func",
    "exclusive_scan_max",
    "exclusive_scan_max_func",
    "exclusive_scan_min",
    "exclusive_scan_min_func",
    "exclusive_scan_scratch_slots",
]
