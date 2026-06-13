# type: ignore
"""Device-wide reduce primitives.

Provides the graph-composable ``qd.algorithms.reduce_{add,min,max}`` on top of the block-tier
``block.reduce_{add,min,max}`` primitives, plus :func:`reduce_scratch_slots` for sizing the caller-owned scratch. See
the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the algorithmic rationale.

Each ``reduce_{add,min,max}`` is a fixed-depth staircase of ``@qd.func`` phases (:func:`_reduce_phase`): call it at the **top
level** of your own ``@qd.kernel`` with the live element count ``n`` as a device ``Expr`` and the recursion depth as a
compile-time ``LOG256_MAX_N``, so one captured graph replays for any count ``<= BLOCK_DIM ** LOG256_MAX_N``. The
per-block partials stage through a **caller-owned** scratch buffer (``u32`` for 4-byte element dtypes, ``u64`` for
8-byte ones; ``~N / BLOCK_DIM`` slots) sized via :func:`reduce_scratch_slots`; the monoid identity (e.g. ``+inf`` for
``min`` over ``f32``) is derived in-kernel from the element dtype, so no runtime identity arg is needed.
"""

import struct

from quadrants.lang.expr import make_constant_expr
from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.reductions import (
    _bin_add,
    _bin_max,
    _bin_min,
    _typed_max_identity,
    _typed_min_identity,
)
from quadrants.types.annotations import template
from quadrants.types.primitive_types import (
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,
)

BLOCK_DIM = 256
"""Threads per block for every device reduce / scan kernel.

Chosen as a portable default: a multiple of every supported subgroup size (32 on CUDA / Vulkan-on-NV / Metal, 64 on
AMDGPU), and small enough to fit comfortably in shared memory budgets across backends. Re-tune (128 / 512) once
benchmarks land per the design doc's open questions.
"""

_MAX_SCRATCH_LEVELS = 8
"""Upper bound on the reduce/scan recursion depth that the ``*_scratch_slots`` sizing arithmetic ever unrolls.

Each level divides the live element count by ``BLOCK_DIM`` (256), so ``_MAX_SCRATCH_LEVELS = 8`` covers ``256 ** 8 ==
2 ** 64`` elements - past any addressable buffer. :func:`_level_partials_slots` loops a fixed ``range`` of this many
levels (so it is valid both host-side and inside a compiled kernel); once the count bottoms out each remaining level
contributes 0, so a generous bound costs nothing in the returned slot count.
"""


def _level_partials_slots(n, start_cursor=0):
    """Scratch slots consumed by the per-level partials of a reduce/scan over ``n`` elements, counting from
    ``start_cursor``.

    Host- **and** kernel-callable: the body is branch-free integer arithmetic over a fixed
    ``range(_MAX_SCRATCH_LEVELS)`` loop that unrolls at compile time, so ``n`` / ``start_cursor`` may be Python ``int``
    s (host sizing) **or** device-read ``Expr`` s (kernel validation). Called inside a kernel this lets a launch
    recompute the requirement from the actual device-``N`` and check it against ``scratch.shape[0]`` on-device.

    Each level turns the live count into ``t = ceil(t / BLOCK_DIM)`` partials and contributes them to scratch only
    while ``t > 1`` (equivalently ``prev > BLOCK_DIM``, since ``ceil(m / BLOCK_DIM) > 1`` iff ``m > BLOCK_DIM`` - the
    final single-tile pass writes straight to the output instead). ``t > 1`` is used as a ``0``/``1`` multiplier
    rather than a Python ``if`` (a Python ``bool`` multiplies like an ``int``; an ``Expr`` comparison like a device
    value), so the same code compiles in either context. Once ``t`` bottoms out at ``1`` every remaining iteration
    adds 0, making the fixed trip count exact rather than an over-estimate.

    **Single-reference on purpose.** Each iteration references the previous ``t`` exactly once (the ``ceil`` divide).
    When a plain function is inlined into a kernel its operand expressions are duplicated by value rather than shared,
    so a form that referenced the running count several times per level (e.g. gating *and* advancing off the same
    value) would blow the unrolled expression up ~``3 ** _MAX_SCRATCH_LEVELS`` and stall kernel compilation. The
    ``t = ceil(t / BLOCK_DIM)`` chain keeps that growth linear.
    """
    cursor = start_cursor
    t = n
    for _ in range(_MAX_SCRATCH_LEVELS):
        t = (t + BLOCK_DIM - 1) // BLOCK_DIM
        cursor = cursor + t * (t > 1)
    return cursor


_SUPPORTED_DTYPES_4B = (i32, u32, f32)
_SUPPORTED_DTYPES_8B = (i64, u64, f64)
_SUPPORTED_DTYPES = _SUPPORTED_DTYPES_4B + _SUPPORTED_DTYPES_8B


def _dtype_width_bytes(dtype) -> int:
    """Return the byte width of ``dtype``: 4 for ``{i32, u32, f32}``, 8 for ``{i64, u64, f64}``. Raises for any
    other dtype.
    """
    if dtype in _SUPPORTED_DTYPES_4B:
        return 4
    if dtype in _SUPPORTED_DTYPES_8B:
        return 8
    raise NotImplementedError(f"device reduce dtype {dtype} not supported")


def _scratch_dtype_for_width(width: int):
    """Return the scratch element dtype an algorithm of element-width ``width`` bytes expects: ``u32`` for 4-byte
    element dtypes, ``u64`` for 8-byte ones (the partials are ``bit_cast`` to / from the element dtype)."""
    return u32 if width == 4 else u64


def _identity_bits(value, dtype) -> int:
    """Reinterpret-cast ``value`` to its unsigned bit pattern: ``u32`` for 4-byte dtypes, ``u64`` for 8-byte.

    Used to ferry monoid identities (e.g. ``+inf`` for ``min`` over ``f32``, ``2**31 - 1`` for ``min`` over ``i32``,
    ``+inf`` over ``f64``, ``2**63 - 1`` over ``i64``) into the reduce kernel as a runtime arg, sidestepping the
    ``default_ip`` overflow check that ``cast(literal, dtype)`` would hit on wide unsigned identities.
    """
    if dtype == u32:
        return int(value) & 0xFFFFFFFF
    if dtype == i32:
        return struct.unpack("<I", struct.pack("<i", int(value)))[0]
    if dtype == f32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dtype == u64:
        return int(value) & 0xFFFFFFFFFFFFFFFF
    if dtype == i64:
        return struct.unpack("<Q", struct.pack("<q", int(value)))[0]
    if dtype == f64:
        return struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    raise NotImplementedError(f"identity bit-pattern for dtype {dtype} not implemented")


# Host-side per-dtype identity tables. The block / subgroup tier introspects ``value``'s dtype in-kernel
# (``_typed_min_identity(value)`` etc.) and emits a typed-constant identity Expr from there, so callers of
# ``block.reduce_min`` / ``block.exclusive_max`` etc. never have to pass an identity. The device tier mirrors
# that contract: dtype is known on the host, so we look up the identity here instead of asking the user.
_MIN_IDENTITY_BY_DTYPE = {
    i32: (1 << 31) - 1,  # INT32_MAX
    u32: (1 << 32) - 1,  # UINT32_MAX
    f32: float("inf"),
    i64: (1 << 63) - 1,  # INT64_MAX
    u64: (1 << 64) - 1,  # UINT64_MAX
    f64: float("inf"),
}

_MAX_IDENTITY_BY_DTYPE = {
    i32: -(1 << 31),  # INT32_MIN
    u32: 0,
    f32: float("-inf"),
    i64: -(1 << 63),  # INT64_MIN
    u64: 0,
    f64: float("-inf"),
}


def _min_identity(dtype):
    """Return the additive-monoid identity for ``min`` over ``dtype`` (largest representable value).

    Mirror of ``simt.reductions._typed_min_identity(value)`` from the in-kernel side, but operates on a host-known
    dtype object. Raises ``NotImplementedError`` for any dtype outside the device-tier supported set.
    """
    if dtype not in _MIN_IDENTITY_BY_DTYPE:
        raise NotImplementedError(f"device min identity for dtype {dtype} not supported")
    return _MIN_IDENTITY_BY_DTYPE[dtype]


def _max_identity(dtype):
    """Return the additive-monoid identity for ``max`` over ``dtype`` (smallest representable value).

    Mirror of ``simt.reductions._typed_max_identity(value)`` from the in-kernel side, but operates on a host-known
    dtype object. Raises ``NotImplementedError`` for any dtype outside the device-tier supported set.
    """
    if dtype not in _MAX_IDENTITY_BY_DTYPE:
        raise NotImplementedError(f"device max identity for dtype {dtype} not supported")
    return _MAX_IDENTITY_BY_DTYPE[dtype]


@kernel
def _reduce_pass(
    src: template(),
    dst: template(),
    src_off: i32,
    dst_off: i32,
    n_valid: i32,
    total_threads: i32,
    identity_bits: u32,
    op: template(),
    dtype: template(),
    src_is_u32: template(),
    dst_is_u32: template(),
):
    """Generic per-pass reduce kernel for 4-byte dtypes ({i32, u32, f32}); one pass of the multi-level driver.

    Reads ``src[src_off : src_off + n_valid]`` (out-of-range threads use the monoid identity so the per-block
    aggregate is correct), reduces in tiles of ``BLOCK_DIM`` via ``block.reduce(op, dtype)``, and writes per-block
    partials to ``dst[dst_off : dst_off + ceil(n_valid / BLOCK_DIM)]``.

    Template flags ``src_is_u32`` / ``dst_is_u32`` switch between the ``qd.bit_cast``-on-access scratch path and the
    direct-read / direct-write path used for the caller's input on the first pass and ``out`` on the last pass
    respectively.

    ``op`` is one of ``subgroup._bin_{add,min,max}`` (or any user-supplied associative ``@qd.func`` binary op of the
    right shape). ``identity_bits`` is the monoid identity for ``op`` over ``dtype``, reinterpreted as a ``u32`` (see
    ``_identity_bits`` on the host side).
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        v = bit_cast(identity_bits, dtype)
        if i < n_valid:
            if static(src_is_u32):
                v = bit_cast(src[src_off + i], dtype)
            else:
                v = src[src_off + i]
        agg = _block.reduce(v, BLOCK_DIM, op, dtype)
        if tid == 0:
            if static(dst_is_u32):
                dst[dst_off + block_id] = bit_cast(agg, u32)
            else:
                dst[dst_off + block_id] = agg


@kernel
def _reduce_pass_u64(
    src: template(),
    dst: template(),
    src_off: i32,
    dst_off: i32,
    n_valid: i32,
    total_threads: i32,
    identity_bits: u64,
    op: template(),
    dtype: template(),
    src_is_u64: template(),
    dst_is_u64: template(),
):
    """8-byte sibling of :func:`_reduce_pass` for ``{i64, u64, f64}``.

    Identical algorithmic shape, but staged through the ``Field(u64)`` scratch instead of the ``Field(u32)`` one,
    with a ``u64`` ``identity_bits`` runtime arg. Kept as a separate kernel rather than a templated single one
    because Quadrants runtime arg dtypes can't be expressed as a template of another template, and the two paths are
    short enough that the duplication is cheaper than the alternative gymnastics.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        v = bit_cast(identity_bits, dtype)
        if i < n_valid:
            if static(src_is_u64):
                v = bit_cast(src[src_off + i], dtype)
            else:
                v = src[src_off + i]
        agg = _block.reduce(v, BLOCK_DIM, op, dtype)
        if tid == 0:
            if static(dst_is_u64):
                dst[dst_off + block_id] = bit_cast(agg, u64)
            else:
                dst[dst_off + block_id] = agg


# ---------------------------------------------------------------------------------------------------------------------
# Graph-composable reduce: a fixed-depth staircase of @qd.func phases (device-resident count, compile-time depth)
# ---------------------------------------------------------------------------------------------------------------------
#
# The ``reduce_{add,min,max}`` @qd.func forms below are the graph-composable reduce (the exact pattern
# ``sort`` uses): call them at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True``
# parent), passing the live count ``n`` as a device ``Expr`` and the recursion depth as a compile-time ``LOG256_MAX_N``.
# ``n`` flows dynamically while ``LOG256_MAX_N`` fixes the launch topology, so one captured graph serves every count
# ``<= BLOCK_DIM ** LOG256_MAX_N``. The identity is derived in-kernel from the element dtype and the per-block partials
# stage through ``bit_cast``-into-scratch - see ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.

_OP_ADD = 0
_OP_MIN = 1
_OP_MAX = 2
_OP_BINS = {_OP_ADD: _bin_add, _OP_MIN: _bin_min, _OP_MAX: _bin_max}


def _typed_zero_expr(dtype):
    """A typed-constant ``0`` ``Expr`` for ``dtype`` (the additive identity). Called at trace time inside a kernel /
    func, like :func:`_typed_min_identity`. ``make_constant_expr`` insists the literal's kind match the dtype's, so
    use ``0.0`` for real dtypes and ``0`` for integers."""
    if dtype in (f32, f64):
        return make_constant_expr(0.0, dtype)
    return make_constant_expr(0, dtype)


def _reduce_depth_for_n(n: int) -> int:
    """Minimal ``LOG256_MAX_N >= 1`` such that ``BLOCK_DIM ** LOG256_MAX_N >= n`` - the number of reduce phases whose
    final phase lands a single value in ``out`` (the base phase consumes ``<= BLOCK_DIM`` partials)."""
    depth = 1
    cap = BLOCK_DIM
    while cap < n:
        cap *= BLOCK_DIM
        depth += 1
    return depth


def _at_least_one(slots):
    """Clamp a (non-negative) scratch-slot count up to a minimum of ``1``.

    The ``*_scratch_slots`` helpers legitimately compute ``0`` for the trivial / single-tile cases (a depth-1 reduce
    writes straight to ``out``, an ``n <= BLOCK_DIM`` scan needs no partials, ``n <= 0`` needs nothing). Zero-sized
    ``qd.field`` / ``qd.ndarray`` allocations are illegal, so we return at least one slot here rather than make every
    caller wrap the result in ``max(..., 1)``. The op never touches that lone slot in those cases, so over-allocating
    by one is free.

    Branch-free and dual host-/kernel-callable, matching the rest of this arithmetic: ``slots < 1`` is ``0`` / ``1``
    as a Python ``int`` (host) and an ``Expr`` (kernel, where ``n`` is a device-read count), whereas the builtin
    ``max`` would need a Python ``bool`` an ``Expr`` cannot provide. ``slots`` is always ``>= 0``, so ``slots + (slots
    < 1)`` equals ``max(slots, 1)``.
    """
    return slots + (slots < 1)


def reduce_scratch_slots(n, log256_max_n: int = None) -> int:
    """Number of scratch slots ``reduce_{add,min,max}`` need to reduce a length-``n`` input.

    The staircase stacks the per-phase partials in scratch: phase 0 writes ``ceil(n / BLOCK_DIM)`` partials, phase 1
    ``ceil(.../BLOCK_DIM)``, ... for ``depth - 1`` phases (the final phase writes the single result straight to
    ``out``). The count is **dtype-width-independent** (a 4-byte reduce stages through a ``u32`` scratch, an 8-byte one
    through ``u64``, both of this many slots)::

        slots = qd.algorithms.reduce_scratch_slots(N)
        scratch = qd.field(qd.u32, shape=slots)   # u64 for i64 / u64 / f64 inputs

    Two ways to call it: **explicit depth** ``reduce_scratch_slots(n, D)`` is host- **and** kernel-callable (branch-free
    arithmetic over the unrolled ``D`` loop, so ``n`` may be a Python ``int`` or a device-read ``Expr``); **auto depth**
    ``reduce_scratch_slots(n)`` derives the minimal ``D`` from ``n`` (host-only). Always returns **at least 1** so the
    result can size an allocation directly; the depth-1 case (``n <= BLOCK_DIM``, single phase writes straight to
    ``out``) needs no real scratch and returns ``1`` (the lone slot is never touched).
    """
    if log256_max_n is None:
        log256_max_n = _reduce_depth_for_n(n)
    cursor = 0
    cur = n
    for _ in range(log256_max_n - 1):
        cur = (cur + (BLOCK_DIM - 1)) // BLOCK_DIM
        cursor = cursor + cur  # ``+=`` would lower to atomic_add on a non-writable Expr in kernel scope
    return _at_least_one(cursor)


@_func
def _reduce_phase(
    src: template(),
    dst: template(),
    src_off: i32,
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
    """One reduce phase: tile-reduce ``src[src_off:src_off+n]`` -> per-tile aggregates
    ``dst[dst_off:dst_off+ceil(n/BLOCK_DIM)]`` under ``OP`` (``_OP_{ADD,MIN,MAX}``, with ``OP_BIN`` the matching
    ``_bin_*`` binary op).

    ``@qd.func`` phase of :func:`_emit_reduce` - its single top-level ``for`` becomes its own offloaded GPU launch (and
    graph node) when inlined into a kernel. ``SRC_WIDE`` / ``DST_WIDE`` switch between the
    ``qd.bit_cast``-through-``WIDE`` (``u32`` / ``u64``) scratch path and the direct caller-tensor path (the input on
    the first phase, ``out`` on the last). Out-of-range lanes contribute the ``OP`` identity (``0`` / ``+extremum`` /
    ``-extremum``), derived in-kernel from ``DTYPE`` so no runtime identity arg is needed. ``v`` is initialised to that
    identity *before* any branch (Quadrants requires a variable's first assignment at the outer scope; later ``if``
    branches only reassign it).
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        # Iteration-boundary barrier so a wrapped grid-stride loop does not let the next iteration's block-reduce
        # overwrite the shared-scratch slots while this iteration's reads are still in flight (WAR data race; UB on
        # all backends, corrupts on Metal / MoltenVK). See _scan._scan_downsweep_phase for the full rationale.
        _block.sync()
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        v = _typed_zero_expr(DTYPE)  # typed additive identity; unconditional first assignment
        if static(OP == _OP_MIN):
            v = _typed_min_identity(v)
        elif static(OP == _OP_MAX):
            v = _typed_max_identity(v)
        if i < n:
            if static(SRC_WIDE):
                v = bit_cast(src[src_off + i], DTYPE)
            else:
                v = src[src_off + i]
        agg = _block.reduce(v, BLOCK_DIM, OP_BIN, DTYPE)
        if tid == 0:
            if static(DST_WIDE):
                dst[dst_off + block_id] = bit_cast(agg, WIDE)
            else:
                dst[dst_off + block_id] = agg


def _emit_reduce_rec(src, src_off, SRC_WIDE, scratch, cursor, out, n, phases_remaining, DTYPE, WIDE, OP, OP_BIN):
    """Emit one rung of the reduce staircase at kernel-compile time, then recurse on the partials.

    ``n`` / ``src_off`` / ``cursor`` are Quadrants ``Expr``s (device, runtime); ``phases_remaining`` is a Python int so
    the depth - and the launch topology - is a compile-time constant. The final rung (``phases_remaining == 1``) reduces
    the remaining ``<= BLOCK_DIM`` values into ``out[0]``; earlier rungs write ``ceil(n/BLOCK_DIM)`` partials to scratch
    above ``cursor`` and recurse. An over-specified depth bottoms out at length-1 buffers that reduce as identity rungs.
    """
    if phases_remaining == 1:
        _reduce_phase(src, out, src_off, 0, n, BLOCK_DIM, DTYPE, WIDE, OP, OP_BIN, SRC_WIDE, False)
        return
    B = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    _reduce_phase(src, scratch, src_off, cursor, n, B * BLOCK_DIM, DTYPE, WIDE, OP, OP_BIN, SRC_WIDE, True)
    _emit_reduce_rec(scratch, cursor, True, scratch, cursor + B, out, B, phases_remaining - 1, DTYPE, WIDE, OP, OP_BIN)


def _emit_reduce(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, OP):
    """Emit a fixed-depth (``LOG256_MAX_N`` phases) reduce of ``arr[0:n]`` into ``out[0]``; see
    :func:`_emit_reduce_rec`."""
    OP_BIN = _OP_BINS[OP]  # resolve the binary op at trace time so the @qd.func receives it as a template
    _emit_reduce_rec(arr, 0, False, scratch, 0, out, n, LOG256_MAX_N, DTYPE, WIDE, OP, OP_BIN)


@_func
def reduce_add(
    arr: template(),
    out: template(),
    scratch: template(),
    n: i32,
    DTYPE: template(),
    LOG256_MAX_N: template(),
):
    """Graph-composable ``out[0] = sum(arr[0:n])``.

    **Experimental** - this API is new and may change in a future release.

    Call at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent); never nest it in ordinary
    runtime ``for`` / ``if`` / ``while`` control flow (that demotes the phase loops and drops the per-phase grid-wide
    barriers). ``n`` is a device ``Expr`` (the live count, read on-device); ``DTYPE`` is the element dtype (an ndarray
    kernel arg exposes no ``.dtype`` in-kernel, so pass it explicitly); ``LOG256_MAX_N`` is the compile-time phase
    count - the emitted reduce handles any count ``<= BLOCK_DIM ** LOG256_MAX_N``. Size ``scratch`` via
    :func:`reduce_scratch_slots` ``(capacity_n, LOG256_MAX_N)``.
    """
    WIDE = _scratch_dtype_for_width(_dtype_width_bytes(DTYPE))  # compile-time (from the DTYPE template); not a static()
    _emit_reduce(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, _OP_ADD)


@_func
def reduce_min(
    arr: template(),
    out: template(),
    scratch: template(),
    n: i32,
    DTYPE: template(),
    LOG256_MAX_N: template(),
):
    """Graph-composable ``out[0] = min(arr[0:n])`` (identity derived from ``DTYPE``). **Experimental** (new API, may
    change). See :func:`reduce_add` for the top-level-call contract and arg semantics."""
    WIDE = _scratch_dtype_for_width(_dtype_width_bytes(DTYPE))  # compile-time (from the DTYPE template); not a static()
    _emit_reduce(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, _OP_MIN)


@_func
def reduce_max(
    arr: template(),
    out: template(),
    scratch: template(),
    n: i32,
    DTYPE: template(),
    LOG256_MAX_N: template(),
):
    """Graph-composable ``out[0] = max(arr[0:n])`` (identity derived from ``DTYPE``). **Experimental** (new API, may
    change). See :func:`reduce_add` for the top-level-call contract and arg semantics."""
    WIDE = _scratch_dtype_for_width(_dtype_width_bytes(DTYPE))  # compile-time (from the DTYPE template); not a static()
    _emit_reduce(arr, out, scratch, n, LOG256_MAX_N, DTYPE, WIDE, _OP_MAX)


__all__ = [
    "reduce_add",
    "reduce_max",
    "reduce_min",
    "reduce_scratch_slots",
]
