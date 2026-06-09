# type: ignore
"""Device-wide reduce primitives.

Implements ``qd.algorithms.reduce_{add,min,max}`` on top of the block-tier ``block.reduce_{add,min,max}``
primitives. See the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the algorithmic rationale.

Layout (host driver builds a recursion plan, kernels are the per-pass workers):

- **First pass** reads the caller's input tensor (of the algorithm's ``dtype``) and writes per-block partials to the
  shared scratch field as ``u32`` via ``qd.bit_cast``.
- **Intermediate passes** (only needed when ``N`` is large enough to require more than two passes total - i.e.
  ``B0 > BLOCK_DIM``) read from one slice of scratch (``u32`` → ``dtype`` via ``qd.bit_cast``) and write to another
  slice (``dtype`` → ``u32`` via ``qd.bit_cast``).
- **Last pass** reduces to a single value and writes it directly to the caller's ``out`` tensor as ``dtype`` (no
  bit_cast on the write side).

A single generic kernel handles every pass; ``src_is_u32`` and ``dst_is_u32`` are compile-time template flags
selecting between the bit_cast and direct-read / direct-write paths.

**Scratch.** ``reduce_*`` needs a **caller-owned** scratch buffer sized via
:func:`reduce_scratch_slots` (``~N / BLOCK_DIM`` slots; the per-block partials live there between launches).
The dtype is ``u32`` for 4-byte element types and ``u64`` for 8-byte ones (the partials are ``bit_cast`` to / from
the element dtype). There is no module-level shared scratch - the caller always owns the buffer (graph- /
multi-stream-safe, no global state); a too-small buffer raises :class:`InsufficientScratchError`.

The reduce monoid identity (e.g. ``+inf`` for ``min`` over ``f32``, ``2**31 - 1`` for ``min`` over ``i32``) is passed
to the kernel as its raw 4-byte bit pattern in a ``u32`` runtime arg, then ``qd.bit_cast``-ed to ``dtype`` inside the
kernel. This bypasses the ``default_ip`` overflow check that ``cast(literal, dtype)`` would otherwise hit on the wider
unsigned identities, and keeps ``identity`` out of the kernel template key (one fewer axis of cache fragmentation).
"""

import struct

from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.reductions import _typed_max_identity, _typed_min_identity
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


class InsufficientScratchError(RuntimeError):
    """Raised by a ``qd.algorithms.*`` device op when the caller-supplied ``scratch`` buffer is too small.

    Shared by every device algorithm (reduce, scan, select, reduce-by-key, radix sort) so callers can catch a single
    type. Subclasses ``RuntimeError`` so existing ``except RuntimeError`` / ``pytest.raises(RuntimeError)`` call sites
    keep working, while exposing the required size programmatically:

    - ``err.required_slots`` - the minimum slot count the caller must allocate (equal to the op's
      ``*_scratch_slots(N)``).
    - ``err.provided_slots`` - what was actually supplied.
    - ``err.slot_bytes`` / ``err.required_bytes`` / ``err.provided_bytes`` - byte-level view (4 for ``u32`` scratch,
      8 for ``u64`` scratch).

    This is the "try and fail with the size" path; the matching ``*_scratch_slots`` function is the "ask first" path.
    """

    def __init__(self, op: str, n: int, required_slots: int, provided_slots: int, slot_bytes: int = 4):
        self.op = op
        self.n = n
        self.required_slots = required_slots
        self.provided_slots = provided_slots
        self.slot_bytes = slot_bytes
        self.required_bytes = required_slots * slot_bytes
        self.provided_bytes = provided_slots * slot_bytes
        width = slot_bytes * 8
        super().__init__(
            f"{op} on N={n} needs >= {required_slots} u{width} scratch slots ({required_slots * slot_bytes} bytes, "
            f"including all recursion levels), but the supplied scratch holds only {provided_slots} slots "
            f"({provided_slots * slot_bytes} bytes). Allocate a 1-D u{width} scratch of at least "
            f"{op}_scratch_slots(N)={required_slots} slots (e.g. qd.field(qd.u{width}, shape={required_slots}))."
        )


def _scratch_dtype_for_width(width: int):
    """Return the scratch element dtype an algorithm of element-width ``width`` bytes expects: ``u32`` for 4-byte
    element dtypes, ``u64`` for 8-byte ones (the partials are ``bit_cast`` to / from the element dtype)."""
    return u32 if width == 4 else u64


def _validate_caller_scratch(op: str, n: int, scratch, required_slots: int, expected_dtype):
    """Validate a caller-owned ``scratch`` buffer for a device algorithm.

    Enforces the shared contract: ``scratch`` is mandatory (no module-level shared fallback), 1-D, of
    ``expected_dtype`` (``u32`` or ``u64``), and holds at least ``required_slots`` slots. A too-small buffer raises
    :class:`InsufficientScratchError` *before* the op launches anything, so partial side effects never corrupt the
    caller's inputs.
    """
    width = 4 if expected_dtype == u32 else 8
    if scratch is None:
        raise TypeError(
            f"{op} requires a caller-provided scratch buffer (there is no shared-scratch fallback); "
            f"allocate a 1-D u{width * 8} scratch of {op}_scratch_slots(N) slots"
        )
    if not hasattr(scratch, "shape") or len(scratch.shape) != 1:
        raise TypeError(f"{op} scratch must be a 1-D u{width * 8} tensor; got shape {getattr(scratch, 'shape', None)}")
    if scratch.dtype != expected_dtype:
        raise TypeError(f"{op} scratch must have dtype u{width * 8}; got {scratch.dtype}")
    if scratch.shape[0] < required_slots:
        raise InsufficientScratchError(op, n, required_slots, scratch.shape[0], slot_bytes=width)


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
# The friendly host entries ``reduce_{add,min,max}`` below validate + size on the host and launch ``_reduce_kernel``
# (one launch; the staircase is emitted *inside* the kernel). The ``reduce_{add,min,max}_func`` @qd.func forms are the
# graph-composable counterparts (the exact pattern ``radix_sort_func`` uses): call them at the **top level** of your own
# ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent), passing the live count ``n`` as a device ``Expr`` and the recursion
# depth as a compile-time ``DEPTH``. ``n`` flows dynamically while ``DEPTH`` fixes the launch topology, so one captured
# graph serves every count ``<= BLOCK_DIM ** DEPTH``. Same op-tree, identity, and ``bit_cast``-into-scratch staging as
# the old host driver - see ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.

_OP_ADD = 0
_OP_MIN = 1
_OP_MAX = 2


def _reduce_depth_for_n(n: int) -> int:
    """Minimal ``DEPTH >= 1`` such that ``BLOCK_DIM ** DEPTH >= n`` - the number of reduce phases whose final phase
    lands a single value in ``out`` (the base phase consumes ``<= BLOCK_DIM`` partials)."""
    depth = 1
    cap = BLOCK_DIM
    while cap < n:
        cap *= BLOCK_DIM
        depth += 1
    return depth


def reduce_scratch_slots(n, depth: int = None) -> int:
    """Number of scratch slots ``reduce_{add,min,max}`` / ``*_func`` need to reduce a length-``n`` input.

    The staircase stacks the per-phase partials in scratch: phase 0 writes ``ceil(n / BLOCK_DIM)`` partials, phase 1
    ``ceil(.../BLOCK_DIM)``, ... for ``depth - 1`` phases (the final phase writes the single result straight to
    ``out``). The count is **dtype-width-independent** (a 4-byte reduce stages through a ``u32`` scratch, an 8-byte one
    through ``u64``, both of this many slots)::

        slots = qd.algorithms.reduce_scratch_slots(N)
        scratch = qd.field(qd.u32, shape=max(slots, 1))   # u64 for i64 / u64 / f64 inputs

    Two ways to call it: **explicit depth** ``reduce_scratch_slots(n, D)`` is host- **and** kernel-callable (branch-free
    arithmetic over the unrolled ``D`` loop, so ``n`` may be a Python ``int`` or a device-read ``Expr``); **auto depth**
    ``reduce_scratch_slots(n)`` derives the minimal ``D`` from ``n`` (host-only). Returns ``0`` for ``depth == 1``
    (``n <= BLOCK_DIM``: the single phase writes straight to ``out``).
    """
    if depth is None:
        depth = _reduce_depth_for_n(n)
    cursor = 0
    cur = n
    for _ in range(depth - 1):
        cur = (cur + (BLOCK_DIM - 1)) // BLOCK_DIM
        cursor = cursor + cur  # ``+=`` would lower to atomic_add on a non-writable Expr in kernel scope
    return cursor


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
    SRC_WIDE: template(),
    DST_WIDE: template(),
):
    """One reduce phase: tile-reduce ``src[src_off:src_off+n]`` -> per-tile aggregates
    ``dst[dst_off:dst_off+ceil(n/BLOCK_DIM)]`` under ``OP`` (``_OP_{ADD,MIN,MAX}``).

    ``@qd.func`` phase of :func:`_emit_reduce` - its single top-level ``for`` becomes its own offloaded GPU launch (and
    graph node) when inlined into a kernel. ``SRC_WIDE`` / ``DST_WIDE`` switch between the ``qd.bit_cast``-through-``WIDE``
    (``u32`` / ``u64``) scratch path and the direct caller-tensor path (the input on the first phase, ``out`` on the
    last). Out-of-range lanes contribute the ``OP`` identity (``0`` / ``+extremum`` / ``-extremum``), derived in-kernel
    from ``DTYPE`` so no runtime identity arg is needed.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        if static(WIDE == u32):
            zero_t = bit_cast(u32(0), DTYPE)
        else:
            zero_t = bit_cast(u64(0), DTYPE)
        if static(OP == _OP_MIN):
            v = _typed_min_identity(zero_t)
        elif static(OP == _OP_MAX):
            v = _typed_max_identity(zero_t)
        else:
            v = zero_t
        if i < n:
            if static(SRC_WIDE):
                v = bit_cast(src[src_off + i], DTYPE)
            else:
                v = src[src_off + i]
        if static(OP == _OP_MIN):
            agg = _block.reduce_min(v, BLOCK_DIM, DTYPE)
        elif static(OP == _OP_MAX):
            agg = _block.reduce_max(v, BLOCK_DIM, DTYPE)
        else:
            agg = _block.reduce_add(v, BLOCK_DIM, DTYPE)
        if tid == 0:
            if static(DST_WIDE):
                dst[dst_off + block_id] = bit_cast(agg, WIDE)
            else:
                dst[dst_off + block_id] = agg


def _emit_reduce_rec(src, src_off, SRC_WIDE, scratch, cursor, out, n, phases_remaining, DTYPE, WIDE, OP):
    """Emit one rung of the reduce staircase at kernel-compile time, then recurse on the partials.

    ``n`` / ``src_off`` / ``cursor`` are Quadrants ``Expr``s (device, runtime); ``phases_remaining`` is a Python int so
    the depth - and the launch topology - is a compile-time constant. The final rung (``phases_remaining == 1``) reduces
    the remaining ``<= BLOCK_DIM`` values into ``out[0]``; earlier rungs write ``ceil(n/BLOCK_DIM)`` partials to scratch
    above ``cursor`` and recurse. An over-specified depth bottoms out at length-1 buffers that reduce as identity rungs.
    """
    if phases_remaining == 1:
        _reduce_phase(src, out, src_off, 0, n, BLOCK_DIM, DTYPE, WIDE, OP, SRC_WIDE, False)
        return
    B = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    _reduce_phase(src, scratch, src_off, cursor, n, B * BLOCK_DIM, DTYPE, WIDE, OP, SRC_WIDE, True)
    _emit_reduce_rec(scratch, cursor, True, scratch, cursor + B, out, B, phases_remaining - 1, DTYPE, WIDE, OP)


def _emit_reduce(arr, out, scratch, n, DEPTH, DTYPE, WIDE, OP):
    """Emit a fixed-depth (``DEPTH`` phases) reduce of ``arr[0:n]`` into ``out[0]``; see :func:`_emit_reduce_rec`."""
    _emit_reduce_rec(arr, 0, False, scratch, 0, out, n, DEPTH, DTYPE, WIDE, OP)


@_func
def reduce_add_func(arr: template(), out: template(), scratch: template(), n: i32, DTYPE: template(), DEPTH: template()):
    """Graph-composable ``out[0] = sum(arr[0:n])`` - the @qd.func form of :func:`reduce_add`.

    Call at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent); never nest it in ordinary
    runtime ``for`` / ``if`` / ``while`` control flow (that demotes the phase loops and drops the per-phase grid-wide
    barriers). ``n`` is a device ``Expr`` (the live count, read on-device); ``DTYPE`` is the element dtype (an ndarray
    kernel arg exposes no ``.dtype`` in-kernel, so pass it explicitly); ``DEPTH`` is the compile-time phase count - the
    emitted reduce handles any count ``<= BLOCK_DIM ** DEPTH``. Size ``scratch`` via
    :func:`reduce_scratch_slots` ``(capacity_n, DEPTH)``.
    """
    WIDE = static(_scratch_dtype_for_width(_dtype_width_bytes(DTYPE)))
    _emit_reduce(arr, out, scratch, n, DEPTH, DTYPE, WIDE, _OP_ADD)


@_func
def reduce_min_func(arr: template(), out: template(), scratch: template(), n: i32, DTYPE: template(), DEPTH: template()):
    """Graph-composable ``out[0] = min(arr[0:n])`` - the @qd.func form of :func:`reduce_min` (identity derived from
    ``DTYPE``). See :func:`reduce_add_func` for the top-level-call contract and arg semantics."""
    WIDE = static(_scratch_dtype_for_width(_dtype_width_bytes(DTYPE)))
    _emit_reduce(arr, out, scratch, n, DEPTH, DTYPE, WIDE, _OP_MIN)


@_func
def reduce_max_func(arr: template(), out: template(), scratch: template(), n: i32, DTYPE: template(), DEPTH: template()):
    """Graph-composable ``out[0] = max(arr[0:n])`` - the @qd.func form of :func:`reduce_max` (identity derived from
    ``DTYPE``). See :func:`reduce_add_func` for the top-level-call contract and arg semantics."""
    WIDE = static(_scratch_dtype_for_width(_dtype_width_bytes(DTYPE)))
    _emit_reduce(arr, out, scratch, n, DEPTH, DTYPE, WIDE, _OP_MAX)


@kernel
def _reduce_kernel(
    arr: template(),
    out: template(),
    scratch: template(),
    n: i32,
    DTYPE: template(),
    OP: template(),
    DEPTH: template(),
):
    """Host-launch wrapper for the reduce staircase: a thin ``@qd.kernel`` dispatching to the matching
    ``reduce_{add,min,max}_func`` at top level. ``n`` is a plain runtime count (the host already knows ``N``, so no
    device round-trip is needed). Private - the public host entries are :func:`reduce_add` / ``_min`` / ``_max``.
    """
    if static(OP == _OP_MIN):
        reduce_min_func(arr, out, scratch, n, DTYPE, DEPTH)
    elif static(OP == _OP_MAX):
        reduce_max_func(arr, out, scratch, n, DTYPE, DEPTH)
    else:
        reduce_add_func(arr, out, scratch, n, DTYPE, DEPTH)


def _reduce_host(arr, *, out, scratch, OP):
    """Shared host entry for ``reduce_{add,min,max}``: validate, size, and launch :func:`_reduce_kernel`.

    Keeps the friendly host contract (dtype / shape / scratch validation up front, with depth and ``N`` derived from
    ``arr.shape``) while the device work is the single-launch graph-composable staircase. The ``reduce_*_func`` forms
    skip this host check (a DtoH would defeat graph capture). ``N == 1`` is handled by the staircase itself (a single
    block reduces the lone element into ``out[0]``), so there is no separate trivial path.
    """
    if not hasattr(arr, "shape") or len(arr.shape) != 1:
        raise TypeError(f"device reduce expects a 1-D input tensor; got shape {getattr(arr, 'shape', None)}")
    if not hasattr(out, "shape") or out.shape != (1,):
        raise TypeError(f"device reduce expects out.shape == (1,); got {out.shape}")
    if arr.dtype != out.dtype:
        raise TypeError(f"device reduce dtype mismatch: arr={arr.dtype}, out={out.dtype}")
    dtype = arr.dtype
    if dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"device reduce dtype {dtype} not supported (need one of "
            f"{[d for d in _SUPPORTED_DTYPES]}); see design doc dtype matrix"
        )
    width = _dtype_width_bytes(dtype)
    N = arr.shape[0]
    depth = _reduce_depth_for_n(N)
    required = reduce_scratch_slots(N, depth)
    _validate_caller_scratch("reduce", N, scratch, required, _scratch_dtype_for_width(width))
    _reduce_kernel(arr, out, scratch, N, dtype, OP, depth)


def reduce_add(arr, out, scratch):
    """Compute ``out[0] = sum(arr)`` on the device.

    Args:
        arr: 1-D tensor of any supported scalar dtype - ``{i32, u32, f32, i64, u64, f64}``. Pass a ``qd.field``,
            ``qd.ndarray``, or ``qd.Tensor`` wrapper around either.
        out: 1-element tensor of the same dtype as ``arr``. Caller-supplied so the call is fully asynchronous - no
            implicit device-to-host sync. To get a Python scalar, do ``out.to_numpy()[0]`` explicitly after this
            call.
        scratch: caller-owned 1-D workspace of :func:`reduce_scratch_slots` ``(N)`` slots, ``u32`` for 4-byte
            ``arr`` dtypes and ``u64`` for 8-byte ones. There is no module-level shared scratch; a too-small buffer
            raises :class:`InsufficientScratchError`.

    A fixed-depth tree reduction built on ``block.reduce_add``, emitted as one kernel launch. To compose the reduce
    inside your own ``graph=True`` parent kernel, call :func:`reduce_add_func` directly. See the design doc at
    ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.
    """
    _reduce_host(arr, out=out, scratch=scratch, OP=_OP_ADD)


def reduce_min(arr, out, scratch):
    """Compute ``out[0] = min(arr)`` on the device.

    Args:
        arr: see ``reduce_add`` (any of ``{i32, u32, f32, i64, u64, f64}``).
        out: see ``reduce_add``.
        scratch: see ``reduce_add``.

    The monoid identity is derived from ``arr.dtype`` automatically (the largest representable value:
    ``+inf`` for ``f32`` / ``f64``, ``INT32_MAX`` / ``INT64_MAX`` for signed ints, ``UINT32_MAX`` / ``UINT64_MAX``
    for unsigned). Mirrors the ``block.reduce_min`` / ``subgroup.reduce_min`` contract: the typed reduce
    primitives do not take an identity argument because (op, dtype) fixes it. Call :func:`reduce_min_func` to compose
    inside your own ``graph=True`` kernel.
    """
    _reduce_host(arr, out=out, scratch=scratch, OP=_OP_MIN)


def reduce_max(arr, out, scratch):
    """Compute ``out[0] = max(arr)`` on the device. Mirror of :func:`reduce_min` with ``max`` and the
    dtype's *negative* extremum (``-inf`` for floats, ``INT32_MIN`` / ``INT64_MIN`` for signed ints, ``0`` for
    unsigned ints), again derived from ``arr.dtype`` automatically. Call :func:`reduce_max_func` to compose inside your
    own ``graph=True`` kernel.
    """
    _reduce_host(arr, out=out, scratch=scratch, OP=_OP_MAX)


__all__ = [
    "InsufficientScratchError",
    "reduce_add",
    "reduce_add_func",
    "reduce_max",
    "reduce_max_func",
    "reduce_min",
    "reduce_min_func",
    "reduce_scratch_slots",
]
