# type: ignore
"""Device-wide reduce primitives.

Implements ``qd.algorithms.device_reduce_{add,min,max}`` on top of the block-tier ``block.reduce_{add,min,max}``
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

**Scratch.** ``device_reduce_*`` needs a **caller-owned** scratch buffer sized via
:func:`device_reduce_scratch_slots` (``~N / BLOCK_DIM`` slots; the per-block partials live there between launches).
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
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.reductions import _bin_add, _bin_max, _bin_min
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


def _plan_levels(N: int):
    """Build the recursion plan for reducing N elements.

    Returns ``(sizes, dst_offsets, total_scratch)`` where:

    - ``sizes[k]`` is the number of valid input elements consumed at level ``k``; ``sizes[-1] == 1``.
    - ``dst_offsets[k]`` is the scratch offset that level ``k`` *writes* to, or ``-1`` for the last pass (which
      writes to ``out`` instead).
    - ``total_scratch`` is the total u32 slots used in scratch.

    Number of passes equals ``len(sizes) - 1``.
    """
    if N < 0:
        raise ValueError(f"N must be non-negative; got {N}")
    sizes = [N]
    while sizes[-1] > 1:
        prev = sizes[-1]
        sizes.append((prev + BLOCK_DIM - 1) // BLOCK_DIM)
    num_passes = len(sizes) - 1
    dst_offsets = []
    cumul = 0
    for k in range(num_passes):
        if k == num_passes - 1:
            dst_offsets.append(-1)
        else:
            dst_offsets.append(cumul)
            cumul += sizes[k + 1]
    return sizes, dst_offsets, cumul


def device_reduce_scratch_slots(n: int) -> int:
    """Number of scratch slots :func:`device_reduce_add` / ``_min`` / ``_max`` need to reduce a length-``n`` input.

    Pure host-side arithmetic. The count is **dtype-width-independent** (it is a slot count, not a byte count): a
    4-byte reduce stages through a ``u32`` scratch and an 8-byte reduce through a ``u64`` scratch, both of this many
    slots. Allocate the matching-width buffer up front::

        slots = qd.algorithms.device_reduce_scratch_slots(N)
        scratch = qd.field(qd.u32, shape=slots)   # u64 for i64 / u64 / f64 inputs

    Returns ``0`` for ``n <= 1`` (the reduce returns the trivial answer without touching scratch).
    """
    _, _, total_scratch = _plan_levels(n)
    return total_scratch


def _device_reduce(arr, *, out, op, identity_value, scratch):
    """Internal driver shared by ``device_reduce_{add,min,max}``.

    Dispatches on ``arr.dtype`` width: 4-byte dtypes go through the ``Field(u32)`` scratch and ``_reduce_pass``;
    8-byte dtypes go through the ``Field(u64)`` scratch and ``_reduce_pass_u64``. Everything else (control flow,
    recursion plan, identity ferrying) is shared.
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
    sizes, dst_offsets, total_scratch = _plan_levels(N)

    _validate_caller_scratch("device_reduce", N, scratch, total_scratch, _scratch_dtype_for_width(width))

    num_passes = len(sizes) - 1
    identity_bits = _identity_bits(identity_value, dtype)

    if num_passes == 0:
        # Trivially short input (N == 0 or N == 1): no reduce kernel needed. N == 0: write `identity` to out[0];
        # N == 1: out[0] = arr[0].
        _device_reduce_trivial(arr, out=out, identity_bits=identity_bits)
        return

    pass_kernel = _reduce_pass if width == 4 else _reduce_pass_u64

    for k in range(num_passes):
        n_in = sizes[k]
        n_out = sizes[k + 1]
        total_threads = n_out * BLOCK_DIM
        is_first = k == 0
        is_last = k == num_passes - 1
        src = arr if is_first else scratch
        dst = out if is_last else scratch
        src_off = 0 if is_first else _src_off(k, dst_offsets)
        dst_off = 0 if is_last else dst_offsets[k]
        pass_kernel(
            src,
            dst,
            src_off,
            dst_off,
            n_in,
            total_threads,
            identity_bits,
            op,
            dtype,
            not is_first,
            not is_last,
        )


def _src_off(k: int, dst_offsets):
    """Source offset for pass ``k`` (k >= 1): equals the dst offset that pass ``k - 1`` wrote to."""
    return dst_offsets[k - 1]


@kernel
def _trivial_write_arr(arr: template(), out: template()):
    """N == 1 path: copy arr[0] to out[0]. Two-element kernel keeps the host driver loop-free for the trivial case."""
    for _ in range(1):
        out[0] = arr[0]


@kernel
def _trivial_write_identity(out: template(), identity_bits: u32, dtype: template()):
    """N == 0 path: write the monoid identity (as a u32 bit pattern) to out[0].

    Quadrants doesn't support 0-shape tensors today, so this path is currently unreachable from a caller - left in
    place for defensiveness against future 0-length support.
    """
    for _ in range(1):
        out[0] = bit_cast(identity_bits, dtype)


def _device_reduce_trivial(arr, *, out, identity_bits):
    N = arr.shape[0]
    if N == 0:
        _trivial_write_identity(out, identity_bits, out.dtype)
    elif N == 1:
        _trivial_write_arr(arr, out)
    else:
        raise AssertionError(f"_device_reduce_trivial called with N={N}")


def device_reduce_add(arr, out, scratch):
    """Compute ``out[0] = sum(arr)`` on the device.

    Args:
        arr: 1-D tensor of any supported scalar dtype - ``{i32, u32, f32, i64, u64, f64}``. Pass a ``qd.field``,
            ``qd.ndarray``, or ``qd.Tensor`` wrapper around either.
        out: 1-element tensor of the same dtype as ``arr``. Caller-supplied so the call is fully asynchronous - no
            implicit device-to-host sync. To get a Python scalar, do ``out.to_numpy()[0]`` explicitly after this
            call.
        scratch: caller-owned 1-D workspace of :func:`device_reduce_scratch_slots` ``(N)`` slots, ``u32`` for 4-byte
            ``arr`` dtypes and ``u64`` for 8-byte ones. There is no module-level shared scratch; a too-small buffer
            raises :class:`InsufficientScratchError`.

    The implementation is a two-or-more-pass tree reduction built on ``block.reduce_add``. See the design doc at
    ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the recursion plan and the ``bit_cast``-into-scratch
    scheme.
    """
    _device_reduce(arr, out=out, op=_bin_add, identity_value=0, scratch=scratch)


def device_reduce_min(arr, out, scratch):
    """Compute ``out[0] = min(arr)`` on the device.

    Args:
        arr: see ``device_reduce_add`` (any of ``{i32, u32, f32, i64, u64, f64}``).
        out: see ``device_reduce_add``.
        scratch: see ``device_reduce_add``.

    The monoid identity is derived from ``arr.dtype`` automatically (the largest representable value:
    ``+inf`` for ``f32`` / ``f64``, ``INT32_MAX`` / ``INT64_MAX`` for signed ints, ``UINT32_MAX`` / ``UINT64_MAX``
    for unsigned). Mirrors the ``block.reduce_min`` / ``subgroup.reduce_min`` contract: the typed reduce
    primitives do not take an identity argument because (op, dtype) fixes it.
    """
    _device_reduce(arr, out=out, op=_bin_min, identity_value=_min_identity(arr.dtype), scratch=scratch)


def device_reduce_max(arr, out, scratch):
    """Compute ``out[0] = max(arr)`` on the device. Mirror of :func:`device_reduce_min` with ``max`` and the
    dtype's *negative* extremum (``-inf`` for floats, ``INT32_MIN`` / ``INT64_MIN`` for signed ints, ``0`` for
    unsigned ints), again derived from ``arr.dtype`` automatically.
    """
    _device_reduce(arr, out=out, op=_bin_max, identity_value=_max_identity(arr.dtype), scratch=scratch)


__all__ = [
    "InsufficientScratchError",
    "device_reduce_add",
    "device_reduce_max",
    "device_reduce_min",
    "device_reduce_scratch_slots",
]
