# type: ignore
"""Device-wide reduce primitives.

Implements ``qd.algorithms.device_reduce_{add,min,max}`` on top of the block-tier ``block.reduce_{add,min,max}``
primitives. See the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the algorithmic rationale.

Layout (host driver builds a recursion plan, kernels are the per-pass workers):

- **First pass** reads the caller's input tensor (of the algorithm's ``dtype``) and writes per-block partials to the
  shared scratch field as ``u32`` via ``qd.bit_cast``.
- **Intermediate passes** (only needed when ``N`` is large enough to require more than two passes total — i.e.
  ``B0 > BLOCK_DIM``) read from one slice of scratch (``u32`` → ``dtype`` via ``qd.bit_cast``) and write to another
  slice (``dtype`` → ``u32`` via ``qd.bit_cast``).
- **Last pass** reduces to a single value and writes it directly to the caller's ``out`` tensor as ``dtype`` (no
  bit_cast on the write side).

A single generic kernel handles every pass; ``src_is_u32`` and ``dst_is_u32`` are compile-time template flags
selecting between the bit_cast and direct-read / direct-write paths.

The shared scratch field is owned by ``quadrants._scratch`` (see that module). For first land, the default 1 MB
capacity covers reductions up to ~64M elements at ``BLOCK_DIM=256``.

The reduce monoid identity (e.g. ``+inf`` for ``min`` over ``f32``, ``2**31 - 1`` for ``min`` over ``i32``) is passed
to the kernel as its raw 4-byte bit pattern in a ``u32`` runtime arg, then ``qd.bit_cast``-ed to ``dtype`` inside the
kernel. This bypasses the ``default_ip`` overflow check that ``cast(literal, dtype)`` would otherwise hit on the wider
unsigned identities, and keeps ``identity`` out of the kernel template key (one fewer axis of cache fragmentation).
"""

import struct

from quadrants._scratch import get_scratch_u32, scratch_capacity_u32
from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.subgroup import _bin_add, _bin_max, _bin_min
from quadrants.types.annotations import template
from quadrants.types.primitive_types import (
    f32,
    i32,
    u32,
)

BLOCK_DIM = 256
"""Threads per block for every device reduce / scan kernel.

Chosen as a portable default: a multiple of every supported subgroup size (32 on CUDA / Vulkan-on-NV / Metal, 64 on
AMDGPU), and small enough to fit comfortably in shared memory budgets across backends. Re-tune (128 / 512) once
benchmarks land per the design doc's open questions.
"""

_SUPPORTED_DTYPES = (i32, u32, f32)


def _identity_bits(value, dtype) -> int:
    """Reinterpret-cast ``value`` to its 32-bit unsigned bit pattern.

    Used to ferry monoid identities (e.g. ``+inf`` for ``min`` over ``f32``, ``2**31 - 1`` for ``min`` over ``i32``)
    into the reduce kernel as a ``u32`` runtime arg, sidestepping the ``default_ip`` overflow check that
    ``cast(literal, dtype)`` would hit on wide unsigned identities.
    """
    if dtype == u32:
        return int(value) & 0xFFFFFFFF
    if dtype == i32:
        return struct.unpack("<I", struct.pack("<i", int(value)))[0]
    if dtype == f32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    raise NotImplementedError(f"identity bit-pattern for dtype {dtype} not implemented")


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
    """Generic per-pass reduce kernel; one pass of the multi-level driver.

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


def _plan_levels(N: int):
    """Build the recursion plan for reducing N elements.

    Returns ``(sizes, dst_offsets, total_scratch)`` where:

    - ``sizes[k]`` is the number of valid input elements consumed at level ``k``; ``sizes[-1] == 1``.
    - ``dst_offsets[k]`` is the scratch offset that level ``k`` *writes* to,
      or ``-1`` for the last pass (which writes to ``out`` instead).
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


def _device_reduce(input, *, out, op, identity_value):  # pylint: disable=redefined-builtin
    """Internal driver shared by ``device_reduce_{add,min,max}``."""
    if not hasattr(input, "shape") or len(input.shape) != 1:
        raise TypeError(f"device reduce expects a 1-D input tensor; got shape {getattr(input, 'shape', None)}")
    if not hasattr(out, "shape") or out.shape != (1,):
        raise TypeError(f"device reduce expects out.shape == (1,); got {out.shape}")
    if input.dtype != out.dtype:
        raise TypeError(f"device reduce dtype mismatch: input={input.dtype}, out={out.dtype}")
    dtype = input.dtype
    if dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"device reduce dtype {dtype} not in first-land set "
            f"{[d for d in _SUPPORTED_DTYPES]}; see design doc dtype matrix"
        )

    N = input.shape[0]
    sizes, dst_offsets, total_scratch = _plan_levels(N)

    if total_scratch > scratch_capacity_u32():
        raise RuntimeError(
            f"device reduce on N={N} needs {total_scratch} u32 scratch slots, "
            f"but only {scratch_capacity_u32()} are configured. Call "
            f"quadrants._scratch.set_scratch_bytes(...) before any algorithm "
            f"runs to raise the cap."
        )

    num_passes = len(sizes) - 1
    identity_bits = _identity_bits(identity_value, dtype)

    if num_passes == 0:
        # Trivially short input (N == 0 or N == 1): no reduce kernel needed.
        # N == 0: write `identity` to out[0]. N == 1: out[0] = input[0].
        _device_reduce_trivial(input, out=out, identity_bits=identity_bits)
        return

    scratch = get_scratch_u32()

    for k in range(num_passes):
        n_in = sizes[k]
        n_out = sizes[k + 1]
        total_threads = n_out * BLOCK_DIM
        is_first = k == 0
        is_last = k == num_passes - 1
        src = input if is_first else scratch
        dst = out if is_last else scratch
        src_off = 0 if is_first else _src_off(k, dst_offsets)
        dst_off = 0 if is_last else dst_offsets[k]
        _reduce_pass(
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
def _trivial_write_input(input: template(), out: template()):  # pylint: disable=redefined-builtin
    """N == 1 path: copy input[0] to out[0]. Two-element kernel keeps the host driver loop-free for the trivial case."""
    for _ in range(1):
        out[0] = input[0]


@kernel
def _trivial_write_identity(out: template(), identity_bits: u32, dtype: template()):
    """N == 0 path: write the monoid identity (as a u32 bit pattern) to out[0].

    Quadrants doesn't support 0-shape tensors today, so this path is currently unreachable from a caller — left in
    place for defensiveness against future 0-length support.
    """
    for _ in range(1):
        out[0] = bit_cast(identity_bits, dtype)


def _device_reduce_trivial(input, *, out, identity_bits):  # pylint: disable=redefined-builtin
    N = input.shape[0]
    if N == 0:
        _trivial_write_identity(out, identity_bits, out.dtype)
    elif N == 1:
        _trivial_write_input(input, out)
    else:
        raise AssertionError(f"_device_reduce_trivial called with N={N}")


def device_reduce_add(input, *, out):  # pylint: disable=redefined-builtin
    """Compute ``out[0] = sum(input)`` on the device.

    Args:
        input: 1-D tensor of ``i32``, ``u32``, or ``f32``. Pass a ``qd.field``,
            ``qd.ndarray``, or ``qd.Tensor`` wrapper around either.
        out: 1-element tensor of the same dtype as ``input``. Caller-supplied so the call is fully asynchronous — no
            implicit device→host sync. To get a Python scalar, do ``out.to_numpy()[0]`` explicitly after this call.

    The implementation is a two-or-more-pass tree reduction built on ``block.reduce_add``. Scratch is drawn from the
    quadrants-level shared scratch field; no per-call allocation. See the design doc at
    ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the recursion plan and the ``bit_cast``-into-scratch
    scheme.
    """
    _device_reduce(input, out=out, op=_bin_add, identity_value=0)


def device_reduce_min(input, identity, *, out):  # pylint: disable=redefined-builtin
    """Compute ``out[0] = min(input)`` on the device.

    Args:
        input: see ``device_reduce_add``.
        identity: the monoid identity for ``min`` over ``input.dtype`` — i.e. a value ``e`` such that
            ``min(e, x) == x`` for every ``x`` in the dtype. For ``f32``, that's ``+inf`` (``math.inf``). For ``i32``,
            that's ``2**31 - 1``. Mandatory: there is no portable type-extreme derivable from a value alone, and
            giving callers an implicit one would silently bake in a backend assumption.
        out: see ``device_reduce_add``.
    """
    _device_reduce(input, out=out, op=_bin_min, identity_value=identity)


def device_reduce_max(input, identity, *, out):  # pylint: disable=redefined-builtin
    """Compute ``out[0] = max(input)`` on the device. Mirror of :func:`device_reduce_min` with ``max`` and the
    dtype's *negative* extremum (``-inf`` for ``f32``, ``-2**31`` for ``i32``)."""
    _device_reduce(input, out=out, op=_bin_max, identity_value=identity)


__all__ = ["device_reduce_add", "device_reduce_min", "device_reduce_max"]
