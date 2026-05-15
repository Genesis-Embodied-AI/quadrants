# type: ignore
"""Device-wide exclusive-scan primitives.

Implements ``qd.algorithms.device_exclusive_scan_{add,min,max}`` on top of the block-tier ``block.exclusive_scan``
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

Total scratch usage at ``N = 1M`` and ``BLOCK_DIM = 256``: ``B0 = 4096`` plus
``B1 = 16`` u32 slots = 4112 slots = ~16 KB, trivial relative to the 5 MB default.

The ``PrefixSumExecutor`` class in ``_algorithms.py`` predates this work; it is kept for backward compat. The new
functional API is preferred for new code - see ``docs/source/user_guide/algorithms.md``.
"""

from quadrants._scratch import (
    get_scratch_u32,
    get_scratch_u64,
    scratch_capacity_u32,
    scratch_capacity_u64,
)
from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.reductions import _bin_add, _bin_max, _bin_min
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32, u64

from ._reduce import (
    _SUPPORTED_DTYPES as _REDUCE_SUPPORTED_DTYPES,
)
from ._reduce import (
    BLOCK_DIM,
    _dtype_width_bytes,
    _identity_bits,
    _max_identity,
    _min_identity,
    _reduce_pass,
    _reduce_pass_u64,
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

    Threads with ``i >= n_valid`` participate with ``identity`` (so the
    block-scope scan algorithm sees a clean monoid) but do not write back.
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
    if partials_off + B > scratch_capacity_u32():
        raise RuntimeError(
            f"device exclusive scan ran out of scratch at recursion level "
            f"n={n}, B={B}, partials_off={partials_off}, capacity="
            f"{scratch_capacity_u32()}. Call _scratch.set_scratch_bytes(...) "
            f"before any algorithm runs."
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

    Used internally by the 64-bit ``device_exclusive_scan_*`` path. Mirrors the 32-bit recursion shape: tile-reduce
    into ``scratch[partials_off : partials_off + B]``, recurse on those partials, then downsweep back over the
    original ``scratch[off : off + n]`` to apply per-tile prefixes.
    """
    if n <= BLOCK_DIM:
        _scan_block_inplace_u64(scratch, off, n, identity_bits, op, dtype)
        return

    B = (n + BLOCK_DIM - 1) // BLOCK_DIM
    partials_off = partials_cursor
    if partials_off + B > scratch_capacity_u64():
        raise RuntimeError(
            f"device exclusive scan ran out of u64 scratch at recursion level n={n}, B={B}, "
            f"partials_off={partials_off}, capacity={scratch_capacity_u64()}. "
            f"Call _scratch.set_scratch_bytes(...) before any algorithm runs."
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


def _device_exclusive_scan(arr, *, out, op, identity_value):
    """Internal driver shared by ``device_exclusive_scan_{add,min,max}``."""
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

    N = arr.shape[0]
    identity_bits = _identity_bits(identity_value, dtype)

    if N == 0:
        return
    if N == 1:
        if width == 4:
            _scan_trivial_n1(out, identity_bits, dtype)
        else:
            _scan_trivial_n1_u64(out, identity_bits, dtype)
        return

    if N <= BLOCK_DIM:
        if width == 4:
            _scan_single_tile_input_to_out(arr, out, N, identity_bits, op, dtype)
        else:
            _scan_single_tile_input_to_out_u64(arr, out, N, identity_bits, op, dtype)
        return

    if width == 4:
        scratch = get_scratch_u32()
        scratch_cap = scratch_capacity_u32()
    else:
        scratch = get_scratch_u64()
        scratch_cap = scratch_capacity_u64()
    B0 = (N + BLOCK_DIM - 1) // BLOCK_DIM
    # Reserve scratch slots: scratch[0:B0] for the top-level partials. The
    # recursive scan sub-allocates from scratch[B0:] for any deeper levels.
    if B0 > scratch_cap:
        raise RuntimeError(
            f"device exclusive scan on N={N} (dtype={dtype}) needs >= {B0} {scratch.dtype} scratch slots, "
            f"but only {scratch_cap} are configured. "
            f"Call _scratch.set_scratch_bytes(...) before any algorithm runs."
        )

    reduce_pass_kernel = _reduce_pass if width == 4 else _reduce_pass_u64
    scan_inplace_driver = _exclusive_scan_inplace_u32 if width == 4 else _exclusive_scan_inplace_u64
    pass3_kernel = _scan_pass3 if width == 4 else _scan_pass3_u64

    # Pass 1: tile-reduce arr -> scratch[0:B0]
    reduce_pass_kernel(
        arr,
        scratch,
        0,
        0,
        N,
        B0 * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        False,
        True,
    )

    # Pass 2: exclusive-scan scratch[0:B0] in place (recursive if B0 > BLOCK_DIM).
    scan_inplace_driver(scratch, 0, B0, identity_bits, op, dtype, B0)

    # Pass 3: arr + scratch[0:B0] -> out
    pass3_kernel(
        arr,
        0,
        scratch,
        0,
        out,
        0,
        N,
        B0 * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        False,
        False,
    )


@kernel
def _scan_single_tile_input_to_out(
    src: template(),
    dst: template(),
    n_valid: i32,
    identity_bits: u32,
    op: template(),
    dtype: template(),
):
    """Fast path for ``N <= BLOCK_DIM`` (4-byte dtype): one block reads the input tile,
    exclusive-scans, writes ``out``. No scratch needed."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        identity = bit_cast(identity_bits, dtype)
        v = identity
        if i < n_valid:
            v = src[i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, dtype)
        if i < n_valid:
            dst[i] = prefix


@kernel
def _scan_single_tile_input_to_out_u64(
    src: template(),
    dst: template(),
    n_valid: i32,
    identity_bits: u64,
    op: template(),
    dtype: template(),
):
    """8-byte sibling of :func:`_scan_single_tile_input_to_out`."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        identity = bit_cast(identity_bits, dtype)
        v = identity
        if i < n_valid:
            v = src[i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, dtype)
        if i < n_valid:
            dst[i] = prefix


@kernel
def _scan_trivial_n1(dst: template(), identity_bits: u32, dtype: template()):
    """N == 1 path (4-byte dtype): write the identity to out[0]. Exclusive scan of a single element is just the
    identity."""
    for _ in range(1):
        dst[0] = bit_cast(identity_bits, dtype)


@kernel
def _scan_trivial_n1_u64(dst: template(), identity_bits: u64, dtype: template()):
    """8-byte sibling of :func:`_scan_trivial_n1`."""
    for _ in range(1):
        dst[0] = bit_cast(identity_bits, dtype)


def device_exclusive_scan_add(arr, out):
    """Compute ``out[i] = sum(arr[0:i])`` (exclusive prefix sum) on the device.

    Args:
        arr: 1-D tensor of any supported scalar dtype - ``{i32, u32, f32, i64, u64, f64}``. Pass a ``qd.field``,
            ``qd.ndarray``, or ``qd.Tensor`` wrapper around either.
        out: 1-D tensor with the same dtype and shape as ``arr``. Must be a distinct buffer (no in-place scan).

    The implementation is the three-pass Blelloch-style scan built on ``block.exclusive_scan`` and the shared
    scratch fields (``Field(u32)`` for 4-byte dtypes, ``Field(u64)`` for 8-byte). Recurses on the partials buffer
    when ``N`` is large enough that the partials count exceeds ``BLOCK_DIM``.

    See the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the algorithmic background and
    the ``bit_cast``-into-scratch scheme.
    """
    _device_exclusive_scan(arr, out=out, op=_bin_add, identity_value=0)


def device_exclusive_scan_min(arr, out):
    """Compute ``out[i] = min(arr[0:i])`` (exclusive prefix min) on the device.

    Args:
        arr: see ``device_exclusive_scan_add`` (any of ``{i32, u32, f32, i64, u64, f64}``).
        out: see ``device_exclusive_scan_add``.

    The monoid identity is derived from ``arr.dtype`` automatically (largest representable value: ``+inf`` for
    floats, ``INT{32,64}_MAX`` for signed ints, ``UINT{32,64}_MAX`` for unsigned). Mirrors the
    ``block.exclusive_min`` / ``subgroup.exclusive_min_tiled`` contract: the typed scan primitives do not take an
    identity argument because (op, dtype) fixes it.
    """
    _device_exclusive_scan(arr, out=out, op=_bin_min, identity_value=_min_identity(arr.dtype))


def device_exclusive_scan_max(arr, out):
    """Compute ``out[i] = max(arr[0:i])`` (exclusive prefix max) on the device. Mirror of
    :func:`device_exclusive_scan_min` with ``max`` and the dtype's *negative* extremum (``-inf`` for floats,
    ``INT{32,64}_MIN`` for signed ints, ``0`` for unsigned), again derived from ``arr.dtype`` automatically.
    """
    _device_exclusive_scan(arr, out=out, op=_bin_max, identity_value=_max_identity(arr.dtype))


__all__ = [
    "device_exclusive_scan_add",
    "device_exclusive_scan_max",
    "device_exclusive_scan_min",
]
