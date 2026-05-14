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
``B1 = 16`` u32 slots = 4112 slots = ~16 KB, well under the 1 MB default.

The ``PrefixSumExecutor`` class in ``_algorithms.py`` predates this work; it is kept for backward compat. The new
functional API is preferred for new code - see ``docs/source/user_guide/algorithms.md``.
"""

from quadrants._scratch import (
    get_scratch_f64,
    get_scratch_u32,
    get_scratch_u64,
    scratch_capacity_f64,
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
from quadrants.types.primitive_types import f64, i32, u32, u64

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

# f64 scan uses an f64-typed scratch instead of the u64 one because
# bit_cast(scan_result_f64, u64) loses precision in Quadrants today: the
# bits returned by block.exclusive_scan are full f64 (verified by writing
# directly to an f64 field), but routing them through bit_cast to a u64 slot
# truncates the mantissa down to f32 precision. The reduce path's bit_cast
# uses a different IR shape and is unaffected, so device_reduce_f64 still
# uses the u64 scratch. Issue tracked under qipc_gaps_device.md.

_SUPPORTED_DTYPES = _REDUCE_SUPPORTED_DTYPES  # {i32, u32, f32, i64, u64, f64}


@kernel
def _reduce_pass_f64(
    src: template(),
    dst: template(),
    src_off: i32,
    dst_off: i32,
    n_valid: i32,
    total_threads: i32,
    identity_bits: u64,
    op: template(),
    dtype: template(),
    src_is_f64: template(),
    dst_is_f64: template(),
):
    """f64-staged tile-reduce: write per-block aggregate directly to ``Field(f64)`` scratch (no ``bit_cast`` on the
    write side).

    Mirrors :func:`_reduce_pass_u64` from ``_reduce.py`` in shape, but routes the per-block aggregate through an
    f64-typed slot. ``src`` may be the user's input tensor (``src_is_f64=False``, plain read) or the f64 scratch
    (``src_is_f64=True``, plain read - no cast needed since both sides are f64). ``identity_bits`` is still passed as
    ``u64`` so the host-side identity helper can stay uniform across the 8-byte dtype paths; the kernel casts it
    via ``bit_cast(identity_bits, f64)`` on the read side (which works - the u64 -> f64 reinterpret has no
    precision loss).
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        identity = bit_cast(identity_bits, dtype)
        v = identity
        if i < n_valid:
            v = src[src_off + i]
        agg = _block.reduce(v, BLOCK_DIM, op, dtype)
        if tid == 0:
            dst[dst_off + block_id] = agg


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
def _scan_block_inplace_f64(
    buf: template(),
    buf_off: i32,
    n_valid: i32,
    identity_bits: u64,
    op: template(),
):
    """f64 sibling of :func:`_scan_block_inplace_u64`. ``buf`` is a ``Field(f64)`` so reads / writes are direct."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        identity = bit_cast(identity_bits, f64)
        v = identity
        if i < n_valid:
            v = buf[buf_off + i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, f64)
        if i < n_valid:
            buf[buf_off + i] = prefix


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


@kernel
def _scan_pass3_f64(
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
    src_is_f64: template(),
    dst_is_f64: template(),
):
    """f64 sibling of :func:`_scan_pass3_u64`. Reads / writes the f64 scratch directly (no ``bit_cast`` on either
    side). The template-switched ``src`` / ``dst`` lets the toplevel pass read from the user's input tensor and
    write to the user's output, while the recursive level uses the same f64 scratch buffer on both sides."""
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        identity = bit_cast(identity_bits, f64)
        v = identity
        if i < n_valid:
            v = src[src_off + i]
        tile_prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, f64)
        block_prefix = prefixes[prefixes_off + block_id]
        if i < n_valid:
            scanned = op(block_prefix, tile_prefix)
            dst[dst_off + i] = scanned


def _exclusive_scan_inplace_u32(scratch, off: int, n: int, identity_bits: int, op, dtype, partials_cursor: int):
    """Exclusive-scan ``scratch[off : off + n]`` in place. Recursive.

    ``partials_cursor`` is the next free u32 slot to use for the partials of
    the recursive level; the driver bumps it down each level.
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


def _exclusive_scan_inplace_f64(scratch, off: int, n: int, identity_bits: int, op, partials_cursor: int):
    """f64 sibling of :func:`_exclusive_scan_inplace_u64`. Stages through the ``Field(f64)`` scratch.

    Same recursion shape; uses :func:`_reduce_pass_f64` + :func:`_scan_pass3_f64` which write f64 directly to the
    scratch (no ``bit_cast`` on the write side, side-stepping the f64 -> u64 precision bug).
    """
    if n <= BLOCK_DIM:
        _scan_block_inplace_f64(scratch, off, n, identity_bits, op)
        return

    B = (n + BLOCK_DIM - 1) // BLOCK_DIM
    partials_off = partials_cursor
    if partials_off + B > scratch_capacity_f64():
        raise RuntimeError(
            f"device exclusive scan ran out of f64 scratch at recursion level n={n}, B={B}, "
            f"partials_off={partials_off}, capacity={scratch_capacity_f64()}. "
            f"Call _scratch.set_scratch_bytes(...) before any algorithm runs."
        )

    _reduce_pass_f64(
        scratch,
        scratch,
        off,
        partials_off,
        n,
        B * BLOCK_DIM,
        identity_bits,
        op,
        f64,
        True,
        True,
    )

    _exclusive_scan_inplace_f64(scratch, partials_off, B, identity_bits, op, partials_off + B)

    _scan_pass3_f64(
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
        True,
        True,
    )


def _device_exclusive_scan(inp, *, out, op, identity_value):
    """Internal driver shared by ``device_exclusive_scan_{add,min,max}``."""
    if not hasattr(inp, "shape") or len(inp.shape) != 1:
        raise TypeError(f"device exclusive scan expects a 1-D input tensor; got shape {getattr(inp, 'shape', None)}")
    if not hasattr(out, "shape") or out.shape != inp.shape:
        raise TypeError(
            f"device exclusive scan expects out.shape == input.shape; got input={inp.shape}, out={out.shape}"
        )
    if inp.dtype != out.dtype:
        raise TypeError(f"device exclusive scan dtype mismatch: input={inp.dtype}, out={out.dtype}")
    if inp is out:
        # See design doc: in-place scan is rejected (no benefit when the caller already allocates `out` once and
        # reuses it; protecting against same-buffer aliasing would just complicate the kernels).
        raise ValueError(
            "device exclusive scan does not support in-place operation; "
            "pass a distinct `out` buffer (the API is designed around "
            "caller-supplied out, see qipc_device_algos_design.md)"
        )

    dtype = inp.dtype
    if dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"device exclusive scan dtype {dtype} not supported (need one of "
            f"{[d for d in _SUPPORTED_DTYPES]}); see design doc dtype matrix"
        )
    width = _dtype_width_bytes(dtype)

    N = inp.shape[0]
    identity_bits = _identity_bits(identity_value, dtype)

    is_f64 = dtype == f64

    if N == 0:
        return
    if N == 1:
        if width == 4:
            _scan_trivial_n1(out, identity_bits, dtype)
        elif not is_f64:
            _scan_trivial_n1_u64(out, identity_bits, dtype)
        else:
            _scan_trivial_n1_f64(out, identity_bits)
        return

    if N <= BLOCK_DIM:
        if width == 4:
            _scan_single_tile_input_to_out(inp, out, N, identity_bits, op, dtype)
        elif not is_f64:
            _scan_single_tile_input_to_out_u64(inp, out, N, identity_bits, op, dtype)
        else:
            _scan_single_tile_input_to_out_f64(inp, out, N, identity_bits, op)
        return

    if is_f64:
        scratch = get_scratch_f64()
        scratch_cap = scratch_capacity_f64()
    elif width == 4:
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

    if is_f64:
        # Pass 1: tile-reduce input -> f64 scratch[0:B0] (no bit_cast on the write)
        _reduce_pass_f64(
            inp,
            scratch,
            0,
            0,
            N,
            B0 * BLOCK_DIM,
            identity_bits,
            op,
            f64,
            False,
            True,
        )
        # Pass 2: exclusive-scan f64 scratch[0:B0] in place
        _exclusive_scan_inplace_f64(scratch, 0, B0, identity_bits, op, B0)
        # Pass 3: input + f64 scratch[0:B0] -> out (no bit_cast on either side)
        _scan_pass3_f64(
            inp,
            0,
            scratch,
            0,
            out,
            0,
            N,
            B0 * BLOCK_DIM,
            identity_bits,
            op,
            False,
            False,
        )
        return

    reduce_pass_kernel = _reduce_pass if width == 4 else _reduce_pass_u64
    scan_inplace_driver = _exclusive_scan_inplace_u32 if width == 4 else _exclusive_scan_inplace_u64
    pass3_kernel = _scan_pass3 if width == 4 else _scan_pass3_u64

    # Pass 1: tile-reduce input -> scratch[0:B0]
    reduce_pass_kernel(
        inp,
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

    # Pass 3: input + scratch[0:B0] -> out
    pass3_kernel(
        inp,
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
def _scan_single_tile_input_to_out_f64(
    src: template(),
    dst: template(),
    n_valid: i32,
    identity_bits: u64,
    op: template(),
):
    """f64 sibling of :func:`_scan_single_tile_input_to_out_u64`.

    Functionally identical to the u64 variant for f64 dtype (no scratch involved on the write side), but spelled
    out to keep the f64 scan path free of all ``bit_cast(..., u64)`` of scan results - which is the failure mode we
    work around in the multi-tile path. Templating the dtype out also lets the kernel monomorphise tightly.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        identity = bit_cast(identity_bits, f64)
        v = identity
        if i < n_valid:
            v = src[i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, op, identity, f64)
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


@kernel
def _scan_trivial_n1_f64(dst: template(), identity_bits: u64):
    """f64 sibling of :func:`_scan_trivial_n1_u64`. Same shape; written separately to keep the f64 dispatch
    self-contained."""
    for _ in range(1):
        dst[0] = bit_cast(identity_bits, f64)


def device_exclusive_scan_add(input, out):  # pylint: disable=redefined-builtin
    """Compute ``out[i] = sum(input[0:i])`` (exclusive prefix sum) on the device.

    Args:
        input: 1-D tensor of any supported scalar dtype - ``{i32, u32, f32, i64, u64, f64}``. Pass a ``qd.field``,
            ``qd.ndarray``, or ``qd.Tensor`` wrapper around either.
        out: 1-D tensor with the same dtype and shape as ``input``. Must be a distinct buffer (no in-place scan).

    The implementation is the three-pass Blelloch-style scan built on ``block.exclusive_scan`` and the shared
    scratch fields (``Field(u32)`` for 4-byte dtypes, ``Field(u64)`` for 8-byte). Recurses on the partials buffer
    when ``N`` is large enough that the partials count exceeds ``BLOCK_DIM``.

    See the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md``
    for the algorithmic background and the ``bit_cast``-into-scratch scheme.
    """
    _device_exclusive_scan(input, out=out, op=_bin_add, identity_value=0)


def device_exclusive_scan_min(input, out):  # pylint: disable=redefined-builtin
    """Compute ``out[i] = min(input[0:i])`` (exclusive prefix min) on the device.

    Args:
        input: see ``device_exclusive_scan_add`` (any of ``{i32, u32, f32, i64, u64, f64}``).
        out: see ``device_exclusive_scan_add``.

    The monoid identity is derived from ``input.dtype`` automatically (largest representable value: ``+inf`` for
    floats, ``INT{32,64}_MAX`` for signed ints, ``UINT{32,64}_MAX`` for unsigned). Mirrors the
    ``block.exclusive_min`` / ``subgroup.exclusive_min_tiled`` contract: the typed scan primitives do not take an
    identity argument because (op, dtype) fixes it.
    """
    _device_exclusive_scan(input, out=out, op=_bin_min, identity_value=_min_identity(input.dtype))


def device_exclusive_scan_max(input, out):  # pylint: disable=redefined-builtin
    """Compute ``out[i] = max(input[0:i])`` (exclusive prefix max) on the device. Mirror of
    :func:`device_exclusive_scan_min` with ``max`` and the dtype's *negative* extremum (``-inf`` for floats,
    ``INT{32,64}_MIN`` for signed ints, ``0`` for unsigned), again derived from ``input.dtype`` automatically.
    """
    _device_exclusive_scan(input, out=out, op=_bin_max, identity_value=_max_identity(input.dtype))


__all__ = [
    "device_exclusive_scan_add",
    "device_exclusive_scan_max",
    "device_exclusive_scan_min",
]
