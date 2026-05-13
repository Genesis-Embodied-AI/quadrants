# type: ignore
"""Device-wide LSB radix sort.

Implements ``qd.algorithms.device_radix_sort`` on top of the block-tier ``block.radix_rank_match_atomic_or``
primitive (which is wave32 + wave64 clean since ``cd9e546851``). See the design doc at
``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the broader context and the choice of *not* using single-pass
Onesweep for first land.

Algorithm (classical histogram-scan-scatter LSB radix sort, Knuth Volume 3, Blelloch 1990 sort chapter):

Sort proceeds digit-by-digit from the least significant byte upward. Each digit pass is three internal kernel launches:

1. **Histogram pass** (``_radix_histogram_pass``). Every block computes its per-digit count for the current digit (8
   bits per pass at ``radix_bits=8``) into a 256-bin shared-memory histogram, then publishes it to global scratch laid
   out **digit-major**: ``tile_histograms[d * num_blocks + b]``.
2. **Scan pass** (reuses ``_exclusive_scan_inplace_u32`` from ``_scan.py``). In-place exclusive scan of the flat
   ``tile_histograms`` buffer. After this, ``tile_histograms[d * num_blocks + b]`` holds the global output position
   of the first key in block ``b`` whose digit equals ``d``. The digit-major layout means a single 1-D scan over the
   array suffices: the ordering "all digit-0 keys first, then digit-1, ..., and within each digit, in tile order"
   is naturally encoded.
3. **Scatter pass** (``_radix_scatter_pass``). Each block re-reads its tile, computes per-thread ranks via
   ``block.radix_rank_match_atomic_or``, looks up the per-(digit, block) global offset from the scanned
   tile_histograms, subtracts the block-local ``excl_prefix[digit]`` to obtain the intra-digit offset, and scatters
   ``keys_in[i] -> keys_out[offset + rank]``. Values, if provided, are scattered with the same indices.

After each pass we swap (``keys_in`` ↔ ``keys_out``). Four passes for ``u32`` covering bits 0-31 - even, so the final
result lands back in the caller's ``keys`` buffer.

**Twiddle for i32 / f32.** Radix sort sorts u32 bit patterns lexicographically. To get ascending ``i32`` and
``f32`` order, we apply the standard "sortable key" bit transforms before the first pass and inverse-transform after
the last pass:

- ``u32``: identity.
- ``i32``: XOR sign bit (``0x80000000``) - maps two's-complement to monotone u32.
- ``f32``: if the sign bit is clear (positive), XOR ``0x80000000``; if set (negative), XOR ``0xFFFFFFFF``. Inverse uses
  the *output* sign bit to pick the same masks back.

Both twiddle and untwiddle are in-place over ``keys``; the user's data is restored to the same dtype on return. (NaN
handling is consistent with ``numpy.sort`` for the same input, but is not separately tested as part of first land.)

**Out-of-range threads in the tail block.** When ``N % BLOCK_DIM != 0``, the final block has fewer valid keys than
threads. Out-of-range threads participate in the rank computation with a sentinel ``u32(0xFFFFFFFF)`` key (digit
``0xFF`` for any byte position), ensuring uniform control flow into ``block.radix_rank_match_atomic_or`` (which
requires every thread to participate). The histogram pass gates its atomic_add behind ``i < N``, so the sentinels do
not pollute the global histogram. The scatter pass gates its store behind ``i < N``, so the sentinels do not write
past ``keys_out``.

The ranks of valid digit-``0xFF`` keys in the tail block are unaffected by sentinels because sentinels occupy the
highest thread indices and the rank is computed stably by thread index.

**Scratch budget.** Each digit pass uses ``num_blocks * RADIX_DIGITS = N`` (rounded up to ``BLOCK_DIM`` granularity)
u32 slots in scratch for the tile_histograms, plus the partials buffers that the in-place exclusive scan introduces.
Total scratch footprint: ``≈ N * (1 + 1/256) u32 slots``. The default 1 MB scratch budget covers ``N ≤ ~260_000``;
for ``N = 1M`` (qipc's hot path) the caller must call ``quadrants._scratch.set_scratch_bytes(8 << 20)`` (or larger)
before any algorithm runs. We raise a clear error when scratch is short rather than silently scaling.
"""

from quadrants._scratch import get_scratch_u32, scratch_capacity_u32
from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.subgroup import _bin_add
from quadrants.types.annotations import template
from quadrants.types.primitive_types import f32, i32, u32

from ._reduce import BLOCK_DIM, _identity_bits
from ._scan import _exclusive_scan_inplace_u32

_SUPPORTED_KEY_DTYPES = (u32, i32, f32)
_SUPPORTED_VALUE_DTYPES = (u32, i32, f32)

RADIX_BITS = 8
"""Bits per digit. Matches the ``block.radix_rank_match_atomic_or`` constraint that ``block_dim == 1 << radix_bits``;
with ``BLOCK_DIM = 256`` this is the only legal value."""

RADIX_DIGITS = 1 << RADIX_BITS  # 256


@kernel
def _twiddle_pass(keys: template(), N: i32, dtype: template(), do_twiddle: template()):
    """In-place transform between caller-dtype keys and "sortable u32" keys.

    Set ``do_twiddle=True`` to map dtype -> u32 sort order at start of sort; ``False`` for the inverse at the end of
    sort. Both directions write through ``bit_cast`` so the storage dtype is preserved.

    The two directions are encoded by the same kernel because their bodies differ only in which sign-bit (input's or
    output's) selects the XOR mask - see the docstring on ``_radix_sort.py`` for the bit-twiddle table.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(N):
        if static(dtype == u32):
            pass
        elif static(dtype == i32):
            v = bit_cast(keys[i], u32)
            keys[i] = bit_cast(v ^ u32(0x80000000), dtype)
        else:
            v = bit_cast(keys[i], u32)
            if static(do_twiddle):
                # f32 -> sort-u32: pick mask from *input* sign bit.
                if (v & u32(0x80000000)) != u32(0):
                    keys[i] = bit_cast(v ^ u32(0xFFFFFFFF), dtype)
                else:
                    keys[i] = bit_cast(v ^ u32(0x80000000), dtype)
            else:
                # sort-u32 -> f32: pick mask from *output* sign bit, which is the
                # *opposite* of the sort-u32 sign bit (twiddle swaps them).
                if (v & u32(0x80000000)) != u32(0):
                    keys[i] = bit_cast(v ^ u32(0x80000000), dtype)
                else:
                    keys[i] = bit_cast(v ^ u32(0xFFFFFFFF), dtype)


@kernel
def _radix_histogram_pass(
    keys: template(),
    tile_histograms: template(),
    histograms_off: i32,
    N: i32,
    num_blocks: i32,
    bit_start: i32,
    dtype: template(),
):
    """Per-block histogram of digit ``(key >> bit_start) & 0xFF``.

    Writes to ``tile_histograms[histograms_off + d * num_blocks + b]``
    (digit-major layout - see module docstring on why).

    Out-of-range threads (in the tail block when ``N % BLOCK_DIM != 0``) do not contribute to the histogram. The
    shared-mem zeroing and final write-out still cover all 256 digits.
    """
    loop_config(block_dim=BLOCK_DIM)
    total_threads = num_blocks * BLOCK_DIM
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        hist = _block.SharedArray((RADIX_DIGITS,), i32)
        if tid < RADIX_DIGITS:
            hist[tid] = i32(0)
        _block.sync()
        if i < N:
            key = bit_cast(keys[i], u32)
            digit = i32((key >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
            atomic_add(hist[digit], i32(1))
        _block.sync()
        if tid < RADIX_DIGITS:
            tile_histograms[histograms_off + tid * num_blocks + block_id] = bit_cast(hist[tid], u32)


@kernel
def _radix_scatter_pass(
    keys_in: template(),
    keys_out: template(),
    values_in: template(),
    values_out: template(),
    tile_histograms: template(),
    histograms_off: i32,
    N: i32,
    num_blocks: i32,
    bit_start: i32,
    dtype: template(),
    value_dtype: template(),
    has_values: template(),
):
    """Per-block radix rank + scatter to the global output position.

    For each thread:
      - Read its key (or sentinel ``0xFFFFFFFF`` if past the tail).
      - Compute its block-local rank via ``block.radix_rank_match_atomic_or``,
        which also fills shared ``bins`` and ``excl_prefix`` arrays.
      - Compute the global destination as
        ``tile_histograms[digit * num_blocks + block_id] + (rank - excl_prefix[digit])``. (The subtraction normalizes
        ``rank`` from "position among all keys of any digit in this block" to "position among only the digit-d keys
        in this block".)
      - Scatter ``keys_in[i] -> keys_out[dst]`` and, if values were passed, ``values_in[i] -> values_out[dst]``.
    """
    loop_config(block_dim=BLOCK_DIM)
    total_threads = num_blocks * BLOCK_DIM
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        bins = _block.SharedArray((RADIX_DIGITS,), i32)
        excl_prefix = _block.SharedArray((RADIX_DIGITS,), i32)
        block_offsets = _block.SharedArray((RADIX_DIGITS,), i32)
        key = u32(0xFFFFFFFF)
        if i < N:
            key = bit_cast(keys_in[i], u32)
        rank = _block.radix_rank_match_atomic_or(key, BLOCK_DIM, RADIX_BITS, bit_start, RADIX_BITS, bins, excl_prefix)
        digit = i32((key >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
        if tid < RADIX_DIGITS:
            global_off = bit_cast(tile_histograms[histograms_off + tid * num_blocks + block_id], i32)
            block_offsets[tid] = global_off - excl_prefix[tid]
        _block.sync()
        if i < N:
            dst = block_offsets[digit] + rank
            keys_out[dst] = bit_cast(key, dtype)
            if static(has_values):
                values_out[dst] = values_in[i]


def _validate_inputs(keys, tmp_keys, values, tmp_values, end_bit):
    if not hasattr(keys, "shape") or len(keys.shape) != 1:
        raise TypeError(f"device_radix_sort expects 1-D keys; got shape {getattr(keys, 'shape', None)}")
    if not hasattr(tmp_keys, "shape") or tmp_keys.shape != keys.shape:
        raise TypeError(
            f"device_radix_sort expects tmp_keys.shape == keys.shape; got "
            f"keys={keys.shape}, tmp_keys={tmp_keys.shape}"
        )
    if tmp_keys.dtype != keys.dtype:
        raise TypeError(f"device_radix_sort dtype mismatch: keys={keys.dtype}, tmp_keys={tmp_keys.dtype}")
    if keys is tmp_keys:
        raise ValueError("device_radix_sort requires keys and tmp_keys to be distinct buffers")
    if keys.dtype not in _SUPPORTED_KEY_DTYPES:
        raise NotImplementedError(
            f"device_radix_sort key dtype {keys.dtype} not in first-land set "
            f"{[d for d in _SUPPORTED_KEY_DTYPES]}; see design doc dtype matrix"
        )

    if (values is None) != (tmp_values is None):
        raise ValueError(
            "device_radix_sort: values and tmp_values must be passed together (both or neither). "
            f"Got values={'provided' if values is not None else 'None'}, "
            f"tmp_values={'provided' if tmp_values is not None else 'None'}"
        )
    if values is not None:
        if not hasattr(values, "shape") or values.shape != keys.shape:
            raise TypeError(
                f"device_radix_sort expects values.shape == keys.shape; got "
                f"keys={keys.shape}, values={values.shape}"
            )
        if tmp_values.shape != values.shape:
            raise TypeError(
                f"device_radix_sort expects tmp_values.shape == values.shape; got "
                f"values={values.shape}, tmp_values={tmp_values.shape}"
            )
        if tmp_values.dtype != values.dtype:
            raise TypeError(f"device_radix_sort dtype mismatch: values={values.dtype}, tmp_values={tmp_values.dtype}")
        if values is tmp_values:
            raise ValueError("device_radix_sort requires values and tmp_values to be distinct buffers")
        if values.dtype not in _SUPPORTED_VALUE_DTYPES:
            raise NotImplementedError(
                f"device_radix_sort value dtype {values.dtype} not in first-land set "
                f"{[d for d in _SUPPORTED_VALUE_DTYPES]}; see design doc dtype matrix"
            )

    if end_bit <= 0 or end_bit > 32:
        raise ValueError(f"device_radix_sort end_bit must satisfy 0 < end_bit <= 32; got {end_bit}")
    if end_bit % RADIX_BITS != 0:
        raise ValueError(
            f"device_radix_sort end_bit must be a multiple of {RADIX_BITS} so that an even number of digit passes "
            f"leaves the result back in `keys`; got end_bit={end_bit}"
        )
    num_passes = end_bit // RADIX_BITS
    if num_passes % 2 != 0:
        raise ValueError(
            f"device_radix_sort needs an even number of digit passes (so the ping-pong lands back in `keys`); "
            f"got num_passes={num_passes} for end_bit={end_bit}, RADIX_BITS={RADIX_BITS}"
        )


def device_radix_sort(keys, *, tmp_keys, values=None, tmp_values=None, end_bit=32):  # pylint: disable=too-many-locals
    """Sort ``keys`` ascending on the device using LSB radix sort.

    Args:
        keys: 1-D tensor of ``u32``, ``i32``, or ``f32``. Sorted in place.
            Pass a ``qd.field``, ``qd.ndarray``, or ``qd.Tensor`` wrapper.
        tmp_keys: 1-D tensor with the same shape and dtype as ``keys``, distinct buffer. Used as a ping-pong workspace;
            its contents at return are intermediate and should be considered garbage.
        values: optional 1-D tensor of ``u32`` / ``i32`` / ``f32``, same shape as ``keys``. If provided, values are
            permuted in lock-step with keys (key-value sort), in place.
        tmp_values: required iff ``values`` is provided. Same shape and dtype as ``values``, distinct buffer; same
            workspace semantics as ``tmp_keys``.
        end_bit: number of low bits of the key to consider (default 32 = entire u32 range). Must be a non-zero multiple
            of ``RADIX_BITS = 8`` so that an even number of digit passes leaves the result in ``keys``.

    Sort order matches ``numpy.sort`` for ascending sort (``i32`` two's-complement, ``f32`` IEEE-754 with negatives
    ordered before positives, NaN handling matches numpy).

    Built on ``block.radix_rank_match_atomic_or`` (which is wave64-clean as of ``cd9e546851``) + the shared
    ``Field(u32)`` scratch. The first land is classical histogram-scan-scatter LSB; a single-pass decoupled-lookback
    variant (Onesweep) is a perf follow-up if profiling shows sort in the top of qipc's frame budget.

    **Scratch budget**: requires ``ceil(N / BLOCK_DIM) * RADIX_DIGITS + ...`` u32 slots in the shared scratch (see
    module docstring on ``_radix_sort.py`` for the exact formula). For the default 1 MB scratch, that caps ``N`` at
    ~260_000. Raise the budget via ``quadrants._scratch.set_scratch_bytes(...)`` before any algorithm call if you need
    larger sorts. ``N = 1M`` needs ~5 MB.
    """
    _validate_inputs(keys, tmp_keys, values, tmp_values, end_bit)
    N = keys.shape[0]
    if N <= 1:
        return

    key_dtype = keys.dtype
    has_values = values is not None
    # Provide a non-None placeholder for values_* even when has_values=False so the kernel's template-key includes a
    # real tensor type; the kernel body itself guards on `has_values` so the tensors are never actually dereferenced.
    values_in_arg = values if has_values else keys
    tmp_values_arg = tmp_values if has_values else tmp_keys
    value_dtype = values.dtype if has_values else key_dtype

    num_blocks = (N + BLOCK_DIM - 1) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS  # u32 slots for the per-pass tile_histograms

    scratch = get_scratch_u32()
    cap = scratch_capacity_u32()
    # Scratch layout: scratch[0 : hist_len] = current pass's tile_histograms. The in-place scan over scratch[0 :
    # hist_len] sub-allocates partials from scratch[hist_len : ...] up to its own recursion depth. Worst case the scan
    # needs roughly `hist_len / BLOCK_DIM + ...` extra u32 slots - well under hist_len / 100.
    if hist_len + (hist_len + BLOCK_DIM - 1) // BLOCK_DIM > cap:
        needed = hist_len + (hist_len + BLOCK_DIM - 1) // BLOCK_DIM
        raise RuntimeError(
            f"device_radix_sort on N={N} needs >= {needed} u32 scratch slots "
            f"({needed * 4} bytes), but only {cap} are configured "
            f"({cap * 4} bytes). Call quadrants._scratch.set_scratch_bytes(...) "
            f"before any algorithm runs to raise the cap. For N=1M expect to "
            f"need ~5 MB; for N=10M ~50 MB."
        )

    # Pre-twiddle keys (in-place) for i32 / f32. u32 path is a no-op.
    if key_dtype in (i32, f32):
        _twiddle_pass(keys, N, key_dtype, True)

    identity_bits = _identity_bits(0, u32)
    src = keys
    dst = tmp_keys
    src_values = values_in_arg
    dst_values = tmp_values_arg
    num_passes = end_bit // RADIX_BITS
    for p in range(num_passes):
        bit_start = p * RADIX_BITS
        # Pass A: per-block histograms into scratch[0 : hist_len].
        _radix_histogram_pass(src, scratch, 0, N, num_blocks, bit_start, key_dtype)
        # Pass B: in-place exclusive scan of scratch[0 : hist_len].
        _exclusive_scan_inplace_u32(scratch, 0, hist_len, identity_bits, _bin_add, u32, hist_len)
        # Pass C: scatter from src -> dst using the scanned histograms.
        _radix_scatter_pass(
            src,
            dst,
            src_values,
            dst_values,
            scratch,
            0,
            N,
            num_blocks,
            bit_start,
            key_dtype,
            value_dtype,
            has_values,
        )
        src, dst = dst, src
        src_values, dst_values = dst_values, src_values

    # After an even number of swaps, the sorted result is back in `keys`.
    if key_dtype in (i32, f32):
        _twiddle_pass(keys, N, key_dtype, False)


__all__ = ["device_radix_sort"]
