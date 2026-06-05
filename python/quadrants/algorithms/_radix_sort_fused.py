# type: ignore
"""Single-kernel (fused) device-wide LSB radix sort.

This is *the* device-wide radix sort. It emits the **entire** sort as one ``@qd.kernel`` whose body is a fixed
sequence of top-level ``for`` loops - each of which Quadrants offloads as its own serialized GPU launch, giving the
implicit grid-wide synchronization the algorithm needs between phases. (It replaces an earlier multi-launch form that
launched ``~28`` separate kernels per 4-pass u32 sort from the host.)

Three properties make the fused form graph-friendly:

1. **Fixed launch topology.** The number and order of internal launches is a compile-time constant, fixed by the
   ``log256_max_n`` template (``D``) and ``NUM_PASSES`` - *not* by the runtime ``N``. The scan over the tile
   histograms is normally host-recursive (depth ``= ceil(log256(N))``, so the launch sequence changes with ``N``);
   here it is statically unrolled to exactly ``D - 1`` reduce levels + 1 base scan + ``D - 1`` downsweep levels. Extra
   levels (when ``N`` is smaller than ``256**D``) operate on length-1 buffers and are harmless no-ops that still
   produce the correct scan (reduce of 1 -> 1, base scan of 1 -> identity, downsweep applies an identity prefix).
2. **Device-resident ``N`` + fixed (~core-count) grid.** ``N`` is read from a 0-d ``i32`` ndarray with ``N[()]``, not
   taken as a host int; ``num_blocks`` / ``hist_len`` are derived on-device. Because those sizes reach the per-phase
   loop bounds as ``Expr``s (dynamic, not compile-time constant), the CUDA codegen leaves each offload's grid at the
   *saturating* value (``num_SMs * max_blocks_per_SM * 2``) and the runtime grid-stride loop walks the actual range -
   a fixed ~core-count grid regardless of ``N``, with no ``loop_config(grid_dim=...)`` hook and no hand-rolled
   grid-stride ``while``. See ``perso_hugh/doc/qipc/qipc_sort_as_kernel.md`` §D4.
3. **Single host call.** One ``_fused_radix_sort(...)`` invocation replaces the host ping-pong loop.

Algorithm (classical histogram-scan-scatter LSB radix sort, Knuth Vol. 3 §5.2.5, Blelloch 1990). Each digit pass
(8 bits) is: per-block histogram (digit-major ``scratch[d*num_blocks+b]``) -> exclusive scan of the histograms ->
per-block rank + scatter. ``NUM_PASSES`` passes ping-pong ``keys <-> tmp_keys`` (and ``values <-> tmp_values``); an
even pass count lands the result back in ``keys`` / ``values``.

**Dtypes & twiddle.** Keys may be ``u32`` / ``i32`` / ``f32`` (32-bit, 4 passes) or ``u64`` / ``i64`` / ``f64``
(64-bit, 8 passes). Radix sort orders unsigned bit patterns; signed / float keys are mapped to a monotone unsigned
order by an in-place twiddle (first top-level ``for``) and restored by the inverse twiddle (last top-level ``for``):

- ``u32`` / ``u64``: identity (no twiddle offloads emitted).
- ``i32`` / ``i64``: XOR the sign bit.
- ``f32`` / ``f64``: positives XOR the sign bit, negatives XOR all-ones; the inverse picks masks from the *output*
  sign bit. Order matches ``numpy.sort`` (negatives before positives; NaN as numpy).

**Tail block.** When ``N % BLOCK_DIM != 0`` the last block's out-of-range threads use an all-ones sentinel key (digit
``0xFF`` for any byte) so every thread participates in ``block.radix_rank_match_atomic_or`` (which requires full
participation); histogram ``atomic_add`` and the scatter store are gated on ``i < N`` so sentinels never pollute the
histogram or write past the output.
"""

from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.lang.simt import block as _block
from quadrants.lang.simt.reductions import _bin_add
from quadrants.types import ndarray as ndarray_ann
from quadrants.types.annotations import template
from quadrants.types.primitive_types import f32, f64, i32, i64, u32, u64

from ._radix_sort import RADIX_BITS, RADIX_DIGITS, InsufficientScratchError
from ._reduce import BLOCK_DIM

_SUPPORTED_KEY_DTYPES_32 = (u32, i32, f32)
_SUPPORTED_KEY_DTYPES_64 = (u64, i64, f64)
_SUPPORTED_KEY_DTYPES = _SUPPORTED_KEY_DTYPES_32 + _SUPPORTED_KEY_DTYPES_64
_SUPPORTED_VALUE_DTYPES = (u32, i32, f32, u64, i64, f64)
_TWIDDLE_KEY_DTYPES = (i32, f32, i64, f64)


def _key_width_bits(dtype) -> int:
    if dtype in _SUPPORTED_KEY_DTYPES_32:
        return 32
    if dtype in _SUPPORTED_KEY_DTYPES_64:
        return 64
    raise NotImplementedError(f"device_radix_sort key dtype {dtype} not supported")


# --- Per-phase device bodies (one tile per block; grid sized to the work) ---------------
#
# Each helper is a ``@qd.func`` whose body is a single top-level ``for`` loop. When inlined into a ``@qd.kernel`` the
# loop becomes its own offloaded GPU launch, so calling several of these in sequence from one kernel yields the
# serialized, grid-synchronized launch chain the radix sort needs. ``i`` is the global thread index
# (``i = block_id * BLOCK_DIM + tid``). ``n`` / ``num_blocks`` arrive as device ``Expr``s (derived from the scalar
# ``N`` in the kernel), so the loop bounds are dynamic -> each offload runs on the saturating (~core-count) grid.


@_func
def _fused_twiddle(keys: template(), n: i32, KEY_DTYPE: template(), KEY_WIDTH: template(), DO_TWIDDLE: template()):
    """In-place map between caller-dtype keys and monotone-unsigned "sortable" keys (signed / float only).

    ``DO_TWIDDLE=True`` maps dtype -> sort order before the first pass; ``False`` is the inverse after the last pass.
    Only instantiated for ``i32`` / ``f32`` / ``i64`` / ``f64`` (the kernel skips it for unsigned keys).
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(n):
        if static(KEY_WIDTH == 32):
            v = bit_cast(keys[i], u32)
            if static(KEY_DTYPE == i32):
                keys[i] = bit_cast(v ^ u32(0x80000000), KEY_DTYPE)
            else:  # f32
                if static(DO_TWIDDLE):
                    if (v & u32(0x80000000)) != u32(0):
                        keys[i] = bit_cast(v ^ u32(0xFFFFFFFF), KEY_DTYPE)
                    else:
                        keys[i] = bit_cast(v ^ u32(0x80000000), KEY_DTYPE)
                else:
                    if (v & u32(0x80000000)) != u32(0):
                        keys[i] = bit_cast(v ^ u32(0x80000000), KEY_DTYPE)
                    else:
                        keys[i] = bit_cast(v ^ u32(0xFFFFFFFF), KEY_DTYPE)
        else:  # 64-bit
            w = bit_cast(keys[i], u64)
            if static(KEY_DTYPE == i64):
                keys[i] = bit_cast(w ^ u64(0x8000000000000000), KEY_DTYPE)
            else:  # f64
                if static(DO_TWIDDLE):
                    if (w & u64(0x8000000000000000)) != u64(0):
                        keys[i] = bit_cast(w ^ u64(0xFFFFFFFFFFFFFFFF), KEY_DTYPE)
                    else:
                        keys[i] = bit_cast(w ^ u64(0x8000000000000000), KEY_DTYPE)
                else:
                    if (w & u64(0x8000000000000000)) != u64(0):
                        keys[i] = bit_cast(w ^ u64(0x8000000000000000), KEY_DTYPE)
                    else:
                        keys[i] = bit_cast(w ^ u64(0xFFFFFFFFFFFFFFFF), KEY_DTYPE)


@_func
def _fused_hist(keys: template(), scratch: template(), n: i32, num_blocks: i32, bit_start: i32, KEY_WIDTH: template()):
    """Per-block histogram of digit ``(key >> bit_start) & 0xFF`` into ``scratch`` (digit-major: ``d*num_blocks+b``).

    Tile histograms are ``u32`` regardless of key width (each count <= ``BLOCK_DIM`` = 256). ``KEY_WIDTH`` selects the
    32- vs 64-bit digit extraction.
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
        if i < n:
            if static(KEY_WIDTH == 32):
                key32 = bit_cast(keys[i], u32)
                digit = i32((key32 >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
                atomic_add(hist[digit], i32(1))
            else:
                key64 = bit_cast(keys[i], u64)
                digit = i32((key64 >> u64(bit_start)) & u64(RADIX_DIGITS - 1))
                atomic_add(hist[digit], i32(1))
        _block.sync()
        if tid < RADIX_DIGITS:
            scratch[tid * num_blocks + block_id] = bit_cast(hist[tid], u32)


@_func
def _fused_reduce(scratch: template(), in_off: i32, out_off: i32, n: i32, total_threads: i32):
    """Tile-reduce ``scratch[in_off : in_off+n]`` -> per-tile sums ``scratch[out_off : out_off+ceil(n/BLOCK_DIM)]``.

    u32 / add specialization of ``_reduce._reduce_pass``. One tile per block; out-of-range lanes contribute ``0``.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        v = u32(0)
        if i < n:
            v = scratch[in_off + i]
        agg = _block.reduce_add(v, BLOCK_DIM, u32)
        if tid == 0:
            scratch[out_off + block_id] = agg


@_func
def _fused_base_scan(scratch: template(), off: i32, n_valid: i32):
    """Single-block in-place exclusive scan of ``scratch[off : off+n_valid]`` (``n_valid <= BLOCK_DIM``).

    u32 / add specialization of ``_scan._scan_block_inplace_u32``. Recursion base of the scan staircase.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(BLOCK_DIM):
        v = u32(0)
        if i < n_valid:
            v = scratch[off + i]
        prefix = _block.exclusive_scan(v, BLOCK_DIM, _bin_add, u32(0), u32)
        if i < n_valid:
            scratch[off + i] = prefix


@_func
def _fused_downsweep(scratch: template(), off: i32, part_off: i32, n: i32, total_threads: i32):
    """Downsweep: per-tile exclusive scan of ``scratch[off:off+n]`` + the scanned per-tile prefix at
    ``scratch[part_off + block_id]``, written back in place.

    u32 / add specialization of ``_scan._scan_pass3`` with ``src == dst == scratch``.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        v = u32(0)
        if i < n:
            v = scratch[off + i]
        tile_prefix = _block.exclusive_scan(v, BLOCK_DIM, _bin_add, u32(0), u32)
        block_prefix = scratch[part_off + block_id]
        if i < n:
            scratch[off + i] = block_prefix + tile_prefix


@_func
def _fused_scatter(
    keys_in: template(),
    keys_out: template(),
    values_in: template(),
    values_out: template(),
    scratch: template(),
    n: i32,
    num_blocks: i32,
    bit_start: i32,
    KEY_DTYPE: template(),
    HAS_VALUES: template(),
    KEY_WIDTH: template(),
):
    """Per-block radix rank + scatter ``keys_in[i] -> keys_out[scanned_offset + intra_digit_rank]`` (and values in
    lock-step). Mirror of ``_radix_sort._radix_scatter_pass`` (+ its u64 sibling), lifted into a func.

    For 64-bit keys the rank primitive only consumes the 8-bit digit, so we pre-extract the digit into a ``u32`` and
    feed it at ``bit_start=0``; the full-width key is what gets scattered.
    """
    loop_config(block_dim=BLOCK_DIM)
    total_threads = num_blocks * BLOCK_DIM
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        bins = _block.SharedArray((RADIX_DIGITS,), i32)
        excl_prefix = _block.SharedArray((RADIX_DIGITS,), i32)
        block_offsets = _block.SharedArray((RADIX_DIGITS,), i32)
        if static(KEY_WIDTH == 32):
            key = u32(0xFFFFFFFF)
            if i < n:
                key = bit_cast(keys_in[i], u32)
            rank = _block.radix_rank_match_atomic_or(
                key, BLOCK_DIM, RADIX_BITS, bit_start, RADIX_BITS, bins, excl_prefix
            )
            digit = i32((key >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
            if tid < RADIX_DIGITS:
                global_off = bit_cast(scratch[tid * num_blocks + block_id], i32)
                block_offsets[tid] = global_off - excl_prefix[tid]
            _block.sync()
            if i < n:
                dst = block_offsets[digit] + rank
                keys_out[dst] = bit_cast(key, KEY_DTYPE)
                if static(HAS_VALUES):
                    values_out[dst] = values_in[i]
        else:
            key = u64(0xFFFFFFFFFFFFFFFF)
            if i < n:
                key = bit_cast(keys_in[i], u64)
            digit_only_u32 = u32((key >> u64(bit_start)) & u64(RADIX_DIGITS - 1))
            rank = _block.radix_rank_match_atomic_or(
                digit_only_u32, BLOCK_DIM, RADIX_BITS, 0, RADIX_BITS, bins, excl_prefix
            )
            digit = i32(digit_only_u32)
            if tid < RADIX_DIGITS:
                global_off = bit_cast(scratch[tid * num_blocks + block_id], i32)
                block_offsets[tid] = global_off - excl_prefix[tid]
            _block.sync()
            if i < n:
                dst = block_offsets[digit] + rank
                keys_out[dst] = bit_cast(key, KEY_DTYPE)
                if static(HAS_VALUES):
                    values_out[dst] = values_in[i]


def _emit_scan_staircase(scratch, off, n, levels_remaining: int):
    """Emit a *fixed-depth* in-place exclusive scan of ``scratch[off : off+n]`` (digit-major histograms).

    Plain-Python helper run at kernel-trace time: it makes the ``@qd.func`` calls that become offloaded launches.
    ``off`` / ``n`` flow as Quadrants ``Expr``s (runtime), so the scratch offsets are tight to the actual ``N``;
    ``levels_remaining`` is a Python int (``= log256_max_n - 1``) so the recursion depth - and hence the launch
    topology - is a compile-time constant, independent of ``N``. Same shape as ``_scan._exclusive_scan_inplace_u32``
    except the ``n <= BLOCK_DIM`` base case is reached by exhausting ``levels_remaining`` rather than by inspecting
    ``n`` (forcing a constant depth).
    """
    if levels_remaining == 0:
        _fused_base_scan(scratch, off, n)
        return
    B = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    part_off = off + n
    _fused_reduce(scratch, off, part_off, n, B * BLOCK_DIM)
    _emit_scan_staircase(scratch, part_off, B, levels_remaining - 1)
    _fused_downsweep(scratch, off, part_off, n, B * BLOCK_DIM)


def _emit_pass(keys, tmp_keys, values, tmp_values, scratch, n, num_blocks, hist_len, p, KEY_DTYPE, HAS_VALUES,
               KEY_WIDTH, LOG256_MAX_N):
    """Emit one digit pass (histogram -> scan staircase -> scatter) at trace time.

    Pure-Python (runs during kernel tracing): ``p`` is a Python int from the static pass loop, so the src/dst (and
    value) ping-pong is selected here in plain Python and the field handles are passed straight into the ``@qd.func``
    calls - no field-handle assignment in the traced body (which the tracer rejects).
    """
    bit_start = p * RADIX_BITS
    src = keys if (p % 2 == 0) else tmp_keys
    dst = tmp_keys if (p % 2 == 0) else keys
    vsrc = values if (p % 2 == 0) else tmp_values
    vdst = tmp_values if (p % 2 == 0) else values
    _fused_hist(src, scratch, n, num_blocks, bit_start, KEY_WIDTH)
    _emit_scan_staircase(scratch, 0, hist_len, LOG256_MAX_N - 1)
    _fused_scatter(src, dst, vsrc, vdst, scratch, n, num_blocks, bit_start, KEY_DTYPE, HAS_VALUES, KEY_WIDTH)


@kernel
def _fused_radix_sort(
    keys: template(),
    tmp_keys: template(),
    values: template(),
    tmp_values: template(),
    scratch: template(),
    N: ndarray_ann(dtype=i32, ndim=0),
    KEY_DTYPE: template(),
    HAS_VALUES: template(),
    KEY_WIDTH: template(),
    NEEDS_TWIDDLE: template(),
    NUM_PASSES: template(),
    LOG256_MAX_N: template(),
):
    """Whole LSB radix sort as one kernel; see module docstring.

    ``N`` is a 0-d ``i32`` ndarray read once as ``N[()]``; ``num_blocks`` / ``hist_len`` are derived on-device. The
    pass loop and the scan staircase are statically unrolled (compile-time ``NUM_PASSES`` / ``LOG256_MAX_N``), so the
    launch topology is fixed regardless of ``N``. After an even ``NUM_PASSES`` the result lands in ``keys`` (and
    ``values``).
    """
    n = N[()]
    num_blocks = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    if static(NEEDS_TWIDDLE):
        _fused_twiddle(keys, n, KEY_DTYPE, KEY_WIDTH, True)
    for p in static(range(NUM_PASSES)):
        _emit_pass(keys, tmp_keys, values, tmp_values, scratch, n, num_blocks, hist_len, p, KEY_DTYPE, HAS_VALUES,
                   KEY_WIDTH, LOG256_MAX_N)
    if static(NEEDS_TWIDDLE):
        _fused_twiddle(keys, n, KEY_DTYPE, KEY_WIDTH, False)


def _min_log256_for_n(n: int) -> int:
    """Smallest ``D >= 1`` such that ``256**D >= n`` - the minimal scan depth that keeps the base-case buffer
    ``<= BLOCK_DIM`` for a length-``n`` sort."""
    d = 1
    cap = RADIX_DIGITS
    while cap < n:
        cap *= RADIX_DIGITS
        d += 1
    return d


def fused_radix_sort_scratch_slots(n, log256_max_n: int):
    """Minimum u32 scratch slots the fused sort needs for length ``n`` at depth ``log256_max_n``.

    ``hist_len = ceil(n/BLOCK_DIM) * RADIX_DIGITS`` for the tile histograms, plus the staircase partials. Because the
    staircase is forced to ``log256_max_n - 1`` reduce levels (even when ``n`` would naturally bottom out sooner),
    this can be a few slots larger than the natural recursion for small ``n`` at an over-specified ``D`` (each forced
    extra level adds 1 slot). The caller must allocate **at least** this many slots (more is fine); size against the
    provisioned upper bound on the element count (e.g. qipc's ``padded_N``), **not** ``256**log256_max_n``.

    Host- **and** kernel-callable: the body is pure ``ceil``/multiply/accumulate arithmetic and the ``log256_max_n``
    loop unrolls at trace time, so ``n`` may be a Python ``int`` (host) **or** a device-read ``Expr`` (kernel). The
    kernel path lets the sort re-check the actual device-``N`` against ``scratch.shape[0]`` on-device and raise an
    overflow flag (yield-and-realloc) instead of relying on the host knowing ``N``. ``log256_max_n`` must be a
    compile-time constant in either context.

    No ``n <= 1`` special-case: the formula already yields a valid lower bound for any ``n >= 0`` (``n = 0`` -> 0).
    """
    num_blocks = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    cursor = hist_len
    nn = hist_len
    for _ in range(log256_max_n - 1):
        B = (nn + (BLOCK_DIM - 1)) // BLOCK_DIM
        cursor += B
        nn = B
    return cursor


def _validate(keys, tmp_keys, values, tmp_values, scratch, end_bit):
    if not hasattr(keys, "shape") or len(keys.shape) != 1:
        raise TypeError(f"device_radix_sort_fused expects 1-D keys; got shape {getattr(keys, 'shape', None)}")
    if keys.dtype not in _SUPPORTED_KEY_DTYPES:
        raise NotImplementedError(
            f"device_radix_sort_fused key dtype {keys.dtype} not supported; supported: {list(_SUPPORTED_KEY_DTYPES)}"
        )
    if tmp_keys.shape != keys.shape:
        raise TypeError(f"device_radix_sort_fused expects tmp_keys.shape == keys.shape; got {keys.shape}, {tmp_keys.shape}")
    if tmp_keys.dtype != keys.dtype:
        raise TypeError(f"device_radix_sort_fused dtype mismatch: keys={keys.dtype}, tmp_keys={tmp_keys.dtype}")
    if keys is tmp_keys:
        raise ValueError("device_radix_sort_fused requires keys and tmp_keys to be distinct buffers")
    if scratch.dtype != u32:
        raise TypeError(f"device_radix_sort_fused scratch must be u32; got {scratch.dtype}")
    if (values is None) != (tmp_values is None):
        raise ValueError("device_radix_sort_fused: values and tmp_values must be passed together (both or neither)")
    if values is not None:
        if values.shape != keys.shape:
            raise TypeError(f"device_radix_sort_fused expects values.shape == keys.shape; got {keys.shape}, {values.shape}")
        if tmp_values.shape != values.shape:
            raise TypeError(
                f"device_radix_sort_fused expects tmp_values.shape == values.shape; got {values.shape}, {tmp_values.shape}"
            )
        if tmp_values.dtype != values.dtype:
            raise TypeError(f"device_radix_sort_fused dtype mismatch: values={values.dtype}, tmp_values={tmp_values.dtype}")
        if values is tmp_values:
            raise ValueError("device_radix_sort_fused requires values and tmp_values to be distinct buffers")
        if values.dtype not in _SUPPORTED_VALUE_DTYPES:
            raise NotImplementedError(
                f"device_radix_sort_fused value dtype {values.dtype} not supported; supported: {list(_SUPPORTED_VALUE_DTYPES)}"
            )
    key_width = _key_width_bits(keys.dtype)
    if end_bit <= 0 or end_bit > key_width or end_bit % RADIX_BITS != 0 or (end_bit // RADIX_BITS) % 2 != 0:
        raise ValueError(
            f"device_radix_sort_fused end_bit must be a positive even multiple of {RADIX_BITS} and <= {key_width} "
            f"(key width); got {end_bit}"
        )


def device_radix_sort_fused(
    keys, tmp_keys, scratch, values=None, tmp_values=None, end_bit=None, *, n=None, log256_max_n: int = None
):
    """Fused single-kernel ascending radix sort.

    The graph-capturable unit is the kernel :func:`_fused_radix_sort`, which reads its element count from a 0-d ``i32``
    ndarray in device memory. This eager wrapper supplies that scalar (or accepts a caller-provided one for the
    device-``N`` / graph path) and validates inputs.

    Args:
        keys: 1-D tensor of ``u32`` / ``i32`` / ``f32`` (32-bit) or ``u64`` / ``i64`` / ``f64`` (64-bit), sorted in
            place. Order matches ``numpy.sort``.
        tmp_keys: 1-D ping-pong workspace, same shape/dtype as ``keys``, distinct buffer.
        scratch: 1-D ``u32`` workspace sized via :func:`fused_radix_sort_scratch_slots`. Raises
            :class:`InsufficientScratchError` if too small.
        values: optional 1-D tensor (any supported scalar dtype), same shape as ``keys``; permuted in lock-step
            (key-value sort), in place.
        tmp_values: required iff ``values`` is given; same shape/dtype as ``values``, distinct buffer.
        end_bit: number of low key bits to sort (positive even multiple of ``RADIX_BITS``, ``<=`` key width).
            Defaults to the full key width.
        n: optional 0-d ``i32`` ndarray (``shape=()``) holding the element count on-device. If ``None``, the count is
            ``keys.shape[0]`` and a 0-d ndarray is allocated and filled with it.
        log256_max_n: compile-time scan depth ``D``. The kernel handles any element count ``<= 256**D``; a captured
            graph for a given ``D`` is reusable across all such counts. Defaults to the minimal depth for
            ``keys.shape[0]``.
    """
    import quadrants as qd

    key_width = _key_width_bits(keys.dtype)
    if end_bit is None:
        end_bit = key_width
    _validate(keys, tmp_keys, values, tmp_values, scratch, end_bit)

    capacity_n = keys.shape[0]
    if capacity_n <= 1:
        return

    if log256_max_n is None:
        log256_max_n = _min_log256_for_n(capacity_n)
    elif capacity_n > RADIX_DIGITS**log256_max_n:
        raise ValueError(
            f"device_radix_sort_fused: capacity N={capacity_n} exceeds 256**log256_max_n=256**{log256_max_n}="
            f"{RADIX_DIGITS**log256_max_n}; increase log256_max_n"
        )

    num_passes = end_bit // RADIX_BITS
    has_values = values is not None
    needs_twiddle = keys.dtype in _TWIDDLE_KEY_DTYPES
    # Placeholder tensors for the values args when there are no values, so the kernel's template key gets a real
    # tensor type; the body guards every value access on ``HAS_VALUES`` so they are never dereferenced.
    values_arg = values if has_values else keys
    tmp_values_arg = tmp_values if has_values else tmp_keys

    # Scratch is sized to the provisioned capacity; the kernel-callable ``fused_radix_sort_scratch_slots`` re-checks
    # the actual device count on-device for the graph path.
    needed = fused_radix_sort_scratch_slots(capacity_n, log256_max_n)
    if needed > scratch.shape[0]:
        raise InsufficientScratchError(capacity_n, needed, scratch.shape[0])

    if n is None:
        n = qd.ndarray(i32, shape=())
        n.fill(capacity_n)

    _fused_radix_sort(
        keys, tmp_keys, values_arg, tmp_values_arg, scratch, n,
        keys.dtype, has_values, key_width, needs_twiddle, num_passes, log256_max_n,
    )


__all__ = [
    "device_radix_sort_fused",
    "fused_radix_sort_scratch_slots",
]
