# type: ignore
"""Device-wide LSB radix sort (one capturable launch chain).

The sort is exposed as two public callables sharing one device body (the same design B split as the other
``qd.algorithms.*`` ops - a friendly host entry plus a graph-composable ``@qd.func``):

- :func:`radix_sort` - the friendly **host (Python) entry**: validates inputs, derives the compile-time params
  (pass count / scan depth / twiddle) and the device-resident count, then launches the sort as one capturable kernel
  chain (via the private ``_radix_sort_kernel`` -> :func:`radix_sort_func`). Use this to sort from the host.
- :func:`radix_sort_func` - a ``@qd.func``: call it at the **top level** of your own ``@qd.kernel`` to compose the
  sort inline with other phases in one compiled kernel / captured graph (the qipc path); it takes the count as a
  device-resident 0-d ``N`` and the params as templates, and does no host-side validation.

The body is a fixed sequence of top-level ``for`` loops - each of which Quadrants offloads as its own serialized GPU
launch, giving the implicit grid-wide synchronization the algorithm needs between phases. (This replaces an earlier
multi-launch form that launched ``~28`` separate kernels per 4-pass u32 sort from the host.)

Three properties make this form graph-friendly:

1. **Fixed launch topology.** The number and order of internal launches is a compile-time constant, fixed by the
   ``log256_max_n`` template (``D``) and the pass count - *not* by the runtime ``N``. The scan over the tile
   histograms is normally host-recursive (depth ``= ceil(log256(N))``, so the launch sequence changes with ``N``);
   here it is statically unrolled to exactly ``D - 1`` reduce levels + 1 base scan + ``D - 1`` downsweep levels. Extra
   levels (when ``N`` is smaller than ``256**D``) operate on length-1 buffers and are harmless no-ops that still
   produce the correct scan.
2. **Device-resident ``N`` + fixed (~core-count) grid.** ``N`` is read from a 0-d ``i32`` ndarray with ``N[()]``, not
   taken as a host int; ``num_blocks`` / ``hist_len`` are derived on-device. Because those sizes reach the per-phase
   loop bounds as ``Expr``s (dynamic, not compile-time constant), the CUDA codegen leaves each offload's grid at the
   *saturating* value (``num_SMs * max_blocks_per_SM * 2``) and the runtime grid-stride loop walks the actual range -
   a fixed ~core-count grid regardless of ``N``. See ``perso_hugh/doc/qipc/qipc_sort_as_kernel.md`` §D4.
3. **Single call.** One :func:`radix_sort_func` invocation replaces the host ping-pong loop.

Algorithm (classical histogram-scan-scatter LSB radix sort, Knuth Vol. 3 §5.2.5, Blelloch 1990). Each digit pass
(8 bits) is: per-block histogram (digit-major ``scratch[d*num_blocks+b]``) -> exclusive scan of the histograms ->
per-block rank + scatter. The pass count (4 for 32-bit keys, 8 for 64-bit) ping-pongs ``keys <-> tmp_keys`` (and
``values <-> tmp_values``); an even pass count lands the result back in ``keys`` / ``values``.

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

**Scratch.** The sort needs a **caller-owned** 1-D ``u32`` ``scratch`` buffer of
:func:`radix_sort_scratch_slots` ``(N, log256_max_n)`` slots (tile histograms + scan partials; ``u32`` regardless of
key width, so 8-byte-key sorts have the same footprint as 4-byte ones). There is **no** module-level shared-scratch
fallback - the caller always owns the buffer (graph- / multi-stream-safe, no global state). The friendly host
:func:`radix_sort` validates ``scratch`` against :func:`radix_sort_scratch_slots` up front (raising
:class:`InsufficientScratchError` before launching); :func:`radix_sort_func` does **no** on-device scratch check (a
DtoH would defeat graph capture), so size ``scratch`` correctly up front when composing the func directly.
"""

from quadrants.lang.impl import ndarray, static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.lang.simt import block as _block
from quadrants._tensor_wrapper import Tensor
from quadrants.types import ndarray as ndarray_ann
from quadrants.types.annotations import template
from quadrants.types.primitive_types import f32, f64, i32, i64, u32, u64

from ._reduce import BLOCK_DIM, InsufficientScratchError, _validate_caller_scratch
from ._scan import _emit_exclusive_scan_add

RADIX_BITS = 8
"""Bits per digit. Matches the ``block.radix_rank_match_atomic_or`` constraint that ``block_dim == 1 << radix_bits``;
with ``BLOCK_DIM = 256`` this is the only legal value."""

RADIX_DIGITS = 1 << RADIX_BITS  # 256

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
    raise NotImplementedError(f"radix_sort key dtype {dtype} not supported")


# --- Per-phase device bodies (one tile per block; grid sized to the work) ---------------
#
# Each helper is a ``@qd.func`` whose body is a single top-level ``for`` loop. When inlined into a ``@qd.kernel`` the
# loop becomes its own offloaded GPU launch, so calling several of these in sequence from one kernel yields the
# serialized, grid-synchronized launch chain the radix sort needs. ``i`` is the global thread index
# (``i = block_id * BLOCK_DIM + tid``). ``n`` / ``num_blocks`` arrive as device ``Expr``s (derived from the scalar
# ``N`` in the kernel), so the loop bounds are dynamic -> each offload runs on the saturating (~core-count) grid.


@_func
def _radix_twiddle(keys: template(), n: i32, KEY_DTYPE: template(), KEY_WIDTH: template(), DO_TWIDDLE: template()):
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
def _radix_hist(keys: template(), scratch: template(), n: i32, num_blocks: i32, bit_start: i32, KEY_WIDTH: template()):
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
def _radix_scatter(
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
    lock-step).

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


def _emit_pass(keys, tmp_keys, values, tmp_values, scratch, n, num_blocks, hist_len, p, KEY_DTYPE, HAS_VALUES,
               KEY_WIDTH, LOG256_MAX_N):
    """Emit one digit pass (histogram -> scan staircase -> scatter) at compile time.

    Pure-Python (runs during kernel compilation): ``p`` is a Python int from the static pass loop, so the src/dst
    (and value) ping-pong is selected here in plain Python and the field handles are passed straight into the
    ``@qd.func`` calls - no field-handle assignment in the compiled body (which the compiler frontend rejects).

    The histogram scan reuses the shared graph-composable staircase :func:`._scan._emit_exclusive_scan_add` (``u32`` /
    add, in place, fixed depth). ``LOG256_MAX_N - 1`` reduce levels makes the digit-major ``scratch[0:hist_len]`` scan a
    compile-time-constant launch topology, independent of the device-resident ``N``.
    """
    bit_start = p * RADIX_BITS
    src = keys if (p % 2 == 0) else tmp_keys
    dst = tmp_keys if (p % 2 == 0) else keys
    vsrc = values if (p % 2 == 0) else tmp_values
    vdst = tmp_values if (p % 2 == 0) else values
    _radix_hist(src, scratch, n, num_blocks, bit_start, KEY_WIDTH)
    _emit_exclusive_scan_add(scratch, 0, hist_len, LOG256_MAX_N - 1)
    _radix_scatter(src, dst, vsrc, vdst, scratch, n, num_blocks, bit_start, KEY_DTYPE, HAS_VALUES, KEY_WIDTH)


@_func
def radix_sort_func(
    keys: template(),
    tmp_keys: template(),
    values: template(),
    tmp_values: template(),
    scratch: template(),
    N: template(),
    KEY_DTYPE: template(),
    HAS_VALUES: template(),
    END_BIT: template(),
    LOG256_MAX_N: template(),
):
    """Whole LSB radix sort as one ``@qd.func`` - the composable form; see module docstring.

    Call it at the **top level** of your own ``@qd.kernel`` (e.g. :func:`radix_sort` below, or a qipc ``graph=True``
    parent that chains it with other phases). Each phase helper's single top-level ``for`` stays its own offloaded GPU
    launch, so the inter-phase grid-wide synchronization survives and every phase is captured as a node in the parent's
    graph. **A ``while qd.graph_do_while(...):`` body counts as top level** - the loops directly inside it still lower as
    separate offloaded launches with grid-wide barriers between them, so calling this func directly in a
    ``graph_do_while`` body is supported and re-sorts correctly every iteration (verified for ``N`` spanning many
    blocks). What you **must not** do is nest the call inside *ordinary* runtime control flow - another ``for``, an
    ``if``, or a plain ``while`` - which demotes the phase loops out of top-level position, collapses the per-phase
    grid-wide barriers and corrupts the sort. Compile-time ``static`` loops (like the pass loop here) are also fine.

    Compile-time params: ``KEY_DTYPE`` (the key element dtype, one of ``{u32, i32, f32, u64, i64, f64}``), ``HAS_VALUES``
    (whether ``values`` / ``tmp_values`` are real buffers or placeholders), ``END_BIT`` (low key bits to sort - positive
    even multiple of ``8``, ``<=`` key width), and ``LOG256_MAX_N`` (scan depth ``D``; the emitted sort handles any
    count ``<= 256 ** D``). The width, pass count and twiddle need are derived from ``KEY_DTYPE`` + ``END_BIT`` at
    compile time. ``KEY_DTYPE`` is an explicit param because an ``ndarray`` kernel argument (the qipc path) exposes no
    ``.dtype`` inside the kernel - pass the dtype you already know (the :func:`radix_sort` wrapper derives it from its
    field / ndarray argument for you).

    ``N`` is a 0-d ``i32`` ndarray handle read once as ``N[()]``; ``num_blocks`` / ``hist_len`` are derived on-device.
    The pass loop and scan staircase are statically unrolled, so the launch topology is fixed regardless of ``N``;
    after an even pass count the result lands in ``keys`` (and ``values``). The caller owns ``scratch`` (size it with
    :func:`radix_sort_scratch_slots` ``(capacity_N, LOG256_MAX_N)``) and the device-resident ``N``. There is **no**
    host-side validation or scratch-sufficiency check (a DtoH would defeat graph capture) - pass distinct, same-shape
    buffers and size ``scratch`` correctly up front.
    """
    KEY_WIDTH = static(_key_width_bits(KEY_DTYPE))
    NUM_PASSES = static(END_BIT // RADIX_BITS)
    NEEDS_TWIDDLE = static(KEY_DTYPE in _TWIDDLE_KEY_DTYPES)
    n = N[()]
    num_blocks = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    if static(NEEDS_TWIDDLE):
        _radix_twiddle(keys, n, KEY_DTYPE, KEY_WIDTH, True)
    for p in static(range(NUM_PASSES)):
        _emit_pass(keys, tmp_keys, values, tmp_values, scratch, n, num_blocks, hist_len, p, KEY_DTYPE, HAS_VALUES,
                   KEY_WIDTH, LOG256_MAX_N)
    if static(NEEDS_TWIDDLE):
        _radix_twiddle(keys, n, KEY_DTYPE, KEY_WIDTH, False)


@kernel
def _radix_sort_kernel(
    keys: Tensor,
    tmp_keys: Tensor,
    values: Tensor,
    tmp_values: Tensor,
    scratch: Tensor,
    N: ndarray_ann(dtype=i32, ndim=0),
    KEY_DTYPE: template(),
    HAS_VALUES: template(),
    END_BIT: template(),
    LOG256_MAX_N: template(),
):
    """Host-launch wrapper for :func:`radix_sort_func` - a thin ``@qd.kernel`` (private; the public host entry is
    :func:`radix_sort`).

    A ``@qd.func`` cannot be launched from the host, so this kernel just calls the func at top level, which inlines it
    and keeps every phase as its own offload. The buffers are ``qd.Tensor`` params so the host entry can pass either a
    ``qd.field`` or a ``qd.ndarray`` (the qipc path); an ndarray kernel arg has no in-kernel ``.dtype``, so the key
    dtype rides along as the ``KEY_DTYPE`` template (the host derives it from the buffer argument). ``N`` is a 0-d
    ``i32`` ndarray holding the element count on-device (the friendly :func:`radix_sort` host builds and fills it);
    ``HAS_VALUES`` / ``END_BIT`` / ``LOG256_MAX_N`` are the compile-time params it derives.
    """
    radix_sort_func(keys, tmp_keys, values, tmp_values, scratch, N, KEY_DTYPE, HAS_VALUES, END_BIT, LOG256_MAX_N)


def _validate_radix_inputs(keys, tmp_keys, values, tmp_values, end_bit):
    """Host-side validation for the friendly :func:`radix_sort` entry (mirrors the checks the other ``qd.algorithms.*``
    host entries do): supported key dtype, distinct same-shape ``tmp_keys``, consistent ``values`` / ``tmp_values``
    pair, and a legal ``end_bit``."""
    if not hasattr(keys, "shape") or len(keys.shape) != 1:
        raise TypeError(f"radix_sort expects a 1-D keys tensor; got shape {getattr(keys, 'shape', None)}")
    if keys.dtype not in _SUPPORTED_KEY_DTYPES:
        raise NotImplementedError(
            f"radix_sort key dtype {keys.dtype} not supported (need one of {[d for d in _SUPPORTED_KEY_DTYPES]})"
        )
    if not hasattr(tmp_keys, "shape") or tmp_keys.shape != keys.shape or tmp_keys.dtype != keys.dtype:
        raise TypeError(
            f"radix_sort expects tmp_keys to match keys (shape {keys.shape}, dtype {keys.dtype}); "
            f"got shape {getattr(tmp_keys, 'shape', None)}, dtype {getattr(tmp_keys, 'dtype', None)}"
        )
    if keys is tmp_keys:
        raise ValueError("radix_sort needs a distinct tmp_keys buffer (ping-pong scratch); got keys is tmp_keys")
    if (values is None) != (tmp_values is None):
        raise ValueError("radix_sort: pass both values and tmp_values, or neither")
    if values is not None:
        if not hasattr(values, "shape") or values.shape != keys.shape:
            raise TypeError(f"radix_sort expects values.shape == keys.shape ({keys.shape}); got {values.shape}")
        if values.dtype not in _SUPPORTED_VALUE_DTYPES:
            raise NotImplementedError(
                f"radix_sort value dtype {values.dtype} not supported "
                f"(need one of {[d for d in _SUPPORTED_VALUE_DTYPES]})"
            )
        if not hasattr(tmp_values, "shape") or tmp_values.shape != keys.shape or tmp_values.dtype != values.dtype:
            raise TypeError(
                f"radix_sort expects tmp_values to match values (shape {keys.shape}, dtype {values.dtype}); "
                f"got shape {getattr(tmp_values, 'shape', None)}, dtype {getattr(tmp_values, 'dtype', None)}"
            )
        if values is tmp_values:
            raise ValueError("radix_sort needs a distinct tmp_values buffer; got values is tmp_values")
    if end_bit is not None:
        if end_bit <= 0 or end_bit % RADIX_BITS != 0 or end_bit > _key_width_bits(keys.dtype):
            raise ValueError(
                f"radix_sort end_bit must be a positive multiple of {RADIX_BITS} and <= the key width "
                f"({_key_width_bits(keys.dtype)} bits); got {end_bit}"
            )


def radix_sort(keys, tmp_keys, scratch, *, values=None, tmp_values=None, end_bit=None, log256_max_n=None, n=None):
    """Sort ``keys`` in place with an LSB radix sort (the friendly host entry).

    Validates inputs, derives the compile-time params (pass count / scan depth / twiddle) and the device-resident
    count, then launches the sort as a single capturable kernel chain (:func:`_radix_sort_kernel` ->
    :func:`radix_sort_func`). To compose the sort inside your own ``graph=True`` parent kernel, call
    :func:`radix_sort_func` directly (the qipc path).

    Args:
        keys: 1-D tensor of a supported dtype (``{u32, i32, f32, u64, i64, f64}``). Sorted in place; after an even
            pass count the result lands back in ``keys``.
        tmp_keys: distinct 1-D buffer of the same shape and dtype as ``keys`` (ping-pong scratch; garbage on return).
        scratch: caller-owned 1-D ``u32`` workspace of :func:`radix_sort_scratch_slots` ``(N[, log256_max_n])`` slots
            (tile histograms + scan partials; ``u32`` regardless of key width). A too-small buffer raises
            :class:`InsufficientScratchError`.
        values / tmp_values: optional parallel payload tensors (same shape as ``keys``); pass **both** to sort
            key-value pairs, or neither for keys-only. Permuted in lock-step with the keys.
        end_bit: low key bits to sort - a positive multiple of 8, ``<=`` the key width. Defaults to the full key width.
        log256_max_n: scan depth ``D`` (the emitted sort handles any count ``<= 256 ** D``). Defaults to the minimal
            depth for ``N`` (see ``n``). Pass an explicit ``D`` (sized against a provisioned upper bound) to keep a
            fixed launch topology across calls with varying counts; size ``scratch`` with the same ``D``.
        n: element count to sort. Defaults to ``keys.shape[0]`` (sort the whole buffer). Pass an explicit ``n`` to sort
            only the first ``n`` slots of oversized reusable buffers (the qipc ``padded_N`` idiom); the trailing slots
            are left untouched. Must satisfy ``0 <= n <= keys.shape[0]``.
    """
    _validate_radix_inputs(keys, tmp_keys, values, tmp_values, end_bit)
    capacity = keys.shape[0]
    N = capacity if n is None else int(n)
    if N < 0 or N > capacity:
        raise ValueError(f"radix_sort n={N} out of range for keys of length {capacity}")
    key_dtype = keys.dtype
    if log256_max_n is None:
        log256_max_n = _min_log256_for_n(N)
    if end_bit is None:
        end_bit = _key_width_bits(key_dtype)
    has_values = values is not None
    values_arg = values if has_values else keys
    tmp_values_arg = tmp_values if has_values else tmp_keys
    _validate_caller_scratch("radix_sort", N, scratch, radix_sort_scratch_slots(N, log256_max_n), u32)
    n_dev = ndarray(i32, shape=())
    n_dev.fill(N)
    _radix_sort_kernel(
        keys, tmp_keys, values_arg, tmp_values_arg, scratch, n_dev, key_dtype, has_values, end_bit, log256_max_n
    )


def _min_log256_for_n(n: int) -> int:
    """Smallest ``D >= 1`` such that ``256**D >= n`` - the minimal scan depth that keeps the base-case buffer
    ``<= BLOCK_DIM`` for a length-``n`` sort."""
    d = 1
    cap = RADIX_DIGITS
    while cap < n:
        cap *= RADIX_DIGITS
        d += 1
    return d


def radix_sort_scratch_slots(n, log256_max_n: int = None):
    """Minimum u32 scratch slots :func:`radix_sort` / :func:`radix_sort_func` need for a length-``n`` input.

    ``hist_len = ceil(n/BLOCK_DIM) * RADIX_DIGITS`` for the tile histograms, plus the scan-staircase partials.
    Dtype-independent (tile histograms are ``u32`` regardless of key width). Use it to size a caller-owned ``scratch``
    buffer up front::

        D = log256_max_n  # the same depth you pass to the sort
        scratch = qd.Tensor(qd.ndarray(qd.u32, shape=max(qd.algorithms.radix_sort_scratch_slots(N, D), 1)))

    ``log256_max_n`` is the compile-time scan depth ``D``. The staircase is forced to ``D - 1`` reduce levels (even
    when ``n`` would naturally bottom out sooner), so for small ``n`` at an over-specified ``D`` this can be a few slots
    larger than the natural recursion (each forced extra level adds 1 slot). Allocate **at least** this many (more is
    fine); size against the provisioned upper bound on the count (e.g. qipc's ``padded_N``), **not** ``256**D``.

    Two ways to call it:

    - **explicit depth** ``radix_sort_scratch_slots(n, D)`` - host- **and** kernel-callable: the body is pure
      ``ceil``/multiply/accumulate arithmetic and the ``D`` loop unrolls at compile time, so ``n`` may be a Python
      ``int`` (host) **or** a device-read ``Expr`` (kernel, e.g. to re-check the actual device-``N`` against
      ``scratch.shape[0]`` on-device). ``D`` must be a compile-time constant in either context.
    - **auto depth** ``radix_sort_scratch_slots(n)`` - host-only convenience: derives the minimal depth from ``n`` via
      :func:`_min_log256_for_n` (a data-dependent loop that cannot compile device-side).

    Returns the real footprint for every ``n >= 0`` (``n = 0`` -> 0; ``n = 1`` -> one tile histogram = ``RADIX_DIGITS``
    slots). Unlike the removed host entry there is no ``n <= 1`` early-out: the kernel always runs all phases, so a
    length-1 sort still needs its histogram slots. Multiply by 4 for the byte size.
    """
    if log256_max_n is None:
        log256_max_n = _min_log256_for_n(n)
    num_blocks = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    cursor = hist_len
    nn = hist_len
    for _ in range(log256_max_n - 1):
        B = (nn + (BLOCK_DIM - 1)) // BLOCK_DIM
        cursor = cursor + B  # ``+=`` would lower to atomic_add on a non-writable Expr in kernel scope
        nn = B
    return cursor


__all__ = [
    "InsufficientScratchError",
    "radix_sort",
    "radix_sort_func",
    "radix_sort_scratch_slots",
]
