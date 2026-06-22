# type: ignore
"""Device-wide LSB radix sort (one capturable launch chain).

The sort is exposed as the graph-composable ``@qd.func`` :func:`sort`: call it at the **top level** of your own
``@qd.kernel`` to compose the sort inline with other phases in one compiled kernel / captured graph (the qipc path); it
takes the count as a device-resident 0-d ``n`` and the params as templates, and does no host-side validation. Size the
caller-owned ``u32`` scratch with :func:`sort_scratch_slots`.

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
2. **Device-resident ``n`` + fixed (~core-count) grid.** ``n`` is read from a 0-d ``i32`` ndarray with ``n[()]``, not
   taken as a host int; ``num_blocks`` / ``hist_len`` are derived on-device. Because those sizes reach the per-phase
   loop bounds as ``Expr``s (dynamic, not compile-time constant), the CUDA codegen leaves each offload's grid at the
   *saturating* value (``num_SMs * max_blocks_per_SM * 2``) and the runtime grid-stride loop walks the actual range -
   a fixed ~core-count grid regardless of ``n``. See ``perso_hugh/doc/qipc/qipc_sort_as_kernel.md`` §D4.
3. **Single call.** One :func:`sort` invocation replaces the host ping-pong loop.

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

**Scratch.** The sort needs a **caller-owned** 1-D ``u32`` ``scratch`` buffer of :func:`sort_scratch_slots`
``(N, log256_max_n)`` slots (tile histograms + scan partials; ``u32`` regardless of key width, so 8-byte-key sorts
have the same footprint as 4-byte ones). There is **no** module-level shared-scratch fallback - the caller always
owns the buffer (graph- / multi-stream-safe, no global state). :func:`sort` does **no** on-device scratch check
(a DtoH would defeat graph capture), so size ``scratch`` correctly up front when composing the func.
"""

from quadrants.lang.impl import static
from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.lang.simt import block as _block
from quadrants.types.annotations import template
from quadrants.types.primitive_types import f32, f64, i32, i64, u32, u64

from ._reduce import BLOCK_DIM, _at_least_one, _validate_log256_max_n
from ._scan import _emit_exclusive_scan_add

RADIX_BITS = 8
"""Bits per digit. Matches the ``block.radix_rank_match_atomic_or`` constraint that ``block_dim == 1 << radix_bits``;
with ``BLOCK_DIM = 256`` this is the only legal value."""

RADIX_DIGITS = 1 << RADIX_BITS  # 256

_SUPPORTED_KEY_DTYPES_32 = (u32, i32, f32)
_SUPPORTED_KEY_DTYPES_64 = (u64, i64, f64)
_TWIDDLE_KEY_DTYPES = (i32, f32, i64, f64)


def _key_width_bits(dtype) -> int:
    if dtype in _SUPPORTED_KEY_DTYPES_32:
        return 32
    if dtype in _SUPPORTED_KEY_DTYPES_64:
        return 64
    raise NotImplementedError(f"sort key dtype {dtype} not supported")


# --- Per-phase device bodies (one tile per block; grid sized to the work) ---------------
#
# Each helper is a ``@qd.func`` whose body is a single top-level ``for`` loop. When inlined into a ``@qd.kernel`` the
# loop becomes its own offloaded GPU launch, so calling several of these in sequence from one kernel yields the
# serialized, grid-synchronized launch chain the radix sort needs. ``i`` is the global thread index
# (``i = block_id * BLOCK_DIM + tid``). ``n`` / ``num_blocks`` arrive as device ``Expr``s (derived from the scalar
# ``count`` in the kernel), so the loop bounds are dynamic -> each offload runs on the saturating (~core-count) grid.


@_func
def _radix_twiddle(keys: template(), n: i32, key_dtype: template(), key_width: template(), do_twiddle: template()):
    """In-place map between caller-dtype keys and monotone-unsigned "sortable" keys (signed / float only).

    ``do_twiddle=True`` maps dtype -> sort order before the first pass; ``False`` is the inverse after the last pass.
    Only instantiated for ``i32`` / ``f32`` / ``i64`` / ``f64`` (the kernel skips it for unsigned keys).
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(n):
        if static(key_width == 32):
            v = bit_cast(keys[i], u32)
            if static(key_dtype == i32):
                keys[i] = bit_cast(v ^ u32(0x80000000), key_dtype)
            else:  # f32
                # Forward (do_twiddle) picks the mask from the *input* sign bit; the inverse picks it from the
                # *output* sign bit, which is the *opposite* bit (the forward twiddle flips it) - hence the two
                # branches select the 0x80000000 / 0xFFFFFFFF masks in swapped order.
                if static(do_twiddle):
                    if (v & u32(0x80000000)) != u32(0):
                        keys[i] = bit_cast(v ^ u32(0xFFFFFFFF), key_dtype)
                    else:
                        keys[i] = bit_cast(v ^ u32(0x80000000), key_dtype)
                else:
                    if (v & u32(0x80000000)) != u32(0):
                        keys[i] = bit_cast(v ^ u32(0x80000000), key_dtype)
                    else:
                        keys[i] = bit_cast(v ^ u32(0xFFFFFFFF), key_dtype)
        else:  # 64-bit
            w = bit_cast(keys[i], u64)
            if static(key_dtype == i64):
                keys[i] = bit_cast(w ^ u64(0x8000000000000000), key_dtype)
            else:  # f64
                # Same input- vs output-sign-bit asymmetry as the 32-bit f32 case above (forward picks from the
                # input sign bit, inverse from the output sign bit), so the masks are selected in swapped order.
                if static(do_twiddle):
                    if (w & u64(0x8000000000000000)) != u64(0):
                        keys[i] = bit_cast(w ^ u64(0xFFFFFFFFFFFFFFFF), key_dtype)
                    else:
                        keys[i] = bit_cast(w ^ u64(0x8000000000000000), key_dtype)
                else:
                    if (w & u64(0x8000000000000000)) != u64(0):
                        keys[i] = bit_cast(w ^ u64(0x8000000000000000), key_dtype)
                    else:
                        keys[i] = bit_cast(w ^ u64(0xFFFFFFFFFFFFFFFF), key_dtype)


@_func
def _radix_hist(keys: template(), scratch: template(), n: i32, num_blocks: i32, bit_start: i32, key_width: template()):
    """Per-block histogram of digit ``(key >> bit_start) & 0xFF`` into ``scratch`` (digit-major: ``d*num_blocks+b``).

    Tile histograms are ``u32`` regardless of key width (each count <= ``BLOCK_DIM`` = 256). ``key_width`` selects the
    32- vs 64-bit digit extraction.
    """
    loop_config(block_dim=BLOCK_DIM)
    total_threads = num_blocks * BLOCK_DIM
    for i in range(total_threads):
        tid = i % BLOCK_DIM
        block_id = i // BLOCK_DIM
        # One extra "dump" slot (index RADIX_DIGITS) absorbs the increments of out-of-range lanes so the shared
        # atomic_add below runs in *uniform* control flow. A guarded `if i < n: atomic_add(...)` would put the atomic
        # - and the acquire-release memory barrier the codegen emits in front of every native atomic - inside
        # thread-divergent control flow; on Metal / MoltenVK spirv-cross lowers that barrier to a full
        # threadgroup_barrier, which is undefined behaviour when only some lanes of the tail block reach it and
        # silently drops histogram counts. Keeping the atomic unconditional sidesteps that entirely.
        hist = _block.SharedArray((RADIX_DIGITS + 1,), i32)
        hist[tid] = i32(0)
        if tid == 0:
            hist[RADIX_DIGITS] = i32(0)
        _block.sync()
        digit = i32(RADIX_DIGITS)  # default to the dump slot for out-of-range lanes (unconditional first assignment)
        if i < n:
            if static(key_width == 32):
                key32 = bit_cast(keys[i], u32)
                digit = i32((key32 >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
            else:
                key64 = bit_cast(keys[i], u64)
                digit = i32((key64 >> u64(bit_start)) & u64(RADIX_DIGITS - 1))
        atomic_add(hist[digit], i32(1))
        _block.sync()
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
    key_dtype: template(),
    has_values: template(),
    key_width: template(),
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
        if static(key_width == 32):
            key = u32(0xFFFFFFFF)
            if i < n:
                key = bit_cast(keys_in[i], u32)
            rank = _block.radix_rank_match_atomic_or(
                key, BLOCK_DIM, RADIX_BITS, bit_start, RADIX_BITS, bins, excl_prefix
            )
            digit = i32((key >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
            if tid < RADIX_DIGITS:
                global_off = bit_cast(scratch[tid * num_blocks + block_id], i32)
                # Subtract the block-local exclusive prefix: rebases rank from "position among all keys in this
                # block" to "position among only this digit's keys in this block" (the intra-digit base offset).
                block_offsets[tid] = global_off - excl_prefix[tid]
            _block.sync()
            if i < n:
                dst = block_offsets[digit] + rank
                keys_out[dst] = bit_cast(key, key_dtype)
                if static(has_values):
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
                keys_out[dst] = bit_cast(key, key_dtype)
                if static(has_values):
                    values_out[dst] = values_in[i]


def _emit_pass(
    keys,
    tmp_keys,
    values,
    tmp_values,
    scratch,
    n,
    num_blocks,
    hist_len,
    p,
    key_dtype,
    has_values,
    key_width,
    log256_max_n,
):
    """Emit one digit pass (histogram -> scan staircase -> scatter) at compile time.

    Pure-Python (runs during kernel compilation): ``p`` is a Python int from the static pass loop, so the src/dst
    (and value) ping-pong is selected here in plain Python and the field handles are passed straight into the
    ``@qd.func`` calls - no field-handle assignment in the compiled body (which the compiler frontend rejects).

    The histogram scan reuses the shared graph-composable staircase :func:`._scan._emit_exclusive_scan_add` (``u32`` /
    add, in place, fixed depth). ``log256_max_n - 1`` reduce levels makes the digit-major ``scratch[0:hist_len]`` scan a
    compile-time-constant launch topology, independent of the device-resident ``n``.
    """
    bit_start = p * RADIX_BITS
    src = keys if (p % 2 == 0) else tmp_keys
    dst = tmp_keys if (p % 2 == 0) else keys
    vsrc = values if (p % 2 == 0) else tmp_values
    vdst = tmp_values if (p % 2 == 0) else values
    _radix_hist(src, scratch, n, num_blocks, bit_start, key_width)
    _emit_exclusive_scan_add(scratch, 0, hist_len, log256_max_n - 1)
    _radix_scatter(src, dst, vsrc, vdst, scratch, n, num_blocks, bit_start, key_dtype, has_values, key_width)


@_func(requires_top_level=True)
def sort(
    keys: template(),
    tmp_keys: template(),
    values: template(),
    tmp_values: template(),
    scratch: template(),
    n: template(),
    key_dtype: template(),
    has_values: template(),
    end_bit: template(),
    log256_max_n: template(),
):
    """Whole LSB radix sort as one ``@qd.func`` - the composable form; see module docstring.

    **Experimental** - this API is new and may change in a future release.

    Call it at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent that chains it with other
    phases). Each phase helper's single top-level ``for`` stays its own offloaded GPU launch, so the inter-phase
    grid-wide synchronization survives and every phase is captured as a node in the parent's graph. A
    ``while qd.graph_do_while(...):`` body **counts as top level** - the loops directly inside it still lower as
    separate offloaded launches with grid-wide barriers between them, so calling this func directly in a
    ``graph_do_while`` body is supported and re-sorts correctly every iteration (verified for ``n`` spanning many
    blocks). What you **must not** do is nest the call inside *ordinary* runtime control flow - another ``for``, an
    ``if``, or a plain ``while`` - which demotes the phase loops out of top-level position, collapses the per-phase
    grid-wide barriers and corrupts the sort. Compile-time ``static`` loops (like the pass loop here) are also fine.

    Compile-time params: ``key_dtype`` (the key element dtype, one of ``{u32, i32, f32, u64, i64, f64}``),
    ``has_values`` (whether ``values`` / ``tmp_values`` are real buffers or placeholders), ``end_bit`` (low key bits to
    sort - positive even multiple of ``8``, ``<=`` key width), and ``log256_max_n`` (scan depth ``D``; the emitted sort
    handles any count ``<= 256 ** D``). The width, pass count and twiddle need are derived from ``key_dtype`` +
    ``end_bit`` at compile time. ``key_dtype`` is an explicit param because an ``ndarray`` kernel argument (the qipc
    path) exposes no ``.dtype`` inside the kernel - pass the dtype you already know.

    ``n`` is a 0-d ``i32`` ndarray handle read once as ``n[()]``; ``num_blocks`` / ``hist_len`` are derived on-device.
    The pass loop and scan staircase are statically unrolled, so the launch topology is fixed regardless of ``n``;
    after an even pass count the result lands in ``keys`` (and ``values``). The caller owns ``scratch`` (size it with
    :func:`sort_scratch_slots` ``(capacity_n, log256_max_n)``) and the device-resident ``n``. There is **no**
    host-side validation or scratch-sufficiency check (a DtoH would defeat graph capture) - pass distinct, same-shape
    buffers and size ``scratch`` correctly up front.
    """
    _validate_log256_max_n(log256_max_n)
    key_width = static(_key_width_bits(key_dtype))
    num_passes = static(end_bit // RADIX_BITS)
    needs_twiddle = static(key_dtype in _TWIDDLE_KEY_DTYPES)
    count = n[()]
    num_blocks = (count + (BLOCK_DIM - 1)) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    if static(needs_twiddle):
        _radix_twiddle(keys, count, key_dtype, key_width, True)
    for p in static(range(num_passes)):
        _emit_pass(
            keys,
            tmp_keys,
            values,
            tmp_values,
            scratch,
            count,
            num_blocks,
            hist_len,
            p,
            key_dtype,
            has_values,
            key_width,
            log256_max_n,
        )
    if static(needs_twiddle):
        _radix_twiddle(keys, count, key_dtype, key_width, False)


def _min_log256_for_n(n: int) -> int:
    """Smallest ``D >= 1`` such that ``256**D >= n`` - the minimal scan depth that keeps the base-case buffer
    ``<= BLOCK_DIM`` for a length-``n`` sort."""
    d = 1
    cap = RADIX_DIGITS
    while cap < n:
        cap *= RADIX_DIGITS
        d += 1
    return d


def sort_scratch_slots(n, log256_max_n: int = None):
    """Minimum u32 scratch slots :func:`sort` needs for a length-``n`` input.

    ``hist_len = ceil(n/BLOCK_DIM) * RADIX_DIGITS`` for the tile histograms, plus the scan-staircase partials.
    Dtype-independent (tile histograms are ``u32`` regardless of key width). Use it to size a caller-owned ``scratch``
    buffer up front::

        D = log256_max_n  # the same depth you pass to the sort
        scratch = qd.Tensor(qd.ndarray(qd.u32, shape=qd.algorithms.sort_scratch_slots(N, D)))

    ``log256_max_n`` is the compile-time scan depth ``D``. The staircase is forced to ``D - 1`` reduce levels (even
    when ``n`` would naturally bottom out sooner), so for small ``n`` at an over-specified ``D`` this can be a few slots
    larger than the natural recursion (each forced extra level adds 1 slot). Allocate **at least** this many (more is
    fine); size against the provisioned upper bound on the count (e.g. qipc's ``padded_N``), **not** ``256**D``.

    Two ways to call it:

    - **explicit depth** ``sort_scratch_slots(n, D)`` - host- **and** kernel-callable: the body is pure
      ``ceil``/multiply/accumulate arithmetic and the ``D`` loop unrolls at compile time, so ``n`` may be a Python
      ``int`` (host) **or** a device-read ``Expr`` (kernel, e.g. to re-check the actual device-``n`` against
      ``scratch.shape[0]`` on-device). ``D`` must be a compile-time constant in either context.
    - **auto depth** ``sort_scratch_slots(n)`` - host-only convenience: derives the minimal depth from ``n`` via
      :func:`_min_log256_for_n` (a data-dependent loop that cannot compile device-side).

    Returns the real footprint for every ``n >= 1`` (``n = 1`` -> one tile histogram = ``RADIX_DIGITS`` slots). There
    is no ``n <= 1`` early-out: the kernel always runs all phases, so a length-1 sort still needs its histogram slots.
    The only case with no real scratch is ``n = 0``, which returns ``1`` (the lone slot is never touched) so the result
    can size an allocation directly. Multiply by 4 for the byte size.
    """
    if log256_max_n is None:
        log256_max_n = _min_log256_for_n(n)
    _validate_log256_max_n(log256_max_n)
    num_blocks = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    cursor = hist_len
    nn = hist_len
    for _ in range(log256_max_n - 1):
        B = (nn + (BLOCK_DIM - 1)) // BLOCK_DIM
        cursor = cursor + B  # ``+=`` would lower to atomic_add on a non-writable Expr in kernel scope
        nn = B
    return _at_least_one(cursor)


__all__ = [
    "sort",
    "sort_scratch_slots",
]
