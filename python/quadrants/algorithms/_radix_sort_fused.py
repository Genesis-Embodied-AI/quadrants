# type: ignore
"""Single-kernel (fused) device-wide LSB radix sort.

This is the ``qd.kernel``-fused sibling of ``_radix_sort.device_radix_sort``. Where the original launches one
``@qd.kernel`` per sub-pass from the host (histogram / scan-recursion / scatter, ``~28`` launches for a 4-pass u32
sort), this variant emits the *entire* sort as a **single** ``@qd.kernel`` whose body is a fixed sequence of top-level
``for`` loops - each of which Quadrants offloads as its own serialized GPU launch, giving the implicit grid-wide
synchronization the algorithm needs between phases.

Three properties make the fused form graph-friendly:

1. **Fixed launch topology.** The number and order of internal launches is a compile-time constant, fixed by the
   ``log256_max_n`` template (``D``) - *not* by the runtime ``N``. The scan over the tile histograms is normally
   host-recursive (depth ``= ceil(log256(N))``, so the launch sequence changes with ``N``); here it is statically
   unrolled to exactly ``D - 1`` reduce levels + 1 base scan + ``D - 1`` downsweep levels. Extra levels (when ``N`` is
   smaller than ``256**D``) operate on length-1 buffers and are harmless no-ops that still produce the correct scan
   (reduce of 1 -> 1, base scan of 1 -> identity, downsweep applies an identity prefix).
2. **Device-resident ``N`` + fixed (~core-count) grid.** ``N`` is read from a 0-d ``i32`` ndarray with ``N[()]``, not
   taken as a host int; ``num_blocks`` / ``hist_len`` are derived on-device. Because those sizes reach the per-phase
   loop bounds as ``Expr``s (dynamic, not compile-time constant), the CUDA codegen leaves each offload's grid at the
   *saturating* value (``num_SMs * max_blocks_per_SM * 2``) and the runtime grid-stride loop walks the actual range.
   So the grid is a fixed ~core-count regardless of ``N`` - no ``loop_config(grid_dim=...)`` hook and no hand-rolled
   grid-stride ``while`` (which would put shared-memory-allocating block primitives inside a loop). See
   ``perso_hugh/doc/qipc/qipc_sort_as_kernel.md`` §D4.
3. **Single host call.** One ``_fused_radix_sort_keys_u32(...)`` invocation replaces the host ping-pong loop.

What is *not* yet done here (tracked as follow-ups, see module TODO):

- **Keys-only ``u32`` only.** Values, ``u64`` keys, and the i32/f32/i64/f64 twiddle are TODO; the structure mirrors
  ``_radix_sort.py`` so they slot in the same way.

See ``_radix_sort.py`` for the algorithm exposition (histogram-scan-scatter, digit-major layout, tail-block
sentinels) - the per-phase bodies below are the same logic, lifted into ``@qd.func`` so they can be inlined into one
kernel.
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
from quadrants.types.primitive_types import i32, u32

from ._radix_sort import RADIX_BITS, RADIX_DIGITS, InsufficientScratchError
from ._reduce import BLOCK_DIM


# --- Per-phase device bodies (one tile per block; grid sized to the work) ---------------
#
# Each helper is a ``@qd.func`` whose body is a single top-level ``for`` loop. When inlined into a ``@qd.kernel`` the
# loop becomes its own offloaded GPU launch, so calling several of these in sequence from one kernel yields the
# serialized, grid-synchronized launch chain the radix sort needs. ``i`` is the global thread index
# (``i = block_id * BLOCK_DIM + tid``), exactly as in the standalone kernels in ``_radix_sort.py`` / ``_scan.py``.


@_func
def _fused_hist(keys: template(), scratch: template(), n: i32, num_blocks: i32, bit_start: i32):
    """Per-block histogram of digit ``(key >> bit_start) & 0xFF`` into ``scratch`` (digit-major: ``d*num_blocks+b``).

    Mirror of ``_radix_sort._radix_histogram_pass`` with ``histograms_off = 0`` and the u32-key path inlined. ``n`` /
    ``num_blocks`` arrive as device ``Expr``s (derived from the scalar-tensor ``N`` in the kernel), so ``total_threads``
    is a dynamic bound -> the offload runs on the saturating (~core-count) grid + runtime grid-stride.
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
            key = bit_cast(keys[i], u32)
            digit = i32((key >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
            atomic_add(hist[digit], i32(1))
        _block.sync()
        if tid < RADIX_DIGITS:
            scratch[tid * num_blocks + block_id] = bit_cast(hist[tid], u32)


@_func
def _fused_reduce(scratch: template(), in_off: i32, out_off: i32, n: i32, total_threads: i32):
    """Tile-reduce ``scratch[in_off : in_off+n]`` -> per-tile sums ``scratch[out_off : out_off+ceil(n/BLOCK_DIM)]``.

    u32 / add specialization of ``_reduce._reduce_pass`` (counts are u32, monoid is +). One tile per block; out-of-range
    lanes contribute the additive identity ``0``.
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
    keys_in: template(), keys_out: template(), scratch: template(), n: i32, num_blocks: i32, bit_start: i32
):
    """Per-block radix rank + scatter ``keys_in[i] -> keys_out[scanned_offset + intra_digit_rank]``.

    Mirror of ``_radix_sort._radix_scatter_pass`` (u32 keys, no values) with ``histograms_off = 0``. ``n`` /
    ``num_blocks`` are device ``Expr``s (see ``_fused_hist``).
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
        if i < n:
            key = bit_cast(keys_in[i], u32)
        rank = _block.radix_rank_match_atomic_or(key, BLOCK_DIM, RADIX_BITS, bit_start, RADIX_BITS, bins, excl_prefix)
        digit = i32((key >> u32(bit_start)) & u32(RADIX_DIGITS - 1))
        if tid < RADIX_DIGITS:
            global_off = bit_cast(scratch[tid * num_blocks + block_id], i32)
            block_offsets[tid] = global_off - excl_prefix[tid]
        _block.sync()
        if i < n:
            dst = block_offsets[digit] + rank
            keys_out[dst] = bit_cast(key, u32)


def _emit_scan_staircase(scratch, off, n, levels_remaining: int):
    """Emit a *fixed-depth* in-place exclusive scan of ``scratch[off : off+n]`` (digit-major histograms).

    Plain Python helper run at kernel-trace time: it makes the ``@qd.func`` calls that become offloaded launches.
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


@kernel
def _fused_radix_sort_keys_u32(
    keys: template(),
    tmp_keys: template(),
    scratch: template(),
    N: ndarray_ann(dtype=i32, ndim=0),
    NUM_PASSES: template(),
    LOG256_MAX_N: template(),
):
    """Whole keys-only u32 LSB radix sort as one kernel.

    ``N`` is a **0-d ``i32`` ndarray** holding the element count in device memory (the "golden source" of ``N`` in a
    qipc-style solver is a device atomic counter, not a host int). An ndarray (not a field) so it reads torch-style with
    ``N[()]`` and can be passed alongside the field ``template()`` buffers. It is read once and all
    derived sizes (``num_blocks``, ``hist_len``) are computed **on-device** - no host-precomputed launch params. Because
    those sizes flow into the per-phase loop bounds as ``Expr``s, every offload runs on the saturating (~core-count)
    grid + runtime grid-stride, so the launch topology is fixed by ``LOG256_MAX_N`` / ``NUM_PASSES`` alone (graph-safe).

    ``NUM_PASSES`` digit passes (4 for full 32-bit) ping-pong ``keys <-> tmp_keys``; each pass is histogram ->
    ``LOG256_MAX_N``-deep static scan staircase -> scatter. After an even ``NUM_PASSES`` the result lands in ``keys``.
    """
    n = N[()]
    num_blocks = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    hist_len = num_blocks * RADIX_DIGITS
    for p in static(range(NUM_PASSES)):
        bit_start = p * RADIX_BITS
        # Static ping-pong: choose src/dst at compile time (``p`` is a static index) by branching the field-argument
        # helper calls. Aliasing a field to a local (``src = keys``) is traced as a runtime assign and rejected, so we
        # pass ``keys`` / ``tmp_keys`` straight into the calls in each ``static`` branch instead.
        if static(p % 2 == 0):
            _fused_hist(keys, scratch, n, num_blocks, bit_start)
        else:
            _fused_hist(tmp_keys, scratch, n, num_blocks, bit_start)
        _emit_scan_staircase(scratch, 0, hist_len, LOG256_MAX_N - 1)
        if static(p % 2 == 0):
            _fused_scatter(keys, tmp_keys, scratch, n, num_blocks, bit_start)
        else:
            _fused_scatter(tmp_keys, keys, scratch, n, num_blocks, bit_start)


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
    this can be a few slots larger than ``_radix_sort.device_radix_sort_scratch_slots(n)`` (each forced extra level
    adds 1 slot). The caller must allocate **at least** this many slots (more is fine); size against the provisioned
    upper bound on the element count (e.g. qipc's ``padded_N``), **not** ``256**log256_max_n``.

    Host- **and** kernel-callable: the body is pure ``ceil``/multiply/accumulate arithmetic and the
    ``log256_max_n`` loop unrolls at trace time, so ``n`` may be a Python ``int`` (host) **or** a device-read
    Quadrants ``Expr`` (kernel). The kernel path lets the sort re-check the actual device-``N`` against
    ``scratch.shape[0]`` on-device and raise an overflow flag (yield-and-realloc), rather than relying on the host
    knowing ``N``. ``log256_max_n`` must be a compile-time constant in either context.

    No ``n <= 1`` special-case: the formula already yields a valid lower bound for any ``n >= 0`` (``n = 0`` -> 0),
    which keeps it branch-free for the device path.
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


def device_radix_sort_fused(keys, tmp_keys, scratch, n=None, *, log256_max_n: int = None, end_bit: int = None):
    """Fused single-kernel keys-only ``u32`` ascending radix sort (M1; see module docstring for scope).

    The graph-capturable unit is the kernel :func:`_fused_radix_sort_keys_u32`, which reads its element count from a
    **0-d ``i32`` ndarray** in device memory. This eager wrapper just supplies that scalar: pass ``n`` as a 0-d
    ndarray to use a device-resident count directly (the qipc path), or omit it and the count is taken from
    ``keys.shape[0]`` and materialized into a fresh 0-d ndarray.

    Args:
        keys: 1-D ``u32`` tensor, sorted in place.
        tmp_keys: 1-D ``u32`` ping-pong workspace, distinct from ``keys``, same shape.
        scratch: 1-D ``u32`` workspace sized via :func:`fused_radix_sort_scratch_slots`. Raises
            :class:`InsufficientScratchError` if too small.
        n: optional 0-d ``i32`` ndarray (``shape=()``) holding the element count on-device. If ``None``, the count is
            ``keys.shape[0]`` and a 0-d ndarray is allocated and filled with it.
        log256_max_n: compile-time scan depth ``D``. The kernel handles any element count ``<= 256**D``; a captured
            graph for a given ``D`` is reusable across all such counts. Defaults to the minimal depth for
            ``keys.shape[0]`` (``ceil(log256 N)``).
        end_bit: number of low key bits to sort (multiple of ``RADIX_BITS``, even number of passes). Defaults to 32.
    """
    import quadrants as qd

    if keys.dtype != u32 or tmp_keys.dtype != u32:
        raise NotImplementedError("device_radix_sort_fused (M1) supports u32 keys only; use device_radix_sort for others")
    if scratch.dtype != u32:
        raise TypeError(f"device_radix_sort_fused scratch must be u32; got {scratch.dtype}")
    if keys is tmp_keys:
        raise ValueError("device_radix_sort_fused requires keys and tmp_keys to be distinct buffers")
    # Host-side capacity bound: how many elements the buffers can hold. The actual sorted count comes from ``N[()]`` on
    # device (``n`` when provided, else this value materialized below).
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

    if end_bit is None:
        end_bit = 32
    if end_bit <= 0 or end_bit % RADIX_BITS != 0 or (end_bit // RADIX_BITS) % 2 != 0:
        raise ValueError(
            f"device_radix_sort_fused end_bit must be a positive even multiple of {RADIX_BITS}; got {end_bit}"
        )
    num_passes = end_bit // RADIX_BITS

    # Scratch is sized to the provisioned capacity (the host-known buffer length). The kernel-callable
    # ``fused_radix_sort_scratch_slots`` re-checks the *actual* device count on-device for the graph path (D6).
    needed = fused_radix_sort_scratch_slots(capacity_n, log256_max_n)
    cap = scratch.shape[0]
    if needed > cap:
        raise InsufficientScratchError(capacity_n, needed, cap)

    if n is None:
        # 0-d i32 ndarray matching the kernel's ``N`` annotation (read torch-style as ``N[()]``). The device-N graph
        # path supplies its own scalar ndarray.
        n = qd.ndarray(i32, shape=())
        n.fill(capacity_n)

    _fused_radix_sort_keys_u32(keys, tmp_keys, scratch, n, num_passes, log256_max_n)


__all__ = [
    "device_radix_sort_fused",
    "fused_radix_sort_scratch_slots",
]
