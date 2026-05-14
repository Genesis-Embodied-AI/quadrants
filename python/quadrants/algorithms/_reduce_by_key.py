# type: ignore
"""Device-wide reduce-by-key.

Implements ``qd.algorithms.device_reduce_by_key_add`` on top of the existing
device exclusive scan internals and the shared ``Field(u32)`` scratch.

Reduce-by-key takes two parallel 1-D tensors - ``keys`` and ``values`` - and collapses every **consecutive run of
equal keys** into a single output entry ``(unique_key, sum_of_values_in_run)``. Keys that are equal but separated by
other keys are treated as separate runs. To compute a global per-key sum, sort by key first (e.g. via
``qd.algorithms.device_radix_sort``) and then reduce-by-key.

Algorithm (scan + scatter; no segmented-scan primitive needed):

1. **Head-flag pass** (``_rbk_head_flags``). Compute ``head_flags[i] = 1`` if ``i == 0 or keys[i] != keys[i-1]``, else
   ``0``, directly into the shared ``Field(u32)`` scratch ``scratch[0:N]`` (storing the ``i32`` flag bit-cast to
   ``u32``).
2. **Exclusive scan of head_flags** (in-place over ``scratch[0:N]``, using ``_reduce_pass`` +
   ``_exclusive_scan_inplace_u32`` + ``_scan_pass3`` reused from ``_reduce.py`` / ``_scan.py``). After this,
   ``scratch[i] = exclusive_scan(head_flags)[i] = sum(head_flags[0:i])``. The 0-indexed run index of element ``i`` is
   then ``positions[i] = scratch[i] + head_flag(i) - 1`` (i.e. ``inclusive_scan(head_flags)[i] - 1``); the scatter
   pass recomputes ``head_flag(i)`` from the two keys at ``i`` and ``i - 1`` so the ``head_flags`` array itself does
   not need to survive the scan. This lets the scan run in place, holding scratch to ~``1.004 * N`` slots.
3. **Zero-init values_out**. The scatter step uses ``atomic_add`` on ``values_out[positions[i]]``; the slots must
   start at the additive identity ``0``.
4. **Scatter pass** (``_rbk_scatter``). For every ``i``:
   - Recompute ``head_flag(i)`` from ``i == 0 or keys[i] != keys[i-1]`` and compute the run index
     ``pos = scratch[i] + head_flag(i) - 1``.
   - ``keys_out[pos] = keys[i]`` - race-free because every thread in a run writes the same key to the same slot.
   - ``atomic_add(values_out[pos], values[i])`` folds the run's values into the run's output slot.
5. **Count pass** (``_rbk_count``). Computes ``num_runs[0] = scratch[N-1] + head_flag(N-1)`` where the head flag at
   ``N-1`` is recomputed from ``keys[N-1] != keys[N-2]`` for ``N >= 2`` (``1`` for ``N == 1``).

This first-land scope supports only the ``add`` reduction. ``min`` / ``max`` variants would need ``atomic_min`` /
``atomic_max``, which have spottier cross-backend support for ``f32`` - defer to a follow-up gated on real qipc usage.

Scratch budget: ``N + ceil(N / 256) + ...`` ``u32`` slots, ≈ ``1.004 * N``. The default 1 MB scratch covers ``N`` up
to ~260_000. For larger ``N``, raise via ``quadrants._scratch.set_scratch_bytes(...)`` before any algorithm call.
"""

from quadrants._scratch import get_scratch_u32, scratch_capacity_u32
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.lang.simt.reductions import _bin_add
from quadrants.types.annotations import template
from quadrants.types.primitive_types import f32, i32, u32

from ._reduce import BLOCK_DIM, _identity_bits, _reduce_pass
from ._scan import _exclusive_scan_inplace_u32, _scan_pass3

_SUPPORTED_KEY_DTYPES = (u32, i32, f32)
_SUPPORTED_VALUE_DTYPES = (u32, i32, f32)


@kernel
def _rbk_head_flags(keys_in: template(), head_flags: template(), head_flags_off: i32, N: i32):
    """Write ``head_flags[i] = 1 if (i == 0 or keys[i] != keys[i-1]) else 0``
    to ``head_flags[head_flags_off + i]`` (as the u32 bit pattern of i32).

    Linear-time, embarrassingly parallel: each thread reads at most two key elements (``keys[i]`` and ``keys[i-1]``)
    and writes one flag. The boundary thread at ``i == 0`` always writes ``1`` since there is no predecessor and a run
    trivially starts there.
    """
    loop_config(block_dim=BLOCK_DIM)
    for i in range(N):
        flag = i32(0)
        if i == 0:
            flag = i32(1)
        else:
            if keys_in[i] != keys_in[i - 1]:
                flag = i32(1)
        head_flags[head_flags_off + i] = bit_cast(flag, u32)


@kernel
def _rbk_zero_values_out(values_out: template(), N: i32, dtype: template()):
    """Set ``values_out[0 : N] = 0`` so the scatter ``atomic_add`` lands onto a clean additive identity. ``N`` is the
    upper bound on ``num_runs``; the caller-supplied ``values_out`` may be longer but we only need the prefix that the
    scatter can touch.

    We write ``bit_cast(u32(0), dtype)`` rather than relying on ``v - v == 0`` because the latter compiles to a real
    subtract for ``f32`` (and yields NaN if the slot held NaN garbage from a prior allocation), whereas the bit-cast
    lowers to a plain store.
    """
    for i in range(N):
        values_out[i] = bit_cast(u32(0), dtype)


@kernel
def _rbk_scatter(
    keys_in: template(),
    values_in: template(),
    positions: template(),
    positions_off: i32,
    keys_out: template(),
    values_out: template(),
    N: i32,
):
    """Per-element scatter:

    - Compute ``head_flag(i)`` on the fly from ``i == 0 or keys[i] != keys[i-1]`` and combine with the in-place
      exclusive scan stored in ``positions`` to recover the inclusive run index
      ``pos = positions[i] + head_flag(i) - 1``.
    - ``keys_out[pos] = keys_in[i]`` - race-free because every thread in a run writes the same key to the same slot.
    - ``atomic_add(values_out[pos], values_in[i])`` - folds the run's values into the run's output slot.
      ``values_out`` must be pre-zeroed (see ``_rbk_zero_values_out``).
    """
    for i in range(N):
        head_i = i32(0)
        if i == 0:
            head_i = i32(1)
        else:
            if keys_in[i] != keys_in[i - 1]:
                head_i = i32(1)
        pos = bit_cast(positions[positions_off + i], i32) + head_i - i32(1)
        keys_out[pos] = keys_in[i]
        atomic_add(values_out[pos], values_in[i])


@kernel
def _rbk_count(keys_in: template(), positions: template(), positions_off: i32, N: i32, num_runs: template()):
    """One-thread tail kernel: write ``num_runs[0] = total head_flag count``.

    Equivalently: ``num_runs = exclusive_scan_at(N-1) + head_flag(N-1) = inclusive_scan_at(N-1) =
    total_head_flags``. We can't read ``scratch[N-1]`` for the original head flag (the in-place scan overwrote it
    with the exclusive prefix), so we recompute the flag from the last two keys. For ``N == 1``,
    ``head_flag(0) == 1`` so ``num_runs = 0 + 1 = 1``.
    """
    for _ in range(1):
        pos_last = bit_cast(positions[positions_off + N - 1], i32)
        head_last = i32(0)
        if N == 1:
            head_last = i32(1)
        else:
            if keys_in[N - 1] != keys_in[N - 2]:
                head_last = i32(1)
        num_runs[0] = pos_last + head_last


def _validate_inputs(keys_in, values_in, keys_out, values_out, num_runs):
    if not hasattr(keys_in, "shape") or len(keys_in.shape) != 1:
        raise TypeError(f"device_reduce_by_key_add expects 1-D keys_in; got shape {getattr(keys_in, 'shape', None)}")
    if not hasattr(values_in, "shape") or values_in.shape != keys_in.shape:
        raise TypeError(
            f"device_reduce_by_key_add expects values_in.shape == keys_in.shape; got "
            f"keys_in={keys_in.shape}, values_in={values_in.shape}"
        )
    if not hasattr(keys_out, "shape") or len(keys_out.shape) != 1:
        raise TypeError(f"device_reduce_by_key_add expects 1-D keys_out; got shape {getattr(keys_out, 'shape', None)}")
    if keys_out.dtype != keys_in.dtype:
        raise TypeError(f"device_reduce_by_key_add dtype mismatch: keys_in={keys_in.dtype}, keys_out={keys_out.dtype}")
    if not hasattr(values_out, "shape") or len(values_out.shape) != 1:
        raise TypeError(
            f"device_reduce_by_key_add expects 1-D values_out; got shape {getattr(values_out, 'shape', None)}"
        )
    if values_out.dtype != values_in.dtype:
        raise TypeError(
            f"device_reduce_by_key_add dtype mismatch: values_in={values_in.dtype}, values_out={values_out.dtype}"
        )
    if keys_out.shape[0] < keys_in.shape[0]:
        raise ValueError(
            f"device_reduce_by_key_add keys_out.shape[0]={keys_out.shape[0]} < keys_in.shape[0]={keys_in.shape[0]}; "
            f"keys_out must hold at least N entries (worst case: every key is unique)"
        )
    if values_out.shape[0] < values_in.shape[0]:
        raise ValueError(
            f"device_reduce_by_key_add values_out.shape[0]={values_out.shape[0]} < values_in.shape[0]={values_in.shape[0]}; "
            f"values_out must hold at least N entries"
        )
    if not hasattr(num_runs, "shape") or num_runs.shape != (1,):
        raise TypeError(f"device_reduce_by_key_add expects num_runs.shape == (1,); got {num_runs.shape}")
    if num_runs.dtype != i32:
        raise TypeError(f"device_reduce_by_key_add expects num_runs.dtype == qd.i32; got {num_runs.dtype}")
    if keys_in.dtype not in _SUPPORTED_KEY_DTYPES:
        raise NotImplementedError(
            f"device_reduce_by_key_add keys dtype {keys_in.dtype} not in first-land set "
            f"{[d for d in _SUPPORTED_KEY_DTYPES]}; see design doc dtype matrix"
        )
    if values_in.dtype not in _SUPPORTED_VALUE_DTYPES:
        raise NotImplementedError(
            f"device_reduce_by_key_add values dtype {values_in.dtype} not in first-land set "
            f"{[d for d in _SUPPORTED_VALUE_DTYPES]}; see design doc dtype matrix"
        )


def device_reduce_by_key_add(keys_in, values_in, keys_out, values_out, num_runs):
    """Collapse every consecutive run of equal ``keys_in`` into ``(key, sum_of_values)``.

    Args:
        keys_in: 1-D tensor of ``u32`` / ``i32`` / ``f32``. Sort by key beforehand (e.g. via
            ``qd.algorithms.device_radix_sort``) if you need a global per-key sum rather than a per-run sum.
        values_in: 1-D tensor of ``u32`` / ``i32`` / ``f32``, same shape as ``keys_in``.
        keys_out: 1-D tensor of the same dtype as ``keys_in``, capacity ``>= N``. Receives the unique-run keys at
            indices ``[0 : num_runs[0])``; the tail is left untouched.
        values_out: 1-D tensor of the same dtype as ``values_in``, capacity ``>= N``. Receives the per-run sums. The
            first ``num_runs[0]`` slots are overwritten; if ``values_out`` was longer, the tail past that prefix is
            left untouched.
        num_runs: 1-element ``i32`` tensor receiving the number of runs.

    Same async / no-implicit-sync contract as the rest of ``qd.algorithms.*``: ``num_runs`` is a tensor (not a Python
    int); fetch the count with ``int(num_runs.to_numpy()[0])`` after the call.

    **NaN handling for f32 keys**: NaN ``!=`` NaN is true, so each NaN becomes its own run. This is consistent with
    treating NaN as "different from everything", which matches the run-length-encoding spirit of reduce-by-key.

    **Scratch budget**: ~``1.004 * N`` u32 slots. Default 1 MB covers ``N`` up to ~260_000; raise via
    ``quadrants._scratch.set_scratch_bytes(...)`` for larger inputs.
    """
    _validate_inputs(keys_in, values_in, keys_out, values_out, num_runs)
    N = keys_in.shape[0]
    if N == 0:
        return

    scratch = get_scratch_u32()
    cap = scratch_capacity_u32()
    B0 = (N + BLOCK_DIM - 1) // BLOCK_DIM
    positions_off = 0
    partials_off = N
    if partials_off + B0 > cap:
        raise RuntimeError(
            f"device_reduce_by_key_add on N={N} needs >= {partials_off + B0} u32 scratch slots, "
            f"but only {cap} are configured. Call quadrants._scratch.set_scratch_bytes(...) "
            f"before any algorithm runs."
        )

    identity_bits = _identity_bits(0, i32)
    op = _bin_add
    dtype = i32

    # Step 1: head_flags -> scratch[0:N].
    _rbk_head_flags(keys_in, scratch, positions_off, N)

    # Step 2: in-place exclusive scan of head_flags -> positions (still in scratch[0:N]).
    # Mirrors the 3-pass dance in _select.py but with scratch as both source and dest
    # for Pass 1 / Pass 3 (the existing kernels support src == dst aliasing).
    if N > BLOCK_DIM:
        _reduce_pass(
            scratch,
            scratch,
            positions_off,
            partials_off,
            N,
            B0 * BLOCK_DIM,
            identity_bits,
            op,
            dtype,
            True,
            True,
        )
        _exclusive_scan_inplace_u32(scratch, partials_off, B0, identity_bits, op, dtype, partials_off + B0)
        _scan_pass3(
            scratch,
            positions_off,
            scratch,
            partials_off,
            scratch,
            positions_off,
            N,
            B0 * BLOCK_DIM,
            identity_bits,
            op,
            dtype,
            True,
            True,
        )
    else:
        # Single-tile fast path: one block scans scratch[0:N] in place. Pass 1 still writes a single partial that is
        # then trivially scanned, but it's cheaper to inline a 1-block scan kernel that reads + writes scratch
        # directly. Reuse _exclusive_scan_inplace_u32's base case here.
        _exclusive_scan_inplace_u32(scratch, positions_off, N, identity_bits, op, dtype, partials_off)

    # Step 3: zero-init values_out (only the prefix that the scatter can touch).
    _rbk_zero_values_out(values_out, N, values_in.dtype)

    # Step 4: scatter keys + atomic-add values.
    _rbk_scatter(keys_in, values_in, scratch, positions_off, keys_out, values_out, N)

    # Step 5: write num_runs.
    _rbk_count(keys_in, scratch, positions_off, N, num_runs)


__all__ = ["device_reduce_by_key_add"]
