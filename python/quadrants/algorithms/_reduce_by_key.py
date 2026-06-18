# type: ignore
"""Device-wide reduce-by-key.

Implements ``qd.algorithms.reduce_by_key_add`` on top of the existing device exclusive scan internals and a
**caller-owned** ``u32`` scratch buffer (sized via :func:`reduce_by_key_scratch_slots`).

Reduce-by-key takes two parallel 1-D tensors - ``keys`` and ``values`` - and collapses every **consecutive run of
equal keys** into a single output entry ``(unique_key, sum_of_values_in_run)``. Keys that are equal but separated by
other keys are treated as separate runs. To compute a global per-key sum, sort by key first (e.g. via
``qd.algorithms.sort``) and then reduce-by-key.

Algorithm (scan + scatter; no segmented-scan primitive needed), emitted as a fixed-depth staircase of ``@qd.func``
phases (call ``reduce_by_key_add`` at the **top level** of your own ``@qd.kernel`` - e.g. a qipc ``graph=True``
parent - with the live count ``n`` as a device ``Expr`` and the compile-time ``LOG256_MAX_N`` phase count):

1. **Head-flag pass** (``_rbk_head_flags_phase``). Compute ``head_flags[i] = 1`` if ``i == 0 or keys[i] != keys[i-1]``,
   else ``0``, directly into the caller's ``u32`` scratch ``scratch[0:N]`` (storing the ``i32`` flag bit-cast to
   ``u32``).
2. **Exclusive scan of head_flags** (in-place over ``scratch[0:N]``, via :func:`_emit_scan_inplace` - the same
   staircase phases as ``exclusive_scan_add``). After this,
   ``scratch[i] = exclusive_scan(head_flags)[i] = sum(head_flags[0:i])``. The 0-indexed run index of element ``i`` is
   then ``positions[i] = scratch[i] + head_flag(i) - 1`` (i.e. ``inclusive_scan(head_flags)[i] - 1``); the scatter
   pass recomputes ``head_flag(i)`` from the two keys at ``i`` and ``i - 1`` so the ``head_flags`` array itself does
   not need to survive the scan. This lets the scan run in place, holding scratch to ~``1.004 * N`` slots.
3. **Zero-init values_out**. The scatter step uses ``atomic_add`` on ``values_out[positions[i]]``; the slots must
   start at the additive identity ``0``.
4. **Scatter pass** (``_rbk_scatter_phase``). For every ``i``:
   - Recompute ``head_flag(i)`` from ``i == 0 or keys[i] != keys[i-1]`` and compute the run index ``pos = scratch[i]
     + head_flag(i) - 1``.
   - ``keys_out[pos] = keys[i]`` - race-free because every thread in a run writes the same key to the same slot.
   - ``atomic_add(values_out[pos], values[i])`` folds the run's values into the run's output slot.
5. **Count pass** (``_rbk_count_phase``). Computes ``num_runs[0] = scratch[N-1] + head_flag(N-1)`` where the head flag
   at ``N-1`` is recomputed from ``keys[N-1] != keys[N-2]`` for ``N >= 2`` (``1`` for ``N == 1``).

This first-land scope supports only the ``add`` reduction. ``min`` / ``max`` variants would need ``atomic_min`` /
``atomic_max``, which have spottier cross-backend support for ``f32`` - defer to a follow-up gated on real qipc usage.

**Scratch.** A **caller-owned** 1-D ``u32`` buffer of :func:`reduce_by_key_scratch_slots` ``(N)`` slots
(``positions = scratch[0:N]`` plus the scan partials above them, ≈ ``1.004 * N``). There is no module-level shared
scratch - the caller always owns the buffer.
"""

from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.misc import loop_config
from quadrants.lang.ops import atomic_add, bit_cast
from quadrants.lang.simt.reductions import _bin_add
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32

from ._reduce import (
    _OP_ADD,
    BLOCK_DIM,
    _at_least_one,
)
from ._scan import _emit_scan_inplace, _scan_total_scratch_slots


@_func
def _rbk_head_flags_phase(keys_in: template(), head_flags: template(), head_flags_off: i32, N: i32):
    """Write ``head_flags[i] = 1 if (i == 0 or keys[i] != keys[i-1]) else 0`` to ``head_flags[head_flags_off + i]``
    (as the u32 bit pattern of i32).

    ``@qd.func`` phase (its single top-level ``for`` becomes its own offloaded launch / graph node). Linear-time,
    embarrassingly parallel: each thread reads at most two key elements (``keys[i]`` and ``keys[i-1]``) and writes one
    flag. The boundary thread at ``i == 0`` always writes ``1`` since there is no predecessor and a run trivially
    starts there.
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


@_func
def _rbk_zero_values_out_phase(values_out: template(), N: i32, VALUE_DTYPE: template()):
    """Set ``values_out[0 : N] = 0`` so the scatter ``atomic_add`` lands onto a clean additive identity. ``N`` is the
    upper bound on ``num_runs``; the caller-supplied ``values_out`` may be longer but we only need the prefix that the
    scatter can touch.

    We write ``bit_cast(u32(0), VALUE_DTYPE)`` rather than relying on ``v - v == 0`` because the latter compiles to a
    real subtract for ``f32`` (and yields NaN if the slot held NaN garbage from a prior allocation), whereas the
    bit-cast lowers to a plain store.
    """
    for i in range(N):
        values_out[i] = bit_cast(u32(0), VALUE_DTYPE)


@_func
def _rbk_scatter_phase(
    keys_in: template(),
    values_in: template(),
    positions: template(),
    positions_off: i32,
    keys_out: template(),
    values_out: template(),
    N: i32,
):
    """Per-element scatter phase:

    - Compute ``head_flag(i)`` on the fly from ``i == 0 or keys[i] != keys[i-1]`` and combine with the in-place
      exclusive scan stored in ``positions`` to recover the inclusive run index
      ``pos = positions[i] + head_flag(i) - 1``.
    - ``keys_out[pos] = keys_in[i]`` - race-free because every thread in a run writes the same key to the same slot.
    - ``atomic_add(values_out[pos], values_in[i])`` - folds the run's values into the run's output slot.
      ``values_out`` must be pre-zeroed (see ``_rbk_zero_values_out_phase``).
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


@_func
def _rbk_count_phase(keys_in: template(), positions: template(), positions_off: i32, N: i32, num_runs: template()):
    """One-thread tail phase: write ``num_runs[0] = total head_flag count``.

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


@_func(requires_top_level=True)
def reduce_by_key_add(
    keys_in: template(),
    values_in: template(),
    keys_out: template(),
    values_out: template(),
    num_runs: template(),
    scratch: template(),
    n: i32,
    VALUE_DTYPE: template(),
    LOG256_MAX_N: template(),
):
    """Graph-composable reduce-by-key (add).

    **Experimental** - this API is new and may change in a future release.

    Call at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent); never nest it in
    ordinary runtime ``for`` / ``if`` / ``while`` control flow. ``n`` is the live element count as a device ``Expr``;
    ``LOG256_MAX_N`` is the compile-time phase count (any count ``<= BLOCK_DIM ** LOG256_MAX_N``). ``VALUE_DTYPE``
    is the values dtype (needed only to write the typed zero before the scatter ``atomic_add``; keys are handled
    generically). The five phases - head flags, in-place exclusive scan of those flags (the same staircase as
    ``exclusive_scan_add``, via :func:`_emit_scan_inplace`), zero ``values_out``, scatter, count - each emit as their
    own offloaded launch. Size ``scratch`` via :func:`reduce_by_key_scratch_slots` ``(capacity_n)``."""
    _rbk_head_flags_phase(keys_in, scratch, 0, n)
    _emit_scan_inplace(scratch, 0, n, LOG256_MAX_N - 1, i32, u32, _OP_ADD, _bin_add)
    _rbk_zero_values_out_phase(values_out, n, VALUE_DTYPE)
    _rbk_scatter_phase(keys_in, values_in, scratch, 0, keys_out, values_out, n)
    _rbk_count_phase(keys_in, scratch, 0, n, num_runs)


def reduce_by_key_scratch_slots(n: int) -> int:
    """Number of ``u32`` scratch slots :func:`reduce_by_key_add` needs for a length-``n`` input.

    Host- **and** kernel-callable (branch-free integer arithmetic over an unrolled fixed loop, no device round-trip):
    pass a Python ``int`` to size an allocation, or call it inside a kernel on a device-read ``N`` to validate
    against ``scratch.shape[0]`` on-device. Dtype-independent (the scan operates on head-flags-as-counts, which are
    ``u32``). Layout: ``scratch[0:n]`` holds the run positions, ``scratch[n:]`` the scan partials (plus deeper
    recursion levels for ``n > BLOCK_DIM``). Allocate up front::

        scratch = qd.Tensor(qd.ndarray(qd.u32, shape=qd.algorithms.reduce_by_key_scratch_slots(N)))

    Always returns **at least 1** so the result can size an allocation directly: ``n`` for ``n <= BLOCK_DIM``
    (single-tile in-place scan) and ``1`` for ``n <= 0`` (no real scratch; the lone slot is never touched).
    """
    pos = n > 0
    big = n > BLOCK_DIM
    small_pos = pos * (1 - big)
    B0 = (n + BLOCK_DIM - 1) // BLOCK_DIM
    return _at_least_one(n * small_pos + _scan_total_scratch_slots(B0, partials_cursor=n + B0) * big)


__all__ = ["reduce_by_key_add", "reduce_by_key_scratch_slots"]
