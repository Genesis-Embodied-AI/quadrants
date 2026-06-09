# type: ignore
"""Device-wide stream compaction (``select`` / ``compact``).

``qd.algorithms.select(arr, flags, out, num_out)`` packs the elements of ``arr`` for which the corresponding
``flags`` entry is set into a dense prefix of ``out``, in stable input order, and writes the count of selected
elements to ``num_out[0]``. Each ``flags[i]`` must be exactly ``0`` or ``1`` (``1`` selects); the algorithm
prefix-sums ``flags`` directly as counts, so non-0/1 values produce wrong indices and counts (caller's responsibility,
no normalization pass).

Algorithm (textbook scan-based compaction), emitted as a fixed-depth staircase of ``@qd.func`` phases inside a
single kernel launch (``select`` validates + sizes on the host, then launches ``_select_kernel`` -> ``select_func``;
``select_func`` is also public so qipc can compose the compaction at the top level of its own ``graph=True`` kernel):

1. Exclusive prefix-sum the ``flags`` (treated as 0 / 1) into the caller's ``u32`` scratch, producing per-element
   write indices. This reuses the same staircase phases (:func:`_reduce_phase` / :func:`_scan_downsweep_phase`) as
   ``exclusive_scan_add`` but targets a scratch slice for the output instead of a caller-supplied ``out`` tensor.
2. A "scatter" phase reads ``arr[i]`` and ``flags[i]``, and if the flag is set, writes ``out[indices[i]] = arr[i]``.
3. A 1-thread tail phase sums ``indices[N-1] + flags[N-1]`` (= total count) and stores it in ``num_out[0]``.

Scratch layout for the scan: ``scratch[0 : N]`` holds the per-element indices (i32 bit-cast to u32).
``scratch[N : N + B0]`` holds the level-0 partials, ``scratch[N + B0 : ...]`` deeper recursion levels (mirrors the
device scan). The scratch is *always* u32 regardless of the element dtype, because the scan operates on
flags-as-counts (i32) which always fit in u32; the element dtype only shows up at scatter time as
``dst[idx] = src[i]``, which lowers per-field for struct dtypes without any scratch reinterpretation.

This is why ``select`` works on any element dtype Quadrants supports for field assignment - scalars
(``i32`` / ``u32`` / ``f32`` / ``i64`` / ``u64`` / ``f64``) and structs (libuipc ``Vector{2,3,4}i``,
``LinearBVHAABB``, etc.).

**Scratch.** ``select`` needs a **caller-owned** 1-D ``u32`` scratch buffer of
:func:`select_scratch_slots` ``(N)`` slots (the per-element indices ``scratch[0:N]`` plus the scan partials
above them). ``u32`` regardless of the element dtype (the scan operates on flags-as-counts). There is no
module-level shared scratch - the caller always owns the buffer; a too-small buffer raises
:class:`InsufficientScratchError`.
"""

from quadrants.lang.kernel_impl import func as _func
from quadrants.lang.kernel_impl import kernel
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt.reductions import _bin_add
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32

from ._reduce import (
    BLOCK_DIM,
    _OP_ADD,
    _reduce_depth_for_n,
    _reduce_phase,
    _validate_caller_scratch,
)
from ._scan import (
    _emit_scan_inplace,
    _scan_downsweep_phase,
    _scan_tile_phase,
    _scan_total_scratch_slots,
)


@_func
def _select_scatter_phase(
    src: template(),
    flags: template(),
    indices: template(),
    indices_off: i32,
    dst: template(),
    n_valid: i32,
):
    """Scatter phase: write ``dst[indices[i]] = src[i]`` for every ``i`` where ``flags[i] != 0``. ``indices`` is the
    u32 scratch slice holding the exclusive scan of ``flags`` (i.e. the destination index of each surviving element);
    we ``bit_cast`` it back to ``i32`` before indexing.

    ``@qd.func`` phase (the single top-level ``for`` becomes its own offloaded launch / graph node). No race: each
    surviving thread writes to a distinct ``dst`` slot (by construction of the exclusive scan over 0 / 1 flags).
    Dropped threads do not write.
    """
    for i in range(n_valid):
        if flags[i] != 0:
            dst_idx = bit_cast(indices[indices_off + i], i32)
            dst[dst_idx] = src[i]


@_func
def _select_count_phase(
    flags: template(),
    indices: template(),
    indices_off: i32,
    n_valid: i32,
    num_out: template(),
):
    """1-thread tail phase: ``num_out[0] = indices[N-1] + flags[N-1]``.

    Its own offloaded launch, so the phase boundary serializes against the preceding scan writes (no grid sync
    needed in the parent).
    """
    for _ in range(1):
        last_idx = bit_cast(indices[indices_off + n_valid - 1], i32)
        last_flag = flags[n_valid - 1]
        last_inc = i32(0)
        if last_flag != 0:
            last_inc = i32(1)
        num_out[0] = last_idx + last_inc


def _emit_select_scan(flags, scratch, n, DEPTH):
    """Emit the exclusive prefix-sum of ``flags`` (0/1 counts) into the ``u32`` index slice ``scratch[0:n]`` at
    kernel-compile time, with the per-tile partials staircase stacked at ``scratch[n:]``.

    The select-specific layout of the same out-of-place staircase that backs :func:`exclusive_scan_add_func`: the
    indices and the partials live in one ``u32`` buffer (indices at offset 0, partials at offset ``n``), so it can't
    reuse ``_emit_scan`` directly (that takes a separate ``out``). ``flags`` is read-only (the scatter / count phases
    re-read it), so the scan is out-of-place into ``scratch``. ``DTYPE`` is ``i32`` (flags-as-counts) staged through a
    ``u32`` (``WIDE``) scratch. ``n`` flows as an ``Expr``; ``DEPTH`` is the compile-time phase count.
    """
    if DEPTH == 1:
        _scan_tile_phase(flags, scratch, 0, 0, n, i32, u32, _OP_ADD, _bin_add, False, True)
        return
    B0 = (n + (BLOCK_DIM - 1)) // BLOCK_DIM
    part_off = n  # indices occupy scratch[0:n]; the level-0 partials start right above them
    _reduce_phase(flags, scratch, 0, part_off, n, B0 * BLOCK_DIM, i32, u32, _OP_ADD, _bin_add, False, True)
    _emit_scan_inplace(scratch, part_off, B0, DEPTH - 2, i32, u32, _OP_ADD, _bin_add)
    _scan_downsweep_phase(flags, scratch, scratch, 0, part_off, 0, n, B0 * BLOCK_DIM, i32, u32, _OP_ADD, _bin_add, False, True)


@_func
def select_func(
    arr: template(),
    flags: template(),
    out: template(),
    num_out: template(),
    scratch: template(),
    n: i32,
    DEPTH: template(),
):
    """Graph-composable stream compaction - the ``@qd.func`` form of :func:`select`.

    Call at the **top level** of your own ``@qd.kernel`` (e.g. a qipc ``graph=True`` parent); never nest it in
    ordinary runtime ``for`` / ``if`` / ``while`` control flow. ``n`` is the live element count as a device ``Expr``;
    ``DEPTH`` is the compile-time phase count - select handles any count ``<= BLOCK_DIM ** DEPTH``. ``flags`` is an
    ``i32`` 0/1 mask the same length as ``arr``; selected ``arr[i]`` are packed into a dense prefix of ``out`` (size
    ``out`` for the all-selected case) and the count lands in ``num_out[0]``. ``scratch`` is a ``u32`` buffer sized via
    :func:`select_scratch_slots` ``(capacity_n)`` (it stages the per-element indices plus the scan partials); no DTYPE
    is needed because the scatter ``out[idx] = arr[i]`` lowers per-field for any element dtype."""
    _emit_select_scan(flags, scratch, n, DEPTH)
    _select_scatter_phase(arr, flags, scratch, 0, out, n)
    _select_count_phase(flags, scratch, 0, n, num_out)


@kernel
def _select_kernel(
    arr: template(),
    flags: template(),
    out: template(),
    num_out: template(),
    scratch: template(),
    n: i32,
    DEPTH: template(),
):
    """Host-launch wrapper for :func:`select_func` (one launch; the scan + scatter + count phases are emitted inside).
    ``n`` is a plain runtime count (the host knows ``N``). Private - the public host entry is :func:`select`."""
    select_func(arr, flags, out, num_out, scratch, n, DEPTH)


def select_scratch_slots(n: int) -> int:
    """Number of ``u32`` scratch slots :func:`select` needs to compact a length-``n`` input.

    Host- **and** kernel-callable (branch-free integer arithmetic over an unrolled fixed loop, no device round-trip):
    pass a Python ``int`` to size an allocation, or call it inside a kernel on a device-read ``N`` to validate
    against ``scratch.shape[0]`` on-device. Dtype-independent (the scan operates on flags-as-counts, which are always
    ``u32``). Layout: ``scratch[0:n]`` holds the per-element write indices, ``scratch[n:]`` the scan partials (plus
    any deeper recursion levels). Allocate up front::

        scratch = qd.Tensor(qd.ndarray(qd.u32, shape=max(qd.algorithms.select_scratch_slots(N), 1)))

    Returns ``0`` for ``n <= 0``.
    """
    pos = n > 0
    B0 = (n + BLOCK_DIM - 1) // BLOCK_DIM
    return _scan_total_scratch_slots(B0, partials_cursor=n + B0) * pos


def select(arr, flags, out, num_out, scratch):
    """Stream-compact ``arr`` by ``flags``: copy ``arr[i]`` to a dense prefix of ``out`` for every ``i`` where
    ``flags[i] == 1``, in stable input order. Write the count of selected elements to ``num_out[0]``.

    Args:
        arr: 1-D tensor of any element dtype that Quadrants supports field-element assignment for: scalars
            (``i32`` / ``u32`` / ``f32`` / ``i64`` / ``u64`` / ``f64``) and structs (``qd.Struct.field({...})`` or
            ``qd.types.struct(...)`` - e.g. the libuipc ``Vector{2,3,4}i`` shapes). The scatter is
            ``dst[idx] = src[i]``, which lowers per-field for struct dtypes, so no scratch reinterpretation is
            needed for wider / composite element types.
        flags: 1-D ``i32`` tensor, same shape as ``arr``. **Every entry must be exactly ``0`` or ``1``** (``1``
            selects). Non-0/1 values produce incorrect results - the algorithm prefix-sums ``flags`` directly as
            counts, so a stray ``2`` would advance the destination cursor by 2 and break the dense-output / count
            contract. Caller-built: populate with a separate kernel that applies your predicate, writing exactly
            ``1`` for selected and ``0`` otherwise.
        out: 1-D tensor with the same dtype as ``arr``. Must hold at least ``N`` elements (so a
            worst-case-everyone-selected run fits); only the prefix ``out[0 : num_out[0]]`` is meaningful on return.
        num_out: 1-element ``i32`` tensor receiving the selected count.
        scratch: caller-owned 1-D ``u32`` workspace of :func:`select_scratch_slots` ``(N)`` slots. There is no
            module-level shared scratch; a too-small buffer raises :class:`InsufficientScratchError`.

    Same async / no-implicit-sync contract as ``reduce_*`` and ``exclusive_scan_*``: ``num_out`` is a
    tensor, not a Python scalar - call ``num_out.to_numpy()[0]`` explicitly to get the count host-side.

    See the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md`` for the scratch-into-indices layout and
    the algorithm reference.
    """
    if not hasattr(arr, "shape") or len(arr.shape) != 1:
        raise TypeError(f"select expects a 1-D arr; got shape {getattr(arr, 'shape', None)}")
    if not hasattr(flags, "shape") or flags.shape != arr.shape:
        raise TypeError(f"select expects flags.shape == arr.shape; got arr={arr.shape}, flags={flags.shape}")
    if flags.dtype != i32:
        raise TypeError(f"select expects flags.dtype == qd.i32; got {flags.dtype}")
    if not hasattr(out, "shape") or len(out.shape) != 1:
        raise TypeError(f"select expects a 1-D out; got shape {getattr(out, 'shape', None)}")
    if out.dtype != arr.dtype:
        raise TypeError(f"select dtype mismatch: arr={arr.dtype}, out={out.dtype}")
    if out.shape[0] < arr.shape[0]:
        raise ValueError(
            f"select out.shape[0]={out.shape[0]} < arr.shape[0]={arr.shape[0]}; "
            "out must hold at least the input size to be safe in the all-selected case"
        )
    if not hasattr(num_out, "shape") or num_out.shape != (1,):
        raise TypeError(f"select expects num_out.shape == (1,); got {num_out.shape}")
    if num_out.dtype != i32:
        raise TypeError(f"select expects num_out.dtype == qd.i32; got {num_out.dtype}")

    N = arr.shape[0]
    if N == 0:
        return

    # Scratch layout: scratch[0:N] = indices, scratch[N : N + B0] = level-0 partials, then deeper levels above.
    _validate_caller_scratch("select", N, scratch, select_scratch_slots(N), u32)
    depth = _reduce_depth_for_n(N)
    # One launch: the scan-of-flags staircase + scatter + count are emitted inside _select_kernel as @qd.func phases
    # (the same fixed-depth scan that backs exclusive_scan_add). N == 1 falls out of the single-tile base case.
    _select_kernel(arr, flags, out, num_out, scratch, N, depth)


__all__ = ["select", "select_func", "select_scratch_slots"]
