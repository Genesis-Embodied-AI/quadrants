# type: ignore
"""Device-wide stream compaction (``select`` / ``compact``).

``qd.algorithms.select(arr, flags, out, num_out)`` packs the elements of ``arr`` for which the corresponding
``flags`` entry is set into a dense prefix of ``out``, in stable input order, and writes the count of selected
elements to ``num_out[0]``. Each ``flags[i]`` must be exactly ``0`` or ``1`` (``1`` selects); the algorithm
prefix-sums ``flags`` directly as counts, so non-0/1 values produce wrong indices and counts (caller's responsibility,
no normalization pass).

Algorithm (textbook scan-based compaction):

1. Exclusive prefix-sum the ``flags`` (treated as 0 / 1) into the shared ``Field(u32)`` scratch, producing per-element
   write indices. This reuses the same three-pass scan internals as ``exclusive_scan_add`` but targets a
   scratch slice for the output instead of a caller-supplied ``out`` tensor.
2. A single fused "scatter" kernel reads ``arr[i]`` and ``flags[i]``, and if the flag is set, writes
   ``out[indices[i]] = arr[i]``.
3. A 1-thread tail kernel sums ``indices[N-1] + flags[N-1]`` (= total count) and stores it in ``num_out[0]``.

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

from quadrants.lang.kernel_impl import kernel
from quadrants.lang.ops import bit_cast
from quadrants.lang.simt.reductions import _bin_add
from quadrants.types.annotations import template
from quadrants.types.primitive_types import i32, u32

from ._reduce import BLOCK_DIM, _identity_bits, _reduce_pass, _validate_caller_scratch
from ._scan import _exclusive_scan_inplace_u32, _scan_pass3, _scan_total_scratch_slots


@kernel
def _select_scatter(
    src: template(),
    flags: template(),
    indices: template(),
    indices_off: i32,
    dst: template(),
    n_valid: i32,
):
    """Scatter pass: write ``dst[indices[i]] = src[i]`` for every ``i`` where ``flags[i] != 0``. ``indices`` is the
    u32 scratch slice holding the exclusive scan of ``flags`` (i.e. the destination index of each surviving element);
    we ``bit_cast`` it back to ``i32`` before indexing.

    No race: each surviving thread writes to a distinct ``dst`` slot (by construction of the exclusive scan over
    0 / 1 flags). Dropped threads do not write.
    """
    for i in range(n_valid):
        if flags[i] != 0:
            dst_idx = bit_cast(indices[indices_off + i], i32)
            dst[dst_idx] = src[i]


@kernel
def _select_count(
    flags: template(),
    indices: template(),
    indices_off: i32,
    n_valid: i32,
    num_out: template(),
):
    """1-thread tail kernel: ``num_out[0] = indices[N-1] + flags[N-1]``.

    Split into its own launch so the host driver doesn't have to insert a grid sync after the scatter; the kernel
    boundary serializes against the preceding scan writes.
    """
    for _ in range(1):
        last_idx = bit_cast(indices[indices_off + n_valid - 1], i32)
        last_flag = flags[n_valid - 1]
        last_inc = i32(0)
        if last_flag != 0:
            last_inc = i32(1)
        num_out[0] = last_idx + last_inc


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
    B0 = (N + BLOCK_DIM - 1) // BLOCK_DIM
    indices_off = 0
    partials_off = N

    identity_bits = _identity_bits(0, i32)
    op = _bin_add
    dtype = i32

    # Three-pass scan of flags into scratch[0:N] (i32 indices, stored as u32 bit pattern). The general 3-pass path
    # collapses gracefully when N <= BLOCK_DIM: pass 1 writes a single partial, pass 2 in-place-scans it, pass 3
    # applies the (trivial) prefix to the single-tile scan.
    # Pass 1: per-block tile reduce of flags -> scratch[partials_off:]
    _reduce_pass(
        flags,
        scratch,
        0,
        partials_off,
        N,
        B0 * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        False,
        True,
    )
    # Pass 2: in-place scan of the partials (recursive if B0 > BLOCK_DIM).
    _exclusive_scan_inplace_u32(scratch, partials_off, B0, identity_bits, op, dtype, partials_off + B0)
    # Pass 3: flags + scanned partials -> scratch[0:N] (u32 indices)
    _scan_pass3(
        flags,
        0,
        scratch,
        partials_off,
        scratch,
        indices_off,
        N,
        B0 * BLOCK_DIM,
        identity_bits,
        op,
        dtype,
        False,
        True,
    )

    # Step 2: scatter src[i] -> dst[indices[i]] for every selected i.
    _select_scatter(arr, flags, scratch, indices_off, out, N)

    # Step 3: write the count.
    _select_count(flags, scratch, indices_off, N, num_out)


__all__ = ["select", "select_scratch_slots"]
