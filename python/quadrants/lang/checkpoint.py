"""User-facing ``qd.checkpoint`` context-manager and its no-op Python-runtime stub.

Mirrors ``graph_status.py``, which holds the other half of the same feature surface (``GraphStatus``). Kept in its own
module to keep ``lang/misc.py`` from growing further -- the AST transformer and the C++ runtime are doing all the actual
implementation work; this file is just the public API entry point.

Re-exported via ``qd.lang.misc`` (and therefore as ``qd.checkpoint``) for the user-facing canonical import path.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def checkpoint(cp_id, yield_on):
    """Marks a section of a graph kernel as a yieldable resume target.

    .. warning::

        **Experimental.** ``qd.checkpoint`` (together with ``qd.GraphStatus`` and ``kernel.resume(from_checkpoint=...)``)
        is an experimental API. The signature, the lowering across backends, the error messages, and the host-side
        yield/resume contract may change in any future release without a deprecation cycle.

    Used as ``with qd.checkpoint(cp_id, yield_on=flag):`` inside a ``@qd.kernel(graph=True, checkpoints=True)`` kernel
    body. The ``cp_id`` is the user-facing label (``int`` or ``IntEnum``) that ``GraphStatus.checkpoint`` and
    ``kernel.resume(from_checkpoint=...)`` will refer to. Labels must be unique within the kernel; values are otherwise
    opaque (the runtime uses dense, source-declaration-order internal cp_ids and maps to/from your labels). The
    ``yield_on`` argument must be a kernel parameter that is a 0-d ``qd.types.ndarray(qd.i32, ndim=0)``; the body may
    write a non-zero value into it to signal "the host needs to handle something before this checkpoint can complete".

    When the kernel is decorated with ``checkpoints=True``, every top-level for-loop in the kernel body that is **not**
    inside a ``with qd.checkpoint(...)`` block is auto-wrapped in an implicit, no-yield checkpoint. On a resume launch
    those implicit checkpoints are skipped along with the explicit ones declared earlier in source order. Implicit
    checkpoints have no user-facing label and never appear in ``GraphStatus.checkpoint``.

    On CUDA SM 9.0+ / 12.4+ each checkpoint compiles to a CUDA graph IF conditional node around its body kernels. On
    other GPU backends it lowers to a small "gate" kernel that rewrites per-kernel indirect-dispatch grid dimensions to
    ``(0, 0, 0)`` to skip the checkpoint body. CPU runs checkpoint bodies behind a host branch.

    Restrictions (enforced at kernel compile time):
      - Must be used inside ``@qd.kernel(graph=True, checkpoints=True)``.
      - ``cp_id`` must be an ``int`` (or ``IntEnum`` value), and must be unique across the kernel.
      - ``yield_on`` must name a kernel parameter that is a 0-d ``qd.types.ndarray(qd.i32, ndim=0)``.
      - Checkpoints cannot be nested inside other checkpoints. Checkpoints inside a ``qd.graph_do_while`` body are fine
        and are the expected pattern.
      - Cannot be combined with ``qd.stream_parallel()`` in the same kernel.
      - The body cannot contain bare top-level statements (assignments, expressions); wrap them in
        ``for _ in range(1):`` so the lowering surfaces the per-statement task cost.

    This function should not be called directly at runtime; it is recognised and transformed during AST compilation. At
    Python runtime (outside kernels), this is a no-op context manager so that doctests / type-checking can import the
    symbol freely.

    See also ``docs/source/user_guide/graph.md`` for the host-side yield/resume loop and cross-backend semantics.
    """
    del cp_id, yield_on
    yield
