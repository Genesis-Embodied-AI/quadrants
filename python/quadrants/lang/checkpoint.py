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
    """Marks a section of a graph kernel as a pause / resume point.

    .. warning::

        **Experimental.** ``qd.checkpoint`` (together with ``qd.GraphStatus`` and
        ``kernel.resume(from_checkpoint=...)``) is an experimental API. The signature, the lowering across backends, the
        error messages, and the host-side yield/resume contract may change in any future release without a deprecation
        cycle.

    Used as ``with qd.checkpoint(cp_id, yield_on=flag):`` inside a ``@qd.kernel(graph=True, checkpoints=True)`` kernel
    body. When the body writes a non-zero value into ``flag``, the kernel pauses at this checkpoint and returns a
    ``GraphStatus`` to the host carrying ``status.checkpoint == cp_id``. The host can then fix things up and call
    ``kernel.resume(..., from_checkpoint=cp_id)`` to continue from the same point on the next launch.

    Arguments:
        cp_id: User-facing label identifying this checkpoint to the host. Must be an ``int`` literal or an ``IntEnum``
            value, and must be unique within the kernel. The value is preserved as-is end-to-end -- if you pass
            ``Stage.SIM`` (an ``IntEnum`` member), ``status.checkpoint`` round-trips back as ``Stage.SIM`` rather than
            the raw int.
        yield_on: Name of a kernel parameter that is a 0-d ``qd.types.ndarray(qd.i32, ndim=0)``. The body may write a
            non-zero value into it to signal "pause here, host needs to handle something". The framework never writes
            into this buffer -- the host owns it end-to-end and must initialise it to ``0`` before the first launch
            (``qd.ndarray`` is not zero-initialised) AND reset it to ``0`` before each ``kernel.resume(...)`` call
            (otherwise the same checkpoint sees the stale non-zero value and yields again).

    Restrictions (enforced at kernel compile time):
      - Must be used inside ``@qd.kernel(graph=True, checkpoints=True)``.
      - ``cp_id`` must be an ``int`` (or ``IntEnum`` value), and must be unique across the kernel.
      - ``yield_on`` must name a kernel parameter that is a 0-d ``qd.types.ndarray(qd.i32, ndim=0)``.
      - Checkpoints cannot be nested inside other checkpoints. Checkpoints inside a ``qd.graph.do_while`` body are fine.
      - Cannot be combined with ``qd.stream_parallel()`` in the same kernel.
      - The body cannot contain bare top-level statements (assignments, expressions); wrap them in
        ``for _ in range(1):`` so the lowering surfaces the per-statement task cost.

    This function should not be called directly at runtime; it is recognised and transformed during AST compilation. At
    Python runtime (outside kernels), this is a no-op context manager so that doctests / type-checking can import the
    symbol freely.

    See ``docs/source/user_guide/graph.md`` for the host-side yield/resume loop and cross-backend semantics.
    """
    del cp_id, yield_on
    yield
