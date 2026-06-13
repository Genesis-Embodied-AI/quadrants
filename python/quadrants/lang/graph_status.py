"""Plain-Python container returned from graph kernels that contain ``qd.checkpoint(yield_on=...)``.

Lives in its own module (with no Quadrants-internal imports) so it can be imported safely from both ``kernel.py`` and
``misc.py`` without re-introducing the circular import chain that ``misc.py -> impl.py -> kernel.py`` would create.

Re-exported via ``qd.lang.misc`` (and therefore as ``qd.GraphStatus``) for the user-facing canonical import path.
"""

from __future__ import annotations


class GraphStatus:
    """Result returned by a graph kernel that contains ``qd.checkpoint(yield_on=...)`` blocks.

    .. warning::

        **Experimental.** ``GraphStatus`` is part of the experimental ``qd.checkpoint`` surface; its attributes and
        the conditions under which it is returned may change in any future release without a deprecation cycle.

    Returned from ``kernel(...)`` and ``kernel.resume(..., from_checkpoint=cp)`` whenever the kernel was decorated with
    ``@qd.kernel(graph=True)`` and contains at least one checkpoint that has declared a ``yield_on=`` parameter. Read
    ``status.yielded`` to decide whether to keep running the host loop, and ``status.checkpoint`` to find out which
    checkpoint asked the host to handle something.

    Canonical usage (mirrors the qipc re-entrant pattern; see ``graph.md``)::

        status = step(arr, overflow_flag, newton_cond)
        while status.yielded:
            handle_overflow_for(status.checkpoint, ...)
            status = step.resume(arr, overflow_flag, newton_cond,
                                 from_checkpoint=status.checkpoint)

    Attributes:
        yielded: ``True`` iff one of the kernel's ``yield_on=`` checkpoints fired its flag on the most recent launch.
            ``False`` means the kernel completed normally and the host loop should exit.
        checkpoint: ``cp_id`` of the checkpoint whose ``yield_on=`` flag was non-zero (or ``None`` when ``yielded`` is
            ``False``). Pass it to ``kernel.resume(..., from_checkpoint=cp)`` to skip every checkpoint with a lower
            ``cp_id`` on the next launch.
    """

    __slots__ = ("yielded", "checkpoint")

    def __init__(self, yielded: bool, checkpoint: int | None):
        self.yielded = yielded
        self.checkpoint = checkpoint

    def __repr__(self) -> str:
        if self.yielded:
            return f"GraphStatus(yielded=True, checkpoint={self.checkpoint})"
        return "GraphStatus(yielded=False)"
