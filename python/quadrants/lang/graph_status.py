"""Plain-Python container returned from graph kernels that contain ``qd.checkpoint(yield_on=...)``.

Lives in its own module (with no Quadrants-internal imports) so it can be imported safely from both ``kernel.py`` and
``misc.py`` without re-introducing the circular import chain that ``misc.py -> impl.py -> kernel.py`` would create.

Re-exported via ``qd.lang.misc`` (and therefore as ``qd.GraphStatus``) for the user-facing canonical import path.
"""

from __future__ import annotations


class GraphStatus:
    """Result returned by a graph kernel that contains ``qd.checkpoint(cp_id, yield_on=...)`` blocks.

    .. warning::

        **Experimental.** ``GraphStatus`` is part of the experimental ``qd.checkpoint`` surface; its attributes and
        the conditions under which it is returned may change in any future release without a deprecation cycle.

    Returned from ``kernel(...)`` and ``kernel.resume(..., from_checkpoint=label)`` whenever the kernel was decorated
    with ``@qd.kernel(graph=True, checkpoints=True)`` and contains at least one ``qd.checkpoint(...)`` block. Read
    ``status.yielded`` to decide whether to keep running the host loop, and ``status.checkpoint`` to find out which
    checkpoint asked the host to handle something.

    Canonical usage (see ``graph.md``)::

        from enum import IntEnum

        class Stage(IntEnum):
            SIM = 0

        overflow_flag[()] = 0  # initialise before the first launch
        status = step(arr, overflow_flag, newton_cond)
        while status.yielded:
            handle_overflow_for(status.checkpoint, ...)
            overflow_flag[()] = 0  # the framework never clears your yield_on flag
            status = step.resume(arr, overflow_flag, newton_cond,
                                 from_checkpoint=status.checkpoint)

    Attributes:
        yielded: ``True`` iff one of the kernel's checkpoints paused on the most recent launch (i.e. its ``yield_on=``
            flag was non-zero). ``False`` means the kernel completed normally and the host loop should exit.
        checkpoint: The ``cp_id`` label of the checkpoint that paused (an ``int`` or the original ``IntEnum`` instance
            you passed to ``qd.checkpoint(cp_id, ...)``), or ``None`` when ``yielded`` is ``False``. Pass it to
            ``kernel.resume(..., from_checkpoint=...)`` to continue from that point on the next launch.
    """

    __slots__ = ("yielded", "checkpoint")

    def __init__(self, yielded: bool, checkpoint: int | None):
        self.yielded = yielded
        self.checkpoint = checkpoint

    def __repr__(self) -> str:
        if self.yielded:
            # Use `!r` so an `IntEnum` cp_id is shown as `<Stage.LOAD: 0>` rather than collapsing to its raw int via
            # `int.__format__` (which Python 3.10's `f"{IntEnum.X}"` does). Plain ints round-trip unchanged.
            return f"GraphStatus(yielded=True, checkpoint={self.checkpoint!r})"
        return "GraphStatus(yielded=False)"
