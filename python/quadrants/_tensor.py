"""Tensors: per-tensor backend and (later) layout.

This module is the user-facing entry point for selecting a tensor backend
(``qd.field`` vs ``qd.ndarray``) on a per-tensor basis. Currently it exports
only the :class:`Backend` enum; the ``qd.tensor(...)`` factory and
``layout=`` support land in subsequent PRs.

See ``docs/source/user_guide/tensor.md`` for the user guide.
"""

from enum import IntEnum

__all__ = ["Backend"]


class Backend(IntEnum):
    """Tensor storage backend.

    Each value selects one of Quadrants' two underlying tensor implementations:

    - :attr:`FIELD` (``qd.field``): faster at runtime; recompiles kernels
      whenever any dimension size changes. Best for tensors whose shape is
      effectively static across a run.
    - :attr:`NDARRAY` (``qd.ndarray``): slightly slower at runtime but avoids
      kernel recompilation when sizes change. Best for tensors whose shape
      varies frequently (e.g. dynamic batch sizes, growing buffers).

    The choice is made per tensor at allocation time. A single program can
    freely mix both backends.
    """

    FIELD = 0
    NDARRAY = 1
