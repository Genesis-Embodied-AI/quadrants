"""Tensors: per-tensor backend and (later) layout.

This module is the user-facing entry point for selecting a tensor backend
(``qd.field`` vs ``qd.ndarray``) on a per-tensor basis.

See ``docs/source/user_guide/tensor.md`` for the user guide.
"""

from enum import IntEnum

__all__ = ["Backend", "tensor"]


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


def _coerce_backend(backend):
    if isinstance(backend, Backend):
        return backend
    try:
        return Backend(backend)
    except (ValueError, TypeError) as e:
        valid = ", ".join(f"qd.Backend.{m.name}" for m in Backend)
        raise ValueError(f"backend={backend!r} is not a valid qd.Backend; expected one of {valid}") from e


def tensor(dtype, shape, *, backend=Backend.FIELD, **kwargs):
    """Allocate a tensor on the chosen backend.

    Thin dispatcher over :func:`quadrants.field` and :func:`quadrants.ndarray`
    that selects between the two via the :class:`Backend` enum.

    Args:
        dtype: Element data type (e.g. ``qd.f32``, ``qd.i32``, or a compound
            type from ``qd.types``).
        shape: Shape of the tensor as an ``int`` or tuple of ``int``.
        backend (Backend, optional): Storage backend. Defaults to
            :attr:`Backend.FIELD`.
        **kwargs: Forwarded verbatim to the underlying ``qd.field`` or
            ``qd.ndarray`` call. Each backend accepts a different set of
            keyword arguments — see their docstrings for details.

    Returns:
        A ``ScalarField`` when ``backend == Backend.FIELD``, or an
        ``Ndarray`` when ``backend == Backend.NDARRAY``.

    Example::

        >>> import quadrants as qd
        >>> qd.init(arch=qd.x64)
        >>> a = qd.tensor(qd.f32, shape=(4, 5))
        >>> b = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY)

    Raises:
        ValueError: If ``backend`` is not a valid :class:`Backend` member.
    """
    backend = _coerce_backend(backend)
    # pylint: disable=import-outside-toplevel
    from quadrants.lang import impl  # late import to break circular dependency

    if backend is Backend.FIELD:
        return impl.field(dtype, shape, **kwargs)
    if backend is Backend.NDARRAY:
        return impl.ndarray(dtype, shape, **kwargs)
    raise AssertionError(f"unhandled Backend member: {backend!r}")
