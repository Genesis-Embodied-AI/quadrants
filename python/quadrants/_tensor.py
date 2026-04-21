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


# Kwargs explicitly accepted by ``qd.tensor`` (in addition to the
# positional ``dtype`` and ``shape``). The factory hard-validates against
# this set so typos and backend-specific options don't silently work on
# one backend and raise cryptic errors deep in the other. Users who need
# backend-specific knobs (e.g. ``offset=`` for field offset indexing,
# ``order=`` for SoA layouts) should call ``qd.field`` or ``qd.ndarray``
# directly — they have explicitly opted out of the unified tensor API.
#
# The set grows as later branches add features:
# - PR 5: ``needs_grad``
# - PR 6: ``layout``
_ACCEPTED_KWARGS = frozenset({"backend"})


def _validate_kwargs(kwargs, *, factory_name):
    extra = set(kwargs) - _ACCEPTED_KWARGS
    if extra:
        accepted = ", ".join(sorted(_ACCEPTED_KWARGS | {"dtype", "shape"}))
        raise TypeError(
            f"{factory_name}() got unexpected keyword argument(s) "
            f"{sorted(extra)!r}; accepted: {accepted}"
        )


def tensor(dtype, shape, *, backend=Backend.NDARRAY, **kwargs):
    """Allocate a tensor on the chosen backend.

    Thin dispatcher over :func:`quadrants.field` and :func:`quadrants.ndarray`
    that selects between the two via the :class:`Backend` enum.

    Args:
        dtype: Element data type (e.g. ``qd.f32``, ``qd.i32``, or a compound
            type from ``qd.types``).
        shape: Shape of the tensor as an ``int`` or tuple of ``int``.
        backend (Backend, optional): Storage backend. Defaults to
            :attr:`Backend.NDARRAY`.

    Returns:
        A ``ScalarField`` when ``backend == Backend.FIELD``, or an
        ``Ndarray`` when ``backend == Backend.NDARRAY``.

    Example::

        >>> import quadrants as qd
        >>> qd.init(arch=qd.x64)
        >>> a = qd.tensor(qd.f32, shape=(4, 5))
        >>> b = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.FIELD)

    Raises:
        ValueError: If ``backend`` is not a valid :class:`Backend` member.
        TypeError: If any keyword argument outside the accepted set is
            passed (see ``_ACCEPTED_KWARGS``).
    """
    _validate_kwargs(kwargs, factory_name="qd.tensor")
    backend = _coerce_backend(backend)
    forwarded = {k: v for k, v in kwargs.items() if k != "backend"}
    # pylint: disable=import-outside-toplevel
    from quadrants.lang import impl

    if backend is Backend.FIELD:
        return impl.field(dtype, shape, **forwarded)
    if backend is Backend.NDARRAY:
        return impl.ndarray(dtype, shape, **forwarded)
    raise AssertionError(f"unhandled Backend member: {backend!r}")


def _tensor_vec(n, dtype, shape, *, backend=Backend.FIELD, **kwargs):
    """Private impl backing ``qd.Vector.tensor``.

    Dispatcher over ``qd.Vector.field`` and ``qd.Vector.ndarray`` selected
    by the ``backend=`` keyword. Not part of the public API — call
    ``qd.Vector.tensor(...)`` instead.
    """
    backend = _coerce_backend(backend)
    # pylint: disable-next=import-outside-toplevel  # late import to break circular dependency
    from quadrants.lang.matrix import Vector

    if backend is Backend.FIELD:
        return Vector.field(n, dtype, shape, **kwargs)
    if backend is Backend.NDARRAY:
        return Vector.ndarray(n, dtype, shape, **kwargs)
    raise AssertionError(f"unhandled Backend member: {backend!r}")


def _tensor_mat(n, m, dtype, shape, *, backend=Backend.FIELD, **kwargs):
    """Private impl backing ``qd.Matrix.tensor``.

    Dispatcher over ``qd.Matrix.field`` and ``qd.Matrix.ndarray`` selected
    by the ``backend=`` keyword. Not part of the public API — call
    ``qd.Matrix.tensor(...)`` instead.
    """
    backend = _coerce_backend(backend)
    # pylint: disable-next=import-outside-toplevel  # late import to break circular dependency
    from quadrants.lang.matrix import Matrix

    if backend is Backend.FIELD:
        return Matrix.field(n, m, dtype, shape, **kwargs)
    if backend is Backend.NDARRAY:
        return Matrix.ndarray(n, m, dtype, shape, **kwargs)
    raise AssertionError(f"unhandled Backend member: {backend!r}")
