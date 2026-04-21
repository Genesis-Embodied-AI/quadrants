"""Tensors: per-tensor backend and (later) layout.

This module is the user-facing entry point for selecting a tensor backend
(``qd.field`` vs ``qd.ndarray``) on a per-tensor basis.

See ``docs/source/user_guide/tensor.md`` for the user guide.
"""

from enum import IntEnum

__all__ = ["Backend", "tensor", "tensor_annotation", "tensor_mat", "tensor_vec"]


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
    # late import to break circular dependency
    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

    if backend is Backend.FIELD:
        return impl.field(dtype, shape, **kwargs)
    if backend is Backend.NDARRAY:
        return impl.ndarray(dtype, shape, **kwargs)
    raise AssertionError(f"unhandled Backend member: {backend!r}")


def tensor_vec(n, dtype, shape, *, backend=Backend.FIELD, **kwargs):
    """Allocate a tensor whose elements are length-``n`` vectors.

    Dispatcher over ``qd.Vector.field`` and ``qd.Vector.ndarray`` selected by
    the ``backend=`` keyword.

    Args:
        n (int): Length of each vector element.
        dtype: Element data type (e.g. ``qd.f32``).
        shape: Shape of the tensor (excluding the vector dimension) as an
            ``int`` or tuple of ``int``.
        backend (Backend, optional): Storage backend. Defaults to
            :attr:`Backend.FIELD`.
        **kwargs: Forwarded verbatim to the underlying ``qd.Vector.field`` /
            ``qd.Vector.ndarray`` call. ``qd.Vector.ndarray`` does not accept
            extra keyword arguments today.

    Example::

        >>> import quadrants as qd
        >>> qd.init(arch=qd.x64)
        >>> v = qd.tensor_vec(3, qd.f32, shape=(4,))
        >>> u = qd.tensor_vec(3, qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)
    """
    backend = _coerce_backend(backend)
    # late import
    from quadrants.lang.matrix import Vector  # pylint: disable=import-outside-toplevel

    if backend is Backend.FIELD:
        return Vector.field(n, dtype, shape, **kwargs)
    if backend is Backend.NDARRAY:
        return Vector.ndarray(n, dtype, shape, **kwargs)
    raise AssertionError(f"unhandled Backend member: {backend!r}")


def tensor_mat(n, m, dtype, shape, *, backend=Backend.FIELD, **kwargs):
    """Allocate a tensor whose elements are ``n``-by-``m`` matrices.

    Dispatcher over ``qd.Matrix.field`` and ``qd.Matrix.ndarray`` selected by
    the ``backend=`` keyword.

    Args:
        n (int): Number of rows of each matrix element.
        m (int): Number of columns of each matrix element.
        dtype: Element data type (e.g. ``qd.f32``).
        shape: Shape of the tensor (excluding the matrix dimensions) as an
            ``int`` or tuple of ``int``.
        backend (Backend, optional): Storage backend. Defaults to
            :attr:`Backend.FIELD`.
        **kwargs: Forwarded verbatim to the underlying ``qd.Matrix.field`` /
            ``qd.Matrix.ndarray`` call. ``qd.Matrix.ndarray`` does not accept
            extra keyword arguments today.

    Example::

        >>> import quadrants as qd
        >>> qd.init(arch=qd.x64)
        >>> a = qd.tensor_mat(2, 3, qd.f32, shape=(4,))
        >>> b = qd.tensor_mat(2, 3, qd.f32, shape=(4,), backend=qd.Backend.NDARRAY)
    """
    backend = _coerce_backend(backend)
    # late import
    from quadrants.lang.matrix import Matrix  # pylint: disable=import-outside-toplevel

    if backend is Backend.FIELD:
        return Matrix.field(n, m, dtype, shape, **kwargs)
    if backend is Backend.NDARRAY:
        return Matrix.ndarray(n, m, dtype, shape, **kwargs)
    raise AssertionError(f"unhandled Backend member: {backend!r}")


def tensor_annotation(backend):
    """Return the kernel-argument annotation appropriate for ``backend``.

    Mirrors the Genesis ``V_ANNOTATION = qd.types.ndarray() if use_ndarray
    else qd.template`` pattern as a single first-class call. Use it once, at
    module load time, to build a uniform annotation that you then attach to
    every tensor kernel argument:

    .. code-block:: python

        V_ANNOTATION = qd.tensor_annotation(qd.Backend.FIELD)

        @qd.kernel
        def fill(x: V_ANNOTATION):
            for i in qd.ndrange(x.shape[0]):
                x[i] = 1.0

    Args:
        backend (Backend): The backend whose tensors will be passed to
            kernels annotated with the returned object.

    Returns:
        An object suitable for use as a kernel-argument type annotation:

        - For ``Backend.FIELD``: an instance of ``qd.template()``.
        - For ``Backend.NDARRAY``: an instance of ``qd.types.ndarray()``.

        Both forms are interchangeable with their direct equivalents — the
        helper just hides the conditional behind one call.
    """
    backend = _coerce_backend(backend)
    from quadrants import types as _types  # pylint: disable=import-outside-toplevel

    if backend is Backend.FIELD:
        return _types.template()
    if backend is Backend.NDARRAY:
        return _types.ndarray()
    raise AssertionError(f"unhandled Backend member: {backend!r}")
