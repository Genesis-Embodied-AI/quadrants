"""Tensors: per-tensor backend and (later) layout.

This module is the user-facing entry point for selecting a tensor backend
(``qd.field`` vs ``qd.ndarray``) on a per-tensor basis.

See ``docs/source/user_guide/tensor.md`` for the user guide.
"""

# pylint: disable=import-outside-toplevel
# (Late imports below are intentional, to break circular import cycles
# between the tensor entry point and the lang/types subpackages.)

from enum import IntEnum

__all__ = ["Backend", "tensor", "tensor_annotation", "tensor_mat", "tensor_vec"]

# ----------------------------------------------------------------------------
# Internal: attach layout metadata to an existing Ndarray.
#
# Public API for ndarray + non-identity layout lands in an earlier change (the
# qd.tensor(..., backend=NDARRAY, layout=...) path is currently gated by
# NotImplementedError). Until then, this private helper exists so the AST
# subscript-rewrite plumbing can be exercised end-to-end in tests without
# changing the user-facing factory signature.
# ----------------------------------------------------------------------------


def _with_layout(ndarray, layout):
    """Tag ``ndarray`` with a canonical-axis permutation. Internal."""
    layout = tuple(layout)
    ndim = len(ndarray.shape)
    if len(layout) != ndim:
        raise ValueError(f"layout has {len(layout)} entries but ndarray has {ndim} dims")
    if sorted(layout) != list(range(ndim)):
        raise ValueError(f"layout={layout!r} is not a permutation of range({ndim})")
    ndarray._qd_layout = layout
    return ndarray


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


def _layout_to_order(layout, ndim):
    """Validate ``layout`` and translate it to the ``order=`` string accepted
    by :func:`quadrants.field`.

    ``layout`` is a tuple of ``ndim`` ints — a permutation of ``range(ndim)``
    — listing the *canonical* axis index at each successive memory-nesting
    level, outermost first. ``layout=(1, 0)`` for a 2-D tensor means axis 1
    is the outer SNode, axis 0 is the inner one (i.e. transposed storage),
    which translates to ``order='ji'``.

    Returns ``None`` for the identity permutation, so the caller can omit
    ``order=`` entirely (matches the unsuffixed default).
    """
    if not isinstance(layout, tuple):
        layout = tuple(layout)
    if len(layout) != ndim:
        raise ValueError(f"layout has {len(layout)} entries but shape has {ndim} " f"dimensions; they must match")
    if sorted(layout) != list(range(ndim)):
        raise ValueError(f"layout={layout!r} is not a permutation of range({ndim})")
    if layout == tuple(range(ndim)):
        return None  # identity layout — no order= needed
    return "".join(chr(ord("i") + axis) for axis in layout)


def tensor(dtype, shape, *, backend=Backend.FIELD, layout=None, **kwargs):
    """Allocate a tensor on the chosen backend, optionally with a custom
    physical layout.

    Thin dispatcher over :func:`quadrants.field` and :func:`quadrants.ndarray`
    that selects between the two via the :class:`Backend` enum.

    Args:
        dtype: Element data type (e.g. ``qd.f32``, ``qd.i32``, or a compound
            type from ``qd.types``).
        shape: Shape of the tensor as an ``int`` or tuple of ``int``.
        backend (Backend, optional): Storage backend. Defaults to
            :attr:`Backend.FIELD`.
        layout (tuple of int, optional): Permutation of canonical axes
            describing the physical memory nesting order, outermost first.
            For a rank-N tensor, must be a permutation of ``range(N)``.
            ``None`` (default) and the identity permutation both mean
            "natural row-major-like layout" (no ``order=`` is forwarded).

            **Currently only supported for ``Backend.FIELD``.** Passing a
            non-identity ``layout`` together with ``Backend.NDARRAY``
            raises :class:`NotImplementedError`. Ndarray-side support
            lands in a later release.
        **kwargs: Forwarded verbatim to the underlying ``qd.field`` or
            ``qd.ndarray`` call. Cannot include ``order=`` — use ``layout=``
            instead.

    Returns:
        A ``ScalarField`` when ``backend == Backend.FIELD``, or an
        ``Ndarray`` when ``backend == Backend.NDARRAY``.

    Example::

        >>> import quadrants as qd
        >>> qd.init(arch=qd.x64)
        >>> a = qd.tensor(qd.f32, shape=(4, 5))                       # default layout
        >>> b = qd.tensor(qd.f32, shape=(4, 5), layout=(1, 0))        # transposed storage
        >>> c = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY)

    Raises:
        ValueError: If ``backend`` is not a valid :class:`Backend` member,
            or if ``layout`` is not a permutation of ``range(len(shape))``.
        NotImplementedError: If a non-identity ``layout`` is requested
            together with ``Backend.NDARRAY``.
    """
    backend = _coerce_backend(backend)
    from quadrants.lang import impl  # late import to break circular dependency

    if "order" in kwargs:
        raise TypeError("qd.tensor(...) does not accept order=; pass layout=(...) instead")

    shape_t = (shape,) if isinstance(shape, int) else tuple(shape)
    order = _layout_to_order(layout, len(shape_t)) if layout is not None else None

    if backend is Backend.FIELD:
        if order is not None:
            kwargs["order"] = order
        return impl.field(dtype, shape, **kwargs)
    if backend is Backend.NDARRAY:
        if order is not None:
            raise NotImplementedError(
                "qd.tensor(..., backend=Backend.NDARRAY, layout=...) with a "
                "non-identity layout is not yet supported. Identity layouts "
                "(layout=None or layout=range(ndim)) work; non-identity "
                "ndarray layout lands in a later release."
            )
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
    from quadrants.lang.matrix import Vector  # late import

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
    from quadrants.lang.matrix import Matrix  # late import

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
    from quadrants import types as _types  # late import
    from quadrants.types.annotations import template  # late import

    if backend is Backend.FIELD:
        return template()
    if backend is Backend.NDARRAY:
        return _types.ndarray()
    raise AssertionError(f"unhandled Backend member: {backend!r}")
