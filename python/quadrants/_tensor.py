"""Tensors: per-tensor backend and layout.

This module is the user-facing entry point for selecting a tensor backend (``qd.field`` vs ``qd.ndarray``) and an
optional physical memory layout on a per-tensor basis.

See ``docs/source/user_guide/tensor.md`` for the user guide.

``qd.Tensor`` is the wrapper *class* defined in ``_tensor_wrapper.py``; it doubles as the polymorphic kernel-argument
annotation. The dispatch sites that previously keyed off the ``_TensorAnnotation`` singleton type now check
``annotation is Tensor`` (the class). See ``_func_base.py``, ``_template_mapper_hotpath.py`` and
``function_def_transformer.py``.
"""

# pylint: disable=import-outside-toplevel
# (Late imports below are intentional, to break circular import cycles between the tensor entry point and the
# lang/types subpackages.)

from enum import IntEnum

# Re-export so ``from quadrants._tensor import *`` still binds ``Tensor`` - keeps the wildcard import in
# ``__init__.py`` simple and atomic.
from quadrants._tensor_wrapper import Tensor
from quadrants._tensor_wrapper import wrap as _wrap_impl

__all__ = [
    "Backend",
    "Tensor",
    "tensor",
]

# Marker tuples prefixed onto cache keys to keep field-resolved and ndarray-resolved instantiations of the same
# qd.Tensor slot distinct.
_TENSOR_T_FIELD_MARKER = "__qd_tensor_t_field__"
_TENSOR_T_NDARRAY_MARKER = "__qd_tensor_t_ndarray__"

# ----------------------------------------------------------------------------
# Internal: attach layout metadata to an existing Ndarray.
#
# Public API for ndarray + non-identity layout lands in an earlier change (the qd.tensor(..., backend=NDARRAY,
# layout=...) path is currently gated by NotImplementedError). Until then, this private helper exists so the AST
# subscript-rewrite plumbing can be exercised end-to-end in tests without changing the user-facing factory signature.
# ----------------------------------------------------------------------------


def _with_layout(ndarray, layout):
    """Tag ``ndarray`` with a canonical-axis permutation. Internal.

    Accepts either a bare ``Ndarray`` or a ``Tensor`` wrapper around one; in the wrapper case the tag goes on the
    underlying impl so the kernel-arg unwrap hook (and the AST rewrite that gates on ``_qd_layout``) sees it.

    If a companion ``grad`` ndarray exists (allocated by ``needs_grad=True``), the tag is propagated to it so kernel
    code reading ``x.grad[...]`` goes through the same canonical->physical AST rewrite as ``x[...]``.
    """
    # Unwrap Tensor wrappers transparently. Imported lazily to dodge the _tensor_wrapper -> _tensor cycle.
    from quadrants._tensor_wrapper import (  # pylint: disable=reimported
        Tensor as _TensorWrapper,
    )

    if isinstance(ndarray, _TensorWrapper):
        ndarray = ndarray._unwrap()

    layout = tuple(layout)
    ndim = len(ndarray.shape)
    if len(layout) != ndim:
        raise ValueError(f"layout has {len(layout)} entries but ndarray has {ndim} dims")
    if sorted(layout) != list(range(ndim)):
        raise ValueError(f"layout={layout!r} is not a permutation of range({ndim})")
    ndarray._qd_layout = layout
    grad = getattr(ndarray, "grad", None)
    if grad is not None and getattr(grad, "_qd_layout", None) != layout:
        grad._qd_layout = layout
    return ndarray


class Backend(IntEnum):
    """Tensor storage backend.

    Each value selects one of Quadrants' two underlying tensor implementations:

    - :attr:`FIELD` (``qd.field``): faster at runtime; recompiles kernels whenever any dimension size changes. Best for
      tensors whose shape is effectively static across a run.
    - :attr:`NDARRAY` (``qd.ndarray``): slightly slower at runtime but avoids kernel recompilation when sizes change.
      Best for tensors whose shape varies frequently (e.g. dynamic batch sizes, growing buffers).

    The choice is made per tensor at allocation time. A single program can freely mix both backends.
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


# Kwargs explicitly accepted by the unified tensor factories (in addition to the positional ``dtype`` / ``shape`` /
# ``n`` / ``m``). The factories hard-validate against these sets so typos and backend-specific options don't silently
# work on one backend and raise cryptic errors deep in the other. Users who need backend-specific knobs (e.g.
# ``offset=`` for field offset indexing, ``order=`` for SoA layouts) should call ``qd.field`` / ``qd.ndarray`` directly
# - they have explicitly opted out of the unified tensor API.
#
# ``layout=`` is on the scalar ``qd.tensor`` factory only; the Vector/Matrix factories reject it because layout
# semantics over an extra element axis are out of scope for now.
_SCALAR_ACCEPTED_KWARGS = frozenset({"backend", "needs_grad", "layout"})
_VEC_MAT_ACCEPTED_KWARGS = frozenset({"backend", "needs_grad"})


def _validate_kwargs(kwargs, *, factory_name, accepted):
    if "order" in kwargs:
        if "layout" in accepted:
            raise TypeError(f"{factory_name}(...) does not accept order=; pass layout=(...) instead")
        raise TypeError(
            f"{factory_name}(...) does not accept order= (or layout=); "
            "use the underlying qd.Vector.field / qd.Matrix.field for non-default storage order"
        )
    extra = set(kwargs) - accepted
    if extra:
        accepted_str = ", ".join(sorted(accepted | {"dtype", "shape"}))
        raise TypeError(
            f"{factory_name}() got unexpected keyword argument(s) " f"{sorted(extra)!r}; accepted: {accepted_str}"
        )


def _layout_to_order(layout, ndim):
    """Validate ``layout`` and translate it to the ``order=`` string accepted by :func:`quadrants.field`.

    ``layout`` is a tuple of ``ndim`` ints - a permutation of ``range(ndim)`` - listing the *canonical* axis index at
    each successive memory-nesting level, outermost first. ``layout=(1, 0)`` for a 2-D tensor means axis 1 is the
    outer SNode, axis 0 is the inner one (i.e. transposed storage), which translates to ``order='ji'``.

    Returns ``None`` for the identity permutation, so the caller can omit ``order=`` entirely (matches the unsuffixed
    default).
    """
    if not isinstance(layout, tuple):
        layout = tuple(layout)
    if len(layout) != ndim:
        raise ValueError(f"layout has {len(layout)} entries but shape has {ndim} " f"dimensions; they must match")
    if sorted(layout) != list(range(ndim)):
        raise ValueError(f"layout={layout!r} is not a permutation of range({ndim})")
    if layout == tuple(range(ndim)):
        return None  # identity layout - no order= needed
    return "".join(chr(ord("i") + axis) for axis in layout)


def tensor(dtype, shape, *, backend=Backend.NDARRAY, layout=None, **kwargs):
    """Allocate a tensor on the chosen backend, optionally with a custom physical layout.

    Thin dispatcher over :func:`quadrants.field` and :func:`quadrants.ndarray` that selects between the two via the
    :class:`Backend` enum.

    Args:
        dtype: Element data type (e.g. ``qd.f32``, ``qd.i32``, or a compound type from ``qd.types``).
        shape: Shape of the tensor as an ``int`` or tuple of ``int``.
        backend (Backend, optional): Storage backend. Defaults to :attr:`Backend.NDARRAY`.
        layout (tuple of int, optional): Permutation of canonical axes describing the physical memory nesting order,
            outermost first. For a rank-N tensor, must be a permutation of ``range(N)``. ``None`` (default) and the
            identity permutation both mean "natural row-major-like layout" (no permutation is applied).

            ``shape`` is always interpreted as the **canonical** shape (the shape you index inside kernels). The
            underlying allocation is automatically sized to the permuted *physical* shape, and kernel subscripts
            ``x[i, j, ...]`` are rewritten to hit the right physical slot. Supported on both ``Backend.FIELD`` and
            ``Backend.NDARRAY``.

    Returns:
        A ``ScalarField`` when ``backend == Backend.FIELD``, or an ``Ndarray`` when ``backend == Backend.NDARRAY``. In
        both cases ``.shape`` reports the canonical shape; the physical layout is managed transparently.

    Example::

        >>> import quadrants as qd
        >>> qd.init(arch=qd.x64)
        >>> a = qd.tensor(qd.f32, shape=(4, 5))                       # default layout
        >>> b = qd.tensor(qd.f32, shape=(4, 5), layout=(1, 0))        # transposed storage
        >>> c = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY)
        >>> d = qd.tensor(qd.f32, shape=(4, 5), backend=qd.Backend.NDARRAY,
        ...               layout=(1, 0))                              # transposed ndarray

    Raises:
        ValueError: If ``backend`` is not a valid :class:`Backend` member, or if ``layout`` is not a permutation of
            ``range(len(shape))``.
        TypeError: If any keyword argument outside the accepted set is passed (see ``_SCALAR_ACCEPTED_KWARGS``).
    """
    from quadrants.lang._ndarray import (
        Ndarray,  # pylint: disable=import-outside-toplevel
    )
    from quadrants.lang.field import Field  # pylint: disable=import-outside-toplevel

    if isinstance(dtype, (Ndarray, Field, Tensor)):
        raise TypeError(
            f"qd.tensor() allocates a new tensor; to wrap an existing {type(dtype).__name__}, use qd.wrap(impl) instead"
        )
    if layout is not None:
        from quadrants.lang.matrix import (
            MatrixType,  # pylint: disable=import-outside-toplevel
        )

        if isinstance(dtype, MatrixType):
            raise TypeError(
                "layout= is not supported with compound dtypes (vector/matrix). "
                "Use qd.Vector.tensor(...) or qd.Matrix.tensor(...) without layout= instead."
            )
    _validate_kwargs(kwargs, factory_name="qd.tensor", accepted=_SCALAR_ACCEPTED_KWARGS)
    backend = _coerce_backend(backend)
    forwarded = {k: v for k, v in kwargs.items() if k != "backend"}
    # pylint: disable-next=import-outside-toplevel  # late import to break circular dependency
    from quadrants.lang import impl

    shape_t = (shape,) if isinstance(shape, int) else tuple(shape)
    order = _layout_to_order(layout, len(shape_t)) if layout is not None else None

    if backend is Backend.FIELD:
        if order is not None:
            forwarded["order"] = order
        f = impl.field(dtype, shape, **forwarded)
        # The canonical->physical layout permutation is attached by ``_field`` itself via ``_qd_layout`` (identical
        # attribute to the one ``Ndarray`` uses). The AST subscript rewrite in ``build_Subscript`` /
        # ``build_struct_for`` reads it to permute user-supplied canonical indices into physical storage order;
        # ``Field.layout`` reads the same attribute for introspection.
        return _wrap_impl(f)
    if backend is Backend.NDARRAY:
        if order is None:
            return _wrap_impl(impl.ndarray(dtype, shape, **forwarded))
        # Non-identity layout: allocate at the physical (permuted) shape and tag the result so the kernel-side
        # subscript rewrite picks up the canonical -> physical translation.
        assert layout is not None  # implied by `order is not None`
        layout_t = tuple(layout)
        physical_shape = tuple(shape_t[axis] for axis in layout_t)
        arr = impl.ndarray(dtype, physical_shape, **forwarded)
        # _with_layout propagates the tag to the companion grad ndarray if one was allocated by needs_grad=True, so
        # kernel code reading x.grad[i, j, ...] goes through the same canonical->physical AST rewrite as the primal
        # access.
        _with_layout(arr, layout_t)
        return _wrap_impl(arr)
    raise AssertionError(f"unhandled Backend member: {backend!r}")


def _tensor_vec(n, dtype, shape, *, backend=Backend.NDARRAY, **kwargs):
    """Private impl backing ``qd.Vector.tensor``.

    Dispatcher over ``qd.Vector.field`` and ``qd.Vector.ndarray`` selected by the ``backend=`` keyword. Not part of
    the public API - call ``qd.Vector.tensor(...)`` instead. Hard-validates kwargs against ``_VEC_MAT_ACCEPTED_KWARGS``
    (no ``layout=`` - layout semantics over an extra element axis are out of scope for now).
    """
    _validate_kwargs(kwargs, factory_name="qd.Vector.tensor", accepted=_VEC_MAT_ACCEPTED_KWARGS)
    backend = _coerce_backend(backend)
    forwarded = {k: v for k, v in kwargs.items() if k != "backend"}
    # pylint: disable-next=import-outside-toplevel  # late import to break circular dependency
    from quadrants._tensor_wrapper import VectorTensor
    from quadrants.lang.matrix import Vector

    if backend is Backend.FIELD:
        return VectorTensor(Vector.field(n, dtype, shape, **forwarded))
    if backend is Backend.NDARRAY:
        return VectorTensor(Vector.ndarray(n, dtype, shape, **forwarded))
    raise AssertionError(f"unhandled Backend member: {backend!r}")


def _tensor_mat(n, m, dtype, shape, *, backend=Backend.NDARRAY, **kwargs):
    """Private impl backing ``qd.Matrix.tensor``.

    Dispatcher over ``qd.Matrix.field`` and ``qd.Matrix.ndarray`` selected by the ``backend=`` keyword. Not part of
    the public API - call ``qd.Matrix.tensor(...)`` instead. Hard-validates kwargs against ``_VEC_MAT_ACCEPTED_KWARGS``
    (no ``layout=`` - layout semantics over an extra element axis are out of scope for now).
    """
    _validate_kwargs(kwargs, factory_name="qd.Matrix.tensor", accepted=_VEC_MAT_ACCEPTED_KWARGS)
    backend = _coerce_backend(backend)
    forwarded = {k: v for k, v in kwargs.items() if k != "backend"}
    # pylint: disable-next=import-outside-toplevel  # late import to break circular dependency
    from quadrants._tensor_wrapper import MatrixTensor
    from quadrants.lang.matrix import Matrix

    if backend is Backend.FIELD:
        return MatrixTensor(Matrix.field(n, m, dtype, shape, **forwarded))
    if backend is Backend.NDARRAY:
        return MatrixTensor(Matrix.ndarray(n, m, dtype, shape, **forwarded))
    raise AssertionError(f"unhandled Backend member: {backend!r}")
