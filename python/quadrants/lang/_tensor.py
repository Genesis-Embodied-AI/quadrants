"""Layout-aware tensor — Phases 1, 1b, 2.

A ``Tensor`` is a logical view of a Quadrants storage object plus a layout
permutation that describes how logical dimensions map to physical memory
dimensions.

Storage backend is selectable at construction via :class:`Backend`:

- :attr:`Backend.NDARRAY` (default) wraps a :class:`~quadrants.lang.Ndarray`.
- :attr:`Backend.FIELD` wraps a Quadrants :func:`~quadrants.lang.impl.field`
  (SNode-backed storage).

Both share a single python-scope API (``shape``, ``ndim``, ``dtype``,
``__getitem__`` / ``__setitem__`` with logical indices, ``to_numpy`` /
``from_numpy``, ``fill``).

Phase 2 (kernel-arg binding):
  Each variant is a ``@dataclasses.dataclass`` with two bridge-recognised
  fields:

  - ``underlying`` — annotation differs per backend so the dataclass
    kernel-arg bridge picks the right binding path:
      * :class:`NdarrayTensor` uses ``qd.types.ndarray()``: bridged as a
        regular ndarray.
      * :class:`FieldTensor` uses ``qd.template``: fields are runtime
        singletons captured by-reference at trace time, just like a bare
        ``qd.field`` kernel arg.
  - ``layout`` — ``qd.template``, captured as a Python tuple at trace time,
    available for ``qd.static(t.layout == (1, 0))`` branching.

  Both ride the existing dataclass bridge in
  ``quadrants.lang._kernel_impl_dataclass`` — no new annotation surface or
  binding code is required, and the backend distinction lands automatically
  in the fastcache key via the ``underlying`` field's annotation type.

Inside a kernel users currently write ``t.underlying[physical_idx]`` (verbose).
Phase 3 (deferred) adds AST sugar so plain ``t[i, j]`` resolves to the
permuted physical access.

The user-facing factory :func:`tensor` dispatches on a :class:`Backend` enum::

    qd.tensor(qd.f32, shape=(N, B))                              # ndarray
    qd.tensor(qd.f32, shape=(N, B), layout=(1, 0))               # transposed ndarray
    qd.tensor(qd.f32, shape=(N, B), backend=qd.Backend.FIELD)    # field

For backward compatibility, ``Tensor`` is an alias for :class:`NdarrayTensor`,
the default backend variant. Kernels that need to accept a field-backed tensor
must annotate the parameter with :class:`FieldTensor` explicitly, since the
dataclass bridge dispatches on the parameter annotation.

See ``perso_hugh/doc/qd_tensor_design.md`` for the full design.
"""

import dataclasses
import enum
from typing import Sequence, Tuple

import numpy as np

from quadrants.lang import impl
from quadrants.types.annotations import template as _template
from quadrants.types.ndarray_type import NdarrayType as _NdarrayType

__all__ = ["Backend", "Tensor", "NdarrayTensor", "FieldTensor", "tensor"]


class Backend(enum.Enum):
    """Storage backend for :func:`tensor`."""

    NDARRAY = "ndarray"
    FIELD = "field"


def _validate_layout(layout: Sequence[int], ndim: int) -> Tuple[int, ...]:
    """Validate that ``layout`` is a permutation of ``range(ndim)`` and return it as a tuple.

    Convention: ``layout[i] = physical_axis_for_logical_dim_i``. So
    ``layout=(0, 1, ..., n-1)`` is identity and ``layout=(1, 0)`` is a 2D transpose.
    """
    layout_t = tuple(int(p) for p in layout)
    if len(layout_t) != ndim:
        raise ValueError(f"layout must have length {ndim} (one entry per dim), got length {len(layout_t)}: {layout_t}")
    if sorted(layout_t) != list(range(ndim)):
        raise ValueError(
            f"layout must be a permutation of range({ndim}), got {layout_t}. "
            f"Each physical axis index must appear exactly once."
        )
    return layout_t


def _logical_to_physical_shape(logical_shape: Sequence[int], layout: Sequence[int]) -> Tuple[int, ...]:
    """``physical_shape[layout[i]] = logical_shape[i]``."""
    ndim = len(logical_shape)
    physical = [0] * ndim
    for i, p in enumerate(layout):
        physical[p] = int(logical_shape[i])
    return tuple(physical)


def _logical_to_physical_indices(idx: Sequence[int], layout: Sequence[int]) -> Tuple[int, ...]:
    """Translate a tuple of logical indices to physical indices via ``layout``."""
    physical = [0] * len(idx)
    for i, p in enumerate(layout):
        physical[p] = idx[i]
    return tuple(physical)


# ---------------------------------------------------------------------------
# _TensorBase: shared logical/physical-index logic, backend-agnostic.
# ---------------------------------------------------------------------------


class _TensorBase:
    """Shared layout-aware behaviour for :class:`NdarrayTensor` and :class:`FieldTensor`.

    Subclasses must be ``@dataclasses.dataclass`` with two fields:

    - ``underlying`` (backend-specific annotation; has ``.shape``, ``.dtype``,
      subscript, ``fill``, ``to_numpy``, ``from_numpy``)
    - ``layout`` (``qd.template``; tuple set in ``__post_init__``)
    """

    # Declared here so type checkers see them on the base; concrete dataclass
    # subclasses override with backend-specific annotations.
    underlying: object
    layout: tuple

    def __post_init__(self):
        layout_t = _validate_layout(self.layout, len(self.underlying.shape))
        # Re-assign as a tuple so equality and hashing work uniformly.
        object.__setattr__(self, "layout", layout_t)
        physical_shape = tuple(int(d) for d in self.underlying.shape)
        # Logical shape = inverse permutation applied to physical_shape:
        # logical_shape[i] = physical_shape[layout[i]].
        logical_shape = tuple(physical_shape[p] for p in layout_t)
        object.__setattr__(self, "_logical_shape", logical_shape)
        object.__setattr__(self, "_physical_shape", physical_shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Logical shape (the user-facing shape)."""
        # _logical_shape is set in __post_init__ via object.__setattr__.
        return self._logical_shape  # pylint: disable=no-member

    @property
    def physical_shape(self) -> Tuple[int, ...]:
        """Physical (in-memory) shape, after applying ``layout``."""
        return self._physical_shape  # pylint: disable=no-member

    @property
    def ndim(self) -> int:
        return len(self._logical_shape)  # pylint: disable=no-member

    @property
    def dtype(self):
        return self.underlying.dtype

    @property
    def backend(self) -> Backend:
        # Concrete subclass sets this as a class attribute.
        return self._backend  # pylint: disable=no-member

    def __repr__(self):
        return (
            f"<qd.{type(self).__name__} shape={self.shape} layout={self.layout} "
            f"physical_shape={self.physical_shape} dtype={self.dtype} backend={self.backend.value}>"
        )

    def _translate_key(self, key) -> Tuple[int, ...]:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) != self.ndim:
            raise IndexError(f"{self.ndim}d tensor indexed with {len(key)}d key {key}")
        for axis, k in enumerate(key):
            if not isinstance(k, (int, np.integer)):
                raise TypeError(
                    f"Tensor python-scope subscript only supports integer indices for now; "
                    f"got {type(k).__name__} on logical axis {axis}. "
                    f"Slicing is not yet implemented (see Phase 3 for kernel-scope support)."
                )
        return _logical_to_physical_indices(key, self.layout)

    def __getitem__(self, key):
        physical_key = self._translate_key(key)
        return self.underlying[physical_key]

    def __setitem__(self, key, value):
        physical_key = self._translate_key(key)
        self.underlying[physical_key] = value

    def fill(self, value):
        """Fill the underlying storage with a scalar value (layout-independent)."""
        self.underlying.fill(value)

    def _inverse_layout(self) -> Tuple[int, ...]:
        """Inverse permutation of ``layout``: ``inverse[p] = i where layout[i] = p``."""
        inverse = [0] * self.ndim
        for i, p in enumerate(self.layout):
            inverse[p] = i
        return tuple(inverse)

    def to_numpy(self) -> np.ndarray:
        """Return the tensor data as a numpy array in *logical* shape.

        ``logical[i] = physical[layout[i]]`` -> ``transpose(layout)``.
        """
        physical_np = self.underlying.to_numpy()
        if self.layout == tuple(range(self.ndim)):
            return physical_np
        return physical_np.transpose(self.layout)

    def from_numpy(self, arr: np.ndarray) -> None:
        """Load data from a logical-shape numpy array.

        Inverse of :meth:`to_numpy`: ``physical = arr.transpose(inverse_layout)``.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"expected numpy.ndarray, got {type(arr).__name__}")
        if tuple(arr.shape) != self.shape:
            raise ValueError(f"from_numpy expects logical shape {self.shape}, got {tuple(arr.shape)}")
        if self.layout == tuple(range(self.ndim)):
            self.underlying.from_numpy(arr)
        else:
            physical = np.ascontiguousarray(arr.transpose(self._inverse_layout()))
            self.underlying.from_numpy(physical)


# ---------------------------------------------------------------------------
# Concrete dataclass variants: one per storage backend.
# ---------------------------------------------------------------------------

# Field-type marker so the dataclass kernel-arg bridge sees the underlying as
# a regular qd.ndarray. ``ndim`` is left None here because we want to accept
# tensors of any rank as kernel args; per-call validation matches at runtime
# inside ``check_matched`` upstream.
_UNDERLYING_NDARRAY_TYPE = _NdarrayType()


@dataclasses.dataclass(repr=False)
class NdarrayTensor(_TensorBase):
    """Layout-aware logical view of an ``Ndarray`` (default backend).

    Constructed via :func:`tensor` with ``backend=Backend.NDARRAY`` (default).
    Kernel-arg binding goes through the standard ndarray path, so passing a
    :class:`NdarrayTensor` to a kernel costs no extra copies.
    """

    # Bridge-bound fields. Annotations matter to the kernel-arg dispatcher.
    underlying: _UNDERLYING_NDARRAY_TYPE  # type: ignore[valid-type]
    layout: _template

    _backend = Backend.NDARRAY


@dataclasses.dataclass(repr=False)
class FieldTensor(_TensorBase):
    """Layout-aware logical view of a :func:`~quadrants.lang.impl.field`.

    Constructed via :func:`tensor` with ``backend=Backend.FIELD``. Fields are
    SNode-backed and bound to kernels by-reference via the ``qd.template``
    path (the same way bare fields are bound). Useful when the underlying
    storage benefits from SNode features (sparse layouts, hierarchical
    placement, ``order=`` overrides).

    Kernel signatures must annotate parameters as :class:`FieldTensor`
    explicitly (not :class:`Tensor` / :class:`NdarrayTensor`) so the
    dataclass bridge picks the template binding path for ``underlying``.
    """

    # ``qd.template`` because qd.field is a runtime singleton, not a transient
    # ndarray buffer.
    underlying: _template
    layout: _template

    _backend = Backend.FIELD


# Backwards-compatible alias: ``qd.Tensor`` resolves to the default ndarray
# variant. New code that needs a field-backed kernel arg must use
# :class:`FieldTensor` explicitly.
Tensor = NdarrayTensor


# ---------------------------------------------------------------------------
# User-facing factory.
# ---------------------------------------------------------------------------


def _coerce_backend(backend) -> Backend:
    if isinstance(backend, Backend):
        return backend
    if isinstance(backend, str):
        # Tolerate string literals for ergonomics, but the canonical form is
        # the enum value.
        try:
            return Backend(backend)
        except ValueError as exc:
            valid = ", ".join(repr(b.value) for b in Backend)
            raise ValueError(f"unknown backend {backend!r}; expected one of {{{valid}}} or a Backend enum") from exc
    raise TypeError(f"backend must be a Backend enum (got {type(backend).__name__}: {backend!r})")


def tensor(dtype, shape, layout=None, backend: Backend = Backend.NDARRAY) -> _TensorBase:
    """Create a layout-aware tensor.

    Args:
        dtype: element data type, e.g. ``qd.f32``, ``qd.i32``.
        shape: logical shape (tuple of positive ints, or a single int for 1D).
        layout: optional permutation of ``range(ndim)``. ``layout[i] = p`` means
            logical dim ``i`` is stored along physical axis ``p``. ``None``
            (default) means identity. For 2D, ``(1, 0)`` is the canonical
            "transposed in memory" choice.
        backend: storage backend, a :class:`Backend` enum value. Defaults to
            :attr:`Backend.NDARRAY`. :attr:`Backend.FIELD` allocates the
            storage as a Quadrants field instead.

    Returns:
        A :class:`NdarrayTensor` or :class:`FieldTensor` wrapping freshly
        allocated storage of the appropriate physical shape.
    """
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(d) for d in shape)
    ndim = len(shape)
    if layout is None:
        layout = tuple(range(ndim))
    layout = _validate_layout(layout, ndim)
    physical_shape = _logical_to_physical_shape(shape, layout)

    backend = _coerce_backend(backend)
    if backend is Backend.NDARRAY:
        underlying = impl.ndarray(dtype, physical_shape)
        return NdarrayTensor(underlying=underlying, layout=layout)
    if backend is Backend.FIELD:
        underlying = impl.field(dtype, shape=physical_shape)
        return FieldTensor(underlying=underlying, layout=layout)
    # Should be unreachable thanks to _coerce_backend.
    raise ValueError(f"unhandled backend: {backend!r}")
