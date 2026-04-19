"""Layout-aware tensor — Phase 1 + Phase 2.

A ``Tensor`` is a logical view of a Quadrants ``Ndarray`` plus a layout
permutation that describes how logical dimensions map to physical memory
dimensions.

Phase 1 (Python scope):
  - ``Tensor`` class wrapping ``(underlying_ndarray, shape, layout)``
  - ``tensor(dtype, shape, layout=None)`` factory; ``layout=None`` means identity
  - layout validation: must be a permutation of ``range(ndim)``
  - python-scope ``__getitem__`` / ``__setitem__`` translate logical -> physical
  - ``to_numpy`` / ``from_numpy`` operate on the logical shape

Phase 2 (kernel-arg binding):
  - ``Tensor`` is a ``@dataclasses.dataclass`` with two bridge-recognised fields:
      * ``underlying``: ``qd.types.ndarray(...)`` — the physical storage,
        bound at the kernel boundary like a regular ``qd.ndarray``.
      * ``layout``: ``qd.template`` — a Python tuple captured at trace time
        so kernels can branch on it via ``qd.static(t.layout == ...)``.
    This rides the existing dataclass kernel-arg bridge in
    ``quadrants.lang._kernel_impl_dataclass``: no new annotation surface or
    binding code is required.
  - Inside a kernel, users currently write ``t.underlying[physical_idx]``
    (verbose). Phase 3 (next) adds AST sugar so plain ``t[i, j]`` resolves to
    the permuted physical access.

See ``perso_hugh/doc/qd_tensor_design.md`` for the full design.
"""

import dataclasses
from typing import Sequence, Tuple

import numpy as np

from quadrants.lang import impl
from quadrants.types.annotations import template as _template
from quadrants.types.ndarray_type import NdarrayType as _NdarrayType

__all__ = ["Tensor", "tensor"]


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
# Tensor: dataclass-bridged layout-aware ndarray view.
# ---------------------------------------------------------------------------

# Field-type marker so the dataclass kernel-arg bridge sees the underlying as
# a regular qd.ndarray. ``ndim`` is left None here because we want to accept
# tensors of any rank as kernel args; per-call validation matches at runtime
# inside ``check_matched`` upstream.
_UNDERLYING_FIELD_TYPE = _NdarrayType()


@dataclasses.dataclass
class Tensor:
    """Layout-aware logical view of an ``Ndarray``.

    Constructed via :func:`tensor`. Inside ``@qd.kernel`` it behaves as a
    dataclass with two bridge-recognised fields:

    - ``underlying`` (``qd.types.ndarray()``) — the physical storage.
    - ``layout`` (``qd.template``) — the permutation tuple, available at
      trace time via ``qd.static(t.layout == (1, 0))`` etc.

    In Python scope, ``t[i, j]`` uses logical indices and is automatically
    translated to the physical position. Phase 3 adds the same logical-subscript
    sugar inside kernels.
    """

    # Bridge-bound fields. Type annotations matter to the kernel-arg dispatcher.
    underlying: _UNDERLYING_FIELD_TYPE  # type: ignore[valid-type]
    layout: _template

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

    def __repr__(self):
        return (
            f"<qd.Tensor shape={self.shape} layout={self.layout} "
            f"physical_shape={self.physical_shape} dtype={self.dtype}>"
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


def tensor(dtype, shape, layout=None) -> Tensor:
    """Create a layout-aware :class:`Tensor`.

    Args:
        dtype: element data type, e.g. ``qd.f32``, ``qd.i32``.
        shape: logical shape (tuple of positive ints, or a single int for 1D).
        layout: optional permutation of ``range(ndim)``. ``layout[i] = p`` means
            logical dim ``i`` is stored along physical axis ``p``. ``None``
            (default) means identity. For 2D, ``(1, 0)`` is the canonical
            "transposed in memory" choice.

    Returns:
        A :class:`Tensor` wrapping a freshly allocated ``Ndarray`` of the
        appropriate physical shape.
    """
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(d) for d in shape)
    ndim = len(shape)
    if layout is None:
        layout = tuple(range(ndim))
    layout = _validate_layout(layout, ndim)
    physical_shape = _logical_to_physical_shape(shape, layout)
    underlying = impl.ndarray(dtype, physical_shape)
    return Tensor(underlying=underlying, layout=layout)
