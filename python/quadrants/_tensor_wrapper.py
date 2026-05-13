"""``qd.Tensor``: backend-agnostic tensor wrapper.

A thin Python wrapper around an underlying ``Ndarray`` or ``ScalarField`` impl. Makes backend symmetry a *type*
property rather than something we police test by test: the wrapper exposes a fixed whitelisted surface uniformly,
regardless of which impl it contains.

Promoted to ``qd.Tensor`` in ``hp/tensor-stork-19``: the class is now the public name, doubles as the polymorphic
kernel-arg annotation (``def f(x: qd.Tensor): ...``), and is what ``qd.tensor()``, ``qd.Vector.tensor()``,
``qd.Matrix.tensor()`` return. Bare impls are still reachable via ``qd.field``, ``qd.ndarray``, ``qd.Vector.field`` etc.

Surface:

- Introspection: ``shape``, ``dtype``, ``layout``, ``_unwrap()``.
- Layout-aware host-side ``__getitem__`` / ``__setitem__`` - permutes the canonical user key to the physical slot on
  layout-tagged ndarrays. Fixes gotcha B from the design doc (§8.11).
- Symmetric pickle via ``__reduce__`` - round-trips through ``to_numpy()`` so it works uniformly on both backends
  (Field, which never supported pickle upstream, is picklable through the wrapper).
- Forwards for ``to_numpy`` / ``from_numpy`` / ``to_torch`` / ``from_torch`` / ``to_dlpack`` / ``fill`` /
  ``copy_from`` - already layout-aware on both backends after stork-15/16, the wrapper delegates.
- Lazy-wrapped ``.grad`` - returns a ``Tensor`` wrapping ``impl.grad`` (identity-stable via
  ``functools.cached_property``).
- ``VectorTensor`` / ``MatrixTensor`` subclasses carrying ``element_shape``.

Out of scope:
- Genesis migration (stork-20): rewrite Genesis ``isinstance`` sites from ``(qd.Field, qd.Ndarray)`` to
  ``qd.Tensor``, switch its tensor allocations to ``qd.tensor()``.

See ``perso_hugh/doc/quadrants-tensor.md`` §8.11 / §8.12.
"""

# pylint: disable=import-outside-toplevel
# (Late imports throughout are intentional, to break circular import cycles between the tensor wrapper and the
# lang/types subpackages.)
from __future__ import annotations

import typing
from functools import cached_property

__all__ = [
    "Tensor",
    "VectorTensor",
    "MatrixTensor",
    "wrap",
]


# PERF-CRITICAL: This flag is checked on every kernel arg in _template_mapper_hotpath._extract_arg,
# kernel.Kernel.__call__, _func_base._inject_template_globals, and args_hasher.stringify_obj_type.
# It gates the isinstance(arg, Tensor) unwrap so that programs which never construct a qd.Tensor pay zero Python
# overhead for the check. Removing this flag or the guards that read it causes a measurable ~4% CPU regression on
# Genesis benchmarks (see regression_2026apr23_stork_log.md).
_any_tensor_constructed = False


def _is_identity(layout: typing.Optional[typing.Tuple[int, ...]]) -> bool:
    if layout is None:
        return True
    return tuple(layout) == tuple(range(len(layout)))


class Tensor:
    """Backend-agnostic tensor wrapper. The public ``qd.Tensor`` class.

    Holds a reference to an underlying impl (``Ndarray`` or ``Field``) and forwards a whitelisted surface. Layout-aware
    host-side indexing lives here: the AST-level canonical->physical rewrite only fires inside ``@qd.kernel`` bodies,
    so ``t[i, j]`` at host scope on a layout-tagged ndarray would otherwise hit the physical slot.

    Construct via ``qd.tensor(...)`` (or its ``qd.Vector.tensor`` / ``qd.Matrix.tensor`` siblings). The wrapper rejects
    double-wrapping: the impl must be a bare ``Ndarray`` or ``Field``.

    Doubles as a kernel parameter annotation: ``def k(x: qd.Tensor)`` accepts either a Field or an Ndarray; dispatch
    happens at extract time. See ``_template_mapper_hotpath._extract_arg``.
    """

    # ``cached_property`` requires ``__dict__``, so no ``__slots__``.

    def __init__(self, impl: typing.Any) -> None:
        from quadrants.lang._ndarray import Ndarray
        from quadrants.lang.field import Field

        if not isinstance(impl, (Ndarray, Field)):
            raise TypeError(f"Tensor(impl) requires an Ndarray or Field; got {type(impl).__name__}")
        self._impl: typing.Any = impl
        global _any_tensor_constructed  # noqa: PLW0603
        _any_tensor_constructed = True  # see comment on the flag definition above

    # ------------------------------------------------------------------
    # Identity / debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        layout = self.layout
        layout_repr = "" if layout is None else f", layout={layout!r}"
        return f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, " f"backend={self._backend_name()}{layout_repr})"

    def _backend_name(self) -> str:
        from quadrants.lang._ndarray import Ndarray

        return "NDARRAY" if isinstance(self._impl, Ndarray) else "FIELD"

    def _backend_enum(self) -> typing.Any:
        from quadrants._tensor import Backend
        from quadrants.lang._ndarray import Ndarray

        return Backend.NDARRAY if isinstance(self._impl, Ndarray) else Backend.FIELD

    # ------------------------------------------------------------------
    # Whitelisted introspection
    # ------------------------------------------------------------------

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        return tuple(self._impl.shape)

    @property
    def dtype(self) -> typing.Any:
        return self._impl.dtype

    @property
    def layout(self) -> typing.Optional[typing.Tuple[int, ...]]:
        # Forwards to the impl's ``layout`` property (symmetric across backends after stork-16).
        return self._impl.layout

    # ------------------------------------------------------------------
    # Internal escape hatch
    # ------------------------------------------------------------------

    def _unwrap(self) -> typing.Any:
        """Return the underlying impl. Used by the kernel-arg unwrap hook in ``Kernel.__call__`` so the JIT cache keys
        off impl identity.
        """
        return self._impl

    # ------------------------------------------------------------------
    # Layout-aware host indexing (fixes gotcha B)
    # ------------------------------------------------------------------

    def _host_physical_layout(self) -> typing.Optional[typing.Tuple[int, ...]]:
        """Return the permutation to apply to canonical host keys.

        Only ``Ndarray`` needs it: its Python-scope ``__getitem__`` / ``__setitem__`` pass the key directly to the
        host accessor, and the canonical->physical rewrite only fires inside ``@qd.kernel``.

        ``Field`` already translates canonical indices via the SNode hierarchy (``order=``) on every host access, so we
        return ``None`` and fall through to a plain delegation.
        """
        from quadrants.lang._ndarray import Ndarray

        if not isinstance(self._impl, Ndarray):
            return None
        layout = getattr(self._impl, "_qd_layout", None)
        if _is_identity(layout):
            return None
        assert layout is not None
        return tuple(layout)

    @staticmethod
    def _permute_key(key: typing.Any, layout: typing.Tuple[int, ...]) -> typing.Tuple[int, ...]:
        """Translate a user-supplied canonical key to physical coords.

        ``physical[p] = canonical[layout[p]]`` by the convention that ``layout[p]`` is the canonical axis at physical
        nesting level ``p`` (outermost first). Only full-rank keys are supported at host scope; partial / slice
        indexing is out of scope for the wrapper and would fall out of ``Ndarray``'s own API anyway.
        """
        if isinstance(key, int):
            # Rank-1 only; for rank>1 we require a tuple/list.
            if len(layout) != 1:
                raise TypeError(f"layout-tagged Tensor requires a full tuple key; got int for rank {len(layout)}")
            return (key,)
        key_t = tuple(key)
        if len(key_t) != len(layout):
            raise TypeError(f"layout-tagged Tensor key has {len(key_t)} entries but rank is {len(layout)}")
        return tuple(key_t[layout[p]] for p in range(len(layout)))

    def __getitem__(self, key: typing.Any) -> typing.Any:
        layout = self._host_physical_layout()
        if layout is None:
            return self._impl[key]
        return self._impl[self._permute_key(key, layout)]

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        layout = self._host_physical_layout()
        if layout is None:
            self._impl[key] = value
            return
        self._impl[self._permute_key(key, layout)] = value

    # ------------------------------------------------------------------
    # Interop forwards (layout-aware on both impls already)
    # ------------------------------------------------------------------

    def to_numpy(self, dtype: typing.Any = None, *, copy: typing.Optional[bool] = True) -> typing.Any:
        kw: dict[str, typing.Any] = {"copy": copy}
        if dtype is not None:
            kw["dtype"] = dtype
        return self._impl.to_numpy(**kw)

    def from_numpy(self, arr: typing.Any) -> None:
        self._impl.from_numpy(arr)

    def to_torch(self, device: typing.Any = None, *, copy: typing.Optional[bool] = True) -> typing.Any:
        kw: dict[str, typing.Any] = {"copy": copy}
        if device is not None:
            kw["device"] = device
        return self._impl.to_torch(**kw)

    def from_torch(self, arr: typing.Any) -> None:
        self._impl.from_torch(arr)

    def to_dlpack(self, versioned: bool = False) -> typing.Any:
        return self._impl.to_dlpack(versioned=versioned)

    def fill(self, value: typing.Any) -> None:
        self._impl.fill(value)

    def copy_from(self, other: typing.Any) -> None:
        # Accept either another ``Tensor`` or a bare impl (convenience: lets Genesis-style code pass raw fields
        # during the migration).
        if isinstance(other, Tensor):
            other = other._impl
        self._impl.copy_from(other)

    # ------------------------------------------------------------------
    # Gradient: lazy wrap so ``t.grad`` is identity-stable.
    # ------------------------------------------------------------------

    @cached_property
    def grad(self) -> typing.Optional["Tensor"]:
        g = getattr(self._impl, "grad", None)
        if g is None:
            return None
        return wrap(g)

    def has_grad(self) -> bool:
        """Whether this tensor's adjoint storage is allocated."""
        return self._impl.has_grad()

    def has_dual(self) -> bool:
        """Whether this tensor's dual storage is allocated."""
        return self._impl.has_dual()

    # ------------------------------------------------------------------
    # Pickle (symmetric across backends)
    # ------------------------------------------------------------------

    def __reduce__(self) -> typing.Tuple[typing.Any, typing.Tuple[typing.Any, ...]]:
        """Serialize via canonical-view numpy + backend metadata.

        Works uniformly on both backends: the upstream ``Field`` never supported pickle because it needs
        runtime-allocated SNodes, but the wrapper bypasses that by reconstructing via ``qd.tensor(...)`` +
        ``from_numpy(...)`` on the other side - ``qd.tensor`` handles the SNode allocation through the usual factory
        path.

        Only *scalar* tensors are supported here. Vector/matrix wrappers override this (they need to encode element
        shape too).
        """
        backend_int = int(self._backend_enum())
        shape = tuple(self._impl.shape)
        layout = self.layout  # ``None`` for identity, else permutation tuple
        data = self._impl.to_numpy()
        return (
            _rebuild_scalar_tensor,
            (backend_int, self._impl.dtype, shape, layout, data),
        )


def _element_shape_of(impl: typing.Any) -> typing.Tuple[int, ...]:
    """Return the per-element shape of a vector/matrix impl.

    ``VectorNdarray`` / ``MatrixNdarray`` expose ``element_shape`` directly. ``MatrixField`` (which backs
    ``qd.Vector.field`` via ``m == 1, ndim == 1`` and ``qd.Matrix.field`` via ``ndim == 2``) doesn't, so we derive it
    from its ``n``/``m``/``ndim`` attributes.
    """
    from quadrants.lang.matrix import MatrixField

    if isinstance(impl, MatrixField):
        if impl.ndim == 0:
            return ()
        if impl.ndim == 1:
            return (impl.n,)
        return (impl.n, impl.m)
    return tuple(impl.element_shape)


class VectorTensor(Tensor):
    """Wrapper for vector-element tensors (``qd.Vector.tensor(...)`` output).

    Accepts either a ``VectorNdarray`` or a ``MatrixField`` allocated via ``qd.Vector.field(...)`` (``m == 1,
    ndim == 1``). ``element_shape`` is ``(n,)``.
    """

    def __init__(self, impl: typing.Any) -> None:
        from quadrants.lang.matrix import MatrixField, VectorNdarray

        if isinstance(impl, VectorNdarray):
            pass
        elif isinstance(impl, MatrixField) and impl.ndim == 1:
            pass
        else:
            raise TypeError(f"VectorTensor requires a vector-element impl; got {type(impl).__name__}")
        self._impl: typing.Any = impl
        global _any_tensor_constructed  # noqa: PLW0603
        _any_tensor_constructed = True

    @property
    def element_shape(self) -> typing.Tuple[int, ...]:
        return _element_shape_of(self._impl)

    def __reduce__(self) -> typing.Tuple[typing.Any, typing.Tuple[typing.Any, ...]]:
        backend_int = int(self._backend_enum())
        shape = tuple(self._impl.shape)
        element_shape = _element_shape_of(self._impl)
        data = self._impl.to_numpy()
        return (
            _rebuild_vector_tensor,
            (backend_int, self._impl.dtype, shape, element_shape, data),
        )


class MatrixTensor(Tensor):
    """Wrapper for matrix-element tensors (``qd.Matrix.tensor(...)`` output).

    Accepts either a ``MatrixNdarray`` or a ``MatrixField`` with ``ndim == 2``. ``element_shape`` is ``(n, m)``.
    """

    def __init__(self, impl: typing.Any) -> None:
        from quadrants.lang.matrix import MatrixField, MatrixNdarray

        if isinstance(impl, MatrixNdarray):
            pass
        elif isinstance(impl, MatrixField) and impl.ndim == 2:
            pass
        else:
            raise TypeError(f"MatrixTensor requires a matrix-element impl; got {type(impl).__name__}")
        self._impl: typing.Any = impl
        global _any_tensor_constructed  # noqa: PLW0603
        _any_tensor_constructed = True

    @property
    def element_shape(self) -> typing.Tuple[int, ...]:
        return _element_shape_of(self._impl)

    def __reduce__(self) -> typing.Tuple[typing.Any, typing.Tuple[typing.Any, ...]]:
        backend_int = int(self._backend_enum())
        shape = tuple(self._impl.shape)
        element_shape = _element_shape_of(self._impl)
        data = self._impl.to_numpy()
        return (
            _rebuild_matrix_tensor,
            (backend_int, self._impl.dtype, shape, element_shape, data),
        )


# PERF-CRITICAL: Hotpath code uses ``type(arg) in _TENSOR_WRAPPER_TYPES`` instead of ``isinstance(arg, Tensor)``
# because ``type(x) is cls`` is a single pointer comparison (~10 ns) whereas ``isinstance`` walks the MRO for
# non-matching types (~100–200 ns). With ~43 struct fields checked per kernel invocation in Genesis, the cumulative
# savings are significant. Keep this tuple in sync with the Tensor class hierarchy.
_TENSOR_WRAPPER_TYPES = (Tensor, VectorTensor, MatrixTensor)


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------


def wrap(impl: typing.Any) -> "Tensor":
    """Wrap an impl in the most specific ``Tensor`` subclass we can.

    Used internally (e.g. by lazy ``.grad``) and by tests.
    """
    from quadrants.lang.matrix import MatrixField, MatrixNdarray, VectorNdarray

    if isinstance(impl, VectorNdarray):
        return VectorTensor(impl)
    if isinstance(impl, MatrixNdarray):
        return MatrixTensor(impl)
    if isinstance(impl, MatrixField):
        if impl.ndim == 1:
            return VectorTensor(impl)
        if impl.ndim == 2:
            return MatrixTensor(impl)
        # ndim == 0: scalar-like; fall through to base Tensor.
    return Tensor(impl)


# ----------------------------------------------------------------------
# Pickle reconstructors (module-level so pickle can find them by name)
# ----------------------------------------------------------------------


def _rebuild_scalar_tensor(
    backend_int: int,
    dtype: typing.Any,
    shape: typing.Tuple[int, ...],
    layout: typing.Optional[typing.Tuple[int, ...]],
    data: typing.Any,
) -> "Tensor":
    import quadrants as qd

    backend = qd.Backend(backend_int)  # type: ignore[reportOptionalCall]
    kwargs: typing.Dict[str, typing.Any] = {"backend": backend}
    if layout is not None:
        kwargs["layout"] = layout
    # ``qd.tensor()`` already returns a Tensor wrapper post stork-19.
    t = qd.tensor(dtype, shape, **kwargs)  # type: ignore[reportOptionalCall]
    t.from_numpy(data)  # type: ignore[reportAttributeAccessIssue]
    return t  # type: ignore[reportReturnType]


def _rebuild_vector_tensor(
    backend_int: int,
    dtype: typing.Any,
    shape: typing.Tuple[int, ...],
    element_shape: typing.Tuple[int, ...],
    data: typing.Any,
) -> "VectorTensor":
    import quadrants as qd

    backend = qd.Backend(backend_int)  # type: ignore[reportOptionalCall]
    (n,) = element_shape
    t = qd.Vector.tensor(n, dtype, shape, backend=backend)  # type: ignore[reportAttributeAccessIssue]
    t.from_numpy(data)
    return t


def _rebuild_matrix_tensor(
    backend_int: int,
    dtype: typing.Any,
    shape: typing.Tuple[int, ...],
    element_shape: typing.Tuple[int, ...],
    data: typing.Any,
) -> "MatrixTensor":
    import quadrants as qd

    backend = qd.Backend(backend_int)  # type: ignore[reportOptionalCall]
    n, m = element_shape
    t = qd.Matrix.tensor(n, m, dtype, shape, backend=backend)  # type: ignore[reportAttributeAccessIssue]
    t.from_numpy(data)
    return t
