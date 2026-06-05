# pyright: reportInvalidTypeForm=false

"""
Register-resident 16x16 tile operations.

Each tile is a 16x16 matrix distributed across 16 threads in a subgroup, one row per thread, with each row stored
in 16 scalar registers held in an unpacked vector field (``self.r``).  Cross-thread communication uses subgroup
shuffles -- no shared memory needed.

The thread's lane index (tid) is obtained internally via ``subgroup.invocation_id()``, so callers never need to
pass it.  See docs/source/user_guide/tile.md for usage documentation.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, NoReturn

import quadrants as qd

if _TYPE_CHECKING:

    class _Tile16x16Proto:  # noqa: E303
        """Static type stub so pyright sees Tile16x16 methods correctly."""

        SIZE: int

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
        @classmethod
        def zeros(cls) -> "_Tile16x16Proto": ...  # noqa: E704
        @classmethod
        def eye(cls) -> "_Tile16x16Proto": ...  # noqa: E704
        def eye_(self) -> None: ...  # noqa: E704
        def cholesky_(self, eps: Any) -> None: ...  # noqa: E704
        def solve_triangular_(self, B: "_Tile16x16Proto", lower: bool = True) -> None: ...  # noqa: E704
        def _load(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store(
            self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _load3d(
            self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _store3d(
            self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _ger_sub(self, a: Any, b: Any) -> None: ...  # noqa: E704
        def _trsm(self, L: "_Tile16x16Proto") -> None: ...  # noqa: E704
        def __isub__(self, other: Any) -> "_Tile16x16Proto": ...  # noqa: E704
        def __getitem__(self, key: Any) -> Any: ...  # noqa: E704
        def __setitem__(self, key: Any, value: Any) -> None: ...  # noqa: E704


_TILE = 16


class _OuterProduct:
    """Deferred outer product proxy for use with augmented assignment on Tile16x16.

    Created by qd.outer(a, b). Not a quadrants expression -- only valid as the RHS of ``tile -= qd.outer(a, b)``.
    """

    _qd_is_deferred = True

    def __init__(self, a: Any, b: Any) -> None:
        self.a = a
        self.b = b

    def __add__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")

    def __radd__(self, other: Any) -> NoReturn:
        raise TypeError("OuterProduct does not support composition; apply each update separately")


def outer(a: Any, b: Any) -> _OuterProduct:
    """Create a deferred outer product for use with Tile16x16 augmented assignment.

    Usage::

        t -= qd.outer(a, b)   # equivalent to t._ger_sub(a, b)
        t -= qd.outer(v, v)   # symmetric case (a == b)
    """
    return _OuterProduct(a, b)


class _DeferredProxyMixin:
    """Raises clear errors if a deferred tile proxy is accidentally used as a value."""

    _proxy_description = "Tile proxy"

    def _misuse(self, op: str = "used") -> NoReturn:
        raise TypeError(
            f"{self._proxy_description} was {op}, but it is only valid in tile operations (tile[:] = ..., ... = tile, qd.outer(...))"
        )

    def __add__(self, other: Any) -> NoReturn:
        self._misuse("added")

    def __radd__(self, other: Any) -> NoReturn:
        self._misuse("added")

    def __sub__(self, other: Any) -> NoReturn:
        self._misuse("subtracted")

    def __mul__(self, other: Any) -> NoReturn:
        self._misuse("multiplied")

    def __getitem__(self, key: Any) -> NoReturn:
        self._misuse("subscripted")

    def __repr__(self) -> str:
        return f"<{self._proxy_description} — not a value; use with tile[:] = ... or qd.outer(...)>"


class _TileSliceProxy(_DeferredProxyMixin):
    """Deferred 2D/3D array slice for tile load/store.

    Created by subscripting a Field or ndarray with 2D slices, e.g. ``arr[row_start:row_stop, col_start:col_stop]``.
    Not a quadrants expression -- only valid as the RHS of a tile assignment (load) or as the LHS target (store).
    """

    _qd_is_deferred = True
    _proxy_description = "Array slice proxy (arr[r0:r1, c0:c1])"

    def __init__(
        self, arr: Any, row_start: Any, row_stop: Any, col_start: Any, col_stop: Any, batch_idx: Any = None
    ) -> None:
        self.arr = arr
        self.row_start = row_start
        self.row_stop = row_stop
        self.col_start = col_start
        self.col_stop = col_stop
        self.batch_idx = batch_idx

    def _assign(self, tile: Any) -> None:
        """Store path: arr[r:r+n_rows, c:c+n_cols] = tile."""
        if self.batch_idx is not None:
            tile._store3d(self.arr, self.batch_idx, self.row_start, self.row_stop, self.col_start, self.col_stop)
        else:
            tile._store(self.arr, self.row_start, self.row_stop, self.col_start, self.col_stop)


class _VecSliceProxy(_DeferredProxyMixin):
    """Deferred column-vector load from a 2D/3D array.

    Created by ``arr[row_start:row_stop, col]`` or ``arr[batch_idx, row_start:row_stop, col]``.
    Each subgroup thread loads one element; out-of-range threads get 0.
    Only valid as an argument to ``qd.outer()`` in tile augmented assignment.
    """

    _qd_is_deferred = True
    _proxy_description = "Vec slice proxy (arr[r0:r1, col])"

    def __init__(self, arr: Any, row_start: Any, row_stop: Any, col: Any, batch_idx: Any = None) -> None:
        self.arr = arr
        self.row_start = row_start
        self.row_stop = row_stop
        self.col = col
        self.batch_idx = batch_idx


class _TileRefProxy:
    """Proxy returned by tile[:] for the LHS of a load assignment.

    Enables ``tile[:] = arr[r:r+16, c:n]``.  The ``[:]`` is required to distinguish in-place tile loads from
    variable rebinding.
    """

    _qd_is_deferred = True

    def __init__(self, tile: Any) -> None:
        self.tile = tile

    def _assign(self, value: Any) -> None:
        """Load path: tile[:] = arr[r:r+n, c:c+n]. Dispatches to _load or _load3d."""
        if isinstance(value, _TileSliceProxy):
            if value.batch_idx is not None:
                self.tile._load3d(
                    value.arr, value.batch_idx, value.row_start, value.row_stop, value.col_start, value.col_stop
                )
            else:
                self.tile._load(value.arr, value.row_start, value.row_stop, value.col_start, value.col_stop)
        else:
            raise TypeError(f"Tile16x16[:] can only be assigned from an array slice, got {type(value)}")


_tile16_cache = {}


def _make_tile16x16(dtype=None) -> "type[_Tile16x16Proto]":
    """Create a Tile16x16 dataclass whose registers use the given scalar dtype (qd.f32 or qd.f64).

    This is an internal factory. Use ``qd.simt.Tile16x16`` (the proxy) instead.
    """
    if dtype is None:
        dtype = qd.f32
    if dtype in _tile16_cache:
        return _tile16_cache[dtype]  # pyright: ignore[reportReturnType]
    cls = _make_tile16x16_class(dtype)
    _tile16_cache[dtype] = cls
    return cls  # pyright: ignore[reportReturnType]


def _make_tile16x16_class(dtype):
    class _Tile16x16:
        """A 16x16 tile distributed one row per subgroup thread, with each row held in 16 scalar registers via an
        unpacked vector field.  ``Tile16x16()`` creates a zero tile."""

        r: qd.types.vector(_TILE, dtype, unpacked=True)

        @qd.func
        def _load(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Load from a 2D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread loads arr[row_start + tid, col_start:col_stop].  Threads where row_start + tid >= row_stop
            skip the load (tile row unchanged).
            """
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        self.r[j] = arr[row, col_start + j]

        @qd.func
        def _load3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Load from a 3D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread loads arr[batch, row_start+tid, col_start:col_stop].  Threads where row_start + tid >=
            row_stop skip the load (tile row unchanged).
            """
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        self.r[j] = arr[batch, row, col_start + j]

        @qd.func
        def _store(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Store to a 2D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread stores to arr[row_start + tid, col_start:col_stop].  Threads where row_start + tid >=
            row_stop skip the store.
            """
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        arr[row, col_start + j] = self.r[j]

        @qd.func
        def _store3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Store to a 3D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread stores to arr[batch, row_start+tid, col_start:col_stop].  Threads where row_start + tid >=
            row_stop skip the store.
            """
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(_TILE)):
                    if col_start + j < col_stop:
                        arr[batch, row, col_start + j] = self.r[j]

        @qd.func
        def eye_(self):
            """Set this tile to the 16x16 identity matrix.  Each thread sets its diagonal element to 1.0 and all
            others to 0.0."""
            tid = qd.simt.subgroup.invocation_id()
            for j in qd.static(range(_TILE)):
                self.r[j] = 1.0 if tid == j else 0.0

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T."""
            for j in qd.static(range(_TILE)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(j))
                self.r[j] = self.r[j] - a * bc

        @qd.func
        def cholesky_(self, eps):
            """In-place 16x16 Cholesky factorization via subgroup shuffles.

            On return, the lower triangle holds L such that A = L @ L^T.  Diagonal clamped to
            sqrt(max(value, eps)) for numerical stability.
            """
            # ``k`` and ``j`` are wrapped in qd.static so the ``if k > j`` predicate folds at compile time and the
            # ``self.r[k]`` / ``self.r[j]`` accesses resolve to a single unpacked-register slot per use (no runtime
            # cascade).  The per-lane row-norm used for the diagonal update is carried in ``my_norm_sq``, so each
            # diagonal step is O(1) rather than O(k).  The off-diagonal ``dot`` is split into two interleaved partial
            # sums (``dot0`` / ``dot1``) so the back-to-back FMA dependency chain is cut in half, exposing more
            # instruction-level parallelism.
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            my_norm_sq = qd.cast(0.0, dtype)
            for k in qd.static(range(_TILE)):
                diag_val = qd.cast(0.0, dtype)
                if tid == k:
                    diag_val = qd.sqrt(qd.max(self.r[k] - my_norm_sq, eps))
                    self.r[k] = diag_val

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

                dot0 = qd.cast(0.0, dtype)
                dot1 = qd.cast(0.0, dtype)
                for j in qd.static(range(_TILE)):
                    if k > j:
                        my_col = self.r[j]
                        Lkj = qd.simt.subgroup.shuffle(my_col, qd.u32(k))
                        if j % 2 == 0:
                            dot0 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                        else:
                            dot1 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                dot = dot0 + dot1

                new_val = qd.cast(0.0, dtype)
                if tid > k:  # type: ignore[reportOperatorIssue]
                    new_val = (self.r[k] - dot) / diag_k  # type: ignore[reportOperatorIssue]
                    self.r[k] = new_val
                if tid > k:  # type: ignore[reportOperatorIssue]
                    my_norm_sq += new_val * new_val

        @qd.func
        def _trsm(self, L):
            """In-place triangular solve: solve self @ L^T = B (original self).

            L is a Tile16x16 holding the lower-triangular Cholesky factor (from cholesky_).  On return, self holds
            the solution X.
            """
            for c in qd.static(range(_TILE)):
                dot = qd.cast(0.0, dtype)
                for j in qd.static(range(_TILE)):
                    if c > j:
                        Lkj = qd.simt.subgroup.shuffle(L.r[j], qd.u32(c))
                        dot += self.r[j] * Lkj  # type: ignore[reportOperatorIssue]

                diag_c = qd.simt.subgroup.shuffle(L.r[c], qd.u32(c))
                self.r[c] = (self.r[c] - dot) / diag_c  # type: ignore[reportOperatorIssue]

        def solve_triangular_(self, B: Any, lower: bool = True) -> None:
            """Triangular solve: X @ self^T = B, storing result X in B in-place.

            self must be lower-triangular and non-singular (all diagonal elements non-zero).  Passing a singular
            matrix causes division by zero, producing inf/NaN without warning.  Only lower=True is supported.
            """
            if not lower:
                raise TypeError("Tile16x16.solve_triangular_: only lower=True is supported")
            B._trsm(self)

        @qd.func
        def _resolve_vec2d(self, arr: qd.template(), row_start, row_stop, col):
            """Load one scalar per thread from a 2D array column, clamped to array bounds."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + tid < row_stop:
                v = arr[row_start + tid, col]
            return v

        @qd.func
        def _resolve_vec3d(self, arr: qd.template(), batch, row_start, row_stop, col):
            """Load one scalar per thread from a 3D array column, clamped to array bounds."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            v = dtype(0.0)
            if row_start + tid < row_stop:
                v = arr[batch, row_start + tid, col]
            return v

        def _resolve_vec_proxy(self, proxy: _VecSliceProxy) -> Any:
            """Materialize a _VecSliceProxy into a scalar by dispatching to _resolve_vec2d or _resolve_vec3d."""
            if proxy.batch_idx is not None:
                return self._resolve_vec3d(proxy.arr, proxy.batch_idx, proxy.row_start, proxy.row_stop, proxy.col)
            return self._resolve_vec2d(proxy.arr, proxy.row_start, proxy.row_stop, proxy.col)

        def _augassign(self, other: Any, op: str) -> None:
            """Handle augmented assignment (e.g. tile -= qd.outer(a, b)).

            Resolves _VecSliceProxy arguments and dispatches to _ger_sub.  Only 'Sub' is supported.
            """
            if isinstance(other, _OuterProduct):
                if op == "Sub":
                    a_orig = other.a
                    b_orig = other.b
                    a = self._resolve_vec_proxy(a_orig) if isinstance(a_orig, _VecSliceProxy) else a_orig
                    b = (
                        a
                        if (b_orig is a_orig)
                        else (self._resolve_vec_proxy(b_orig) if isinstance(b_orig, _VecSliceProxy) else b_orig)
                    )
                    self._ger_sub(a, b)
                else:
                    raise TypeError(f"Tile16x16: unsupported augmented assignment op '{op}' with outer product")
            else:
                raise TypeError(f"Tile16x16: unsupported augmented assignment with {type(other)}")

    # StructType.__call__ already defaults missing args to 0, so Tile() produces a zero-initialized tile
    # without needing default values in the class definition (which @qd.dataclass doesn't support).
    result = qd.dataclass(_Tile16x16)
    result.SIZE = _TILE  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t.eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]
    return result


class _Tile16x16Proxy:
    """Proxy for dtype-at-point-of-use tile creation.

    Use as ``qd.simt.Tile16x16.zeros(dtype=qd.f32)`` inside a kernel. The dtype is resolved at kernel compilation
    time, defaulting to the compile config's ``default_fp`` if omitted.
    """

    SIZE = _TILE

    @staticmethod
    def _resolve(dtype):
        from quadrants.lang import impl  # pylint: disable=import-outside-toplevel
        from quadrants.lang.exception import (  # pylint: disable=import-outside-toplevel
            QuadrantsSyntaxError,
        )

        arch = impl.current_cfg().arch
        if arch in (qd.cpu, qd.x64, getattr(qd, "arm64", None)):
            raise QuadrantsSyntaxError(
                "Tile16x16 requires a GPU backend (cuda, metal, vulkan, amdgpu). " f"Current arch is {arch}."
            )
        if dtype is None:
            dtype = impl.get_runtime().default_fp
        if dtype in _tile16_cache:
            return _tile16_cache[dtype]
        return _make_tile16x16(dtype)

    def zeros(self, *, dtype=None):
        """Zero-initialized tile."""
        return self._resolve(dtype)()

    def eye(self, *, dtype=None):
        """Identity tile (diagonal = 1, rest = 0)."""
        return self._resolve(dtype).eye()


Tile16x16Proxy = _Tile16x16Proxy()
