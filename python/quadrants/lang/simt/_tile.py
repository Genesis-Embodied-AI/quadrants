# pyright: reportInvalidTypeForm=false

"""
Register-resident NxN tile operations.

Each tile is an NxN matrix distributed across N threads in a subgroup, one row per thread, with each row stored in N
scalar registers held in an unpacked vector field (``self.r``).  Cross-thread communication uses subgroup shuffles --
no shared memory needed.

A single factory ``_make_tile_class(N, dtype)`` builds the tile dataclass for both supported tile sizes (N == 16 and
N == 32).  The user-facing entry points are the proxies ``qd.simt.Tile16x16`` and ``qd.simt.Tile32x32``, which defer
dtype resolution to kernel compile time (defaulting to the runtime ``default_fp``).

The thread's lane index (tid) is obtained internally via ``subgroup.invocation_id()``, so callers never need to pass
it.  See docs/source/user_guide/tile.md for usage documentation.
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, NoReturn

import quadrants as qd

if _TYPE_CHECKING:

    class _TileProto:  # noqa: E303
        """Static type stub so pyright sees TileNxN methods correctly (shared by Tile16x16 and Tile32x32)."""

        SIZE: int

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
        @classmethod
        def zeros(cls) -> "_TileProto": ...  # noqa: E704
        @classmethod
        def eye(cls) -> "_TileProto": ...  # noqa: E704
        def eye_(self) -> None: ...  # noqa: E704
        def cholesky_(self, eps: Any) -> None: ...  # noqa: E704
        def solve_triangular_(self, B: "_TileProto", lower: bool = True) -> None: ...  # noqa: E704
        def _get_col(self, k: Any) -> Any: ...  # noqa: E704
        def _set_col(self, k: Any, val: Any) -> None: ...  # noqa: E704
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
        def _trsm(self, L: "_TileProto") -> None: ...  # noqa: E704
        def __isub__(self, other: Any) -> "_TileProto": ...  # noqa: E704
        def __getitem__(self, key: Any) -> Any: ...  # noqa: E704
        def __setitem__(self, key: Any, value: Any) -> None: ...  # noqa: E704


class _OuterProduct:
    """Deferred outer product proxy for use with augmented assignment on a Tile.

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
    """Create a deferred outer product for use with Tile augmented assignment.

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
        return f"<{self._proxy_description} - not a value; use with tile[:] = ... or qd.outer(...)>"


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

    Created by ``arr[row_start:row_stop, col]`` or ``arr[batch_idx, row_start:row_stop, col]``.  Each subgroup thread
    loads one element; out-of-range threads get 0.  Only valid as an argument to ``qd.outer()`` in tile augmented
    assignment.
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

    Enables ``tile[:] = arr[r:r+N, c:n]``.  The ``[:]`` is required to distinguish in-place tile loads from variable
    rebinding.
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
            raise TypeError(f"Tile[:] can only be assigned from an array slice, got {type(value)}")


_tile_cache: dict = {}


def _make_tile(N: int, dtype=None) -> "type[_TileProto]":
    """Create a TileNxN dataclass whose registers use the given scalar dtype (qd.f32 or qd.f64).

    This is an internal factory.  Use ``qd.simt.Tile16x16`` / ``qd.simt.Tile32x32`` (the proxies) instead.
    """
    if dtype is None:
        dtype = qd.f32
    key = (N, dtype)
    if key in _tile_cache:
        return _tile_cache[key]  # pyright: ignore[reportReturnType]
    cls = _make_tile_class(N, dtype)
    _tile_cache[key] = cls
    return cls  # pyright: ignore[reportReturnType]


def _make_tile_class(N: int, dtype):
    name = f"Tile{N}x{N}"

    class _Tile:
        """An NxN tile distributed one row per subgroup thread, with each row held in N scalar registers via an unpacked
        vector field.  ``TileNxN()`` creates a zero tile."""

        r: qd.types.vector(N, dtype, unpacked=True)

        @qd.func
        def _load(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Load from a 2D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread loads arr[row_start + tid, col_start:col_stop].  Threads where row_start + tid >= row_stop skip
            the load (tile row unchanged).
            """
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(N)):
                    if col_start + j < col_stop:
                        self.r[j] = arr[row, col_start + j]

        @qd.func
        def _load3d(self, arr: qd.template(), batch, row_start, row_stop, col_start, col_stop):
            """Load from a 3D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread loads arr[batch, row_start+tid, col_start:col_stop].  Threads where row_start + tid >= row_stop
            skip the load (tile row unchanged).
            """
            arr_row_stop = arr.shape[1]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[2]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(N)):
                    if col_start + j < col_stop:
                        self.r[j] = arr[batch, row, col_start + j]

        @qd.func
        def _store(self, arr: qd.template(), row_start, row_stop, col_start, col_stop):
            """Store to a 2D array within [row_start, row_stop) x [col_start, col_stop).

            Each thread stores to arr[row_start + tid, col_start:col_stop].  Threads where row_start + tid >= row_stop
            skip the store.
            """
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_stop:
                arr_col_stop = arr.shape[1]
                if arr_col_stop < col_stop:
                    col_stop = arr_col_stop
                for j in qd.static(range(N)):
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
                for j in qd.static(range(N)):
                    if col_start + j < col_stop:
                        arr[batch, row, col_start + j] = self.r[j]

        @qd.func
        def eye_(self):
            """Set this tile to the NxN identity matrix.  Each thread sets its diagonal element to 1.0 and all others
            to 0.0."""
            tid = qd.simt.subgroup.invocation_id()
            for j in qd.static(range(N)):
                self.r[j] = 1.0 if tid == j else 0.0

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T."""
            for j in qd.static(range(N)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(j))
                self.r[j] = self.r[j] - a * bc

        @qd.func
        def cholesky_(self, eps):
            """In-place NxN Cholesky factorization via subgroup shuffles.

            On return, the lower triangle holds L such that A = L @ L^T.  Diagonal clamped to sqrt(max(value, eps)) for
            numerical stability.
            """
            # ``k`` and ``j`` are wrapped in qd.static so the ``if k > j`` predicate folds at compile time and the
            # ``self.r[k]`` / ``self.r[j]`` accesses resolve to a single unpacked-register slot per use (no runtime
            # cascade).  The per-lane row-norm used for the diagonal update is carried in ``my_norm_sq``, so each
            # diagonal step is O(1) rather than O(k).  The off-diagonal ``dot`` is split into two interleaved partial
            # sums (``dot0`` / ``dot1``) so the back-to-back FMA dependency chain is cut in half, exposing more ILP.
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            my_norm_sq = qd.cast(0.0, dtype)
            for k in qd.static(range(N)):
                diag_val = qd.cast(0.0, dtype)
                if tid == k:
                    diag_val = qd.sqrt(qd.max(self.r[k] - my_norm_sq, eps))
                    self.r[k] = diag_val

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

                dot0 = qd.cast(0.0, dtype)
                dot1 = qd.cast(0.0, dtype)
                for j in qd.static(range(N)):
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
        def _get_col(self, k):
            """Read register column ``k`` at runtime via a static-unrolled cascade.

            The unpacked vector field rejects runtime indices, so the cascade is emitted explicitly.  With ``k`` a
            runtime int and ``kk`` a python-int from ``qd.static``, the body of each iteration becomes a guarded
            single-slot read; LLVM later selects on ``k`` to pick the matching slot.  Used by ``_trsm`` so the outer
            loop can be a runtime ``range(N)`` (LLVM picks the unroll factor) rather than the fully-unrolled
            ``qd.static(range(N))`` that spikes register pressure.
            """
            val = qd.cast(0.0, dtype)
            for kk in qd.static(range(N)):
                if k == kk:
                    val = self.r[kk]
            return val

        @qd.func
        def _set_col(self, k, val):
            """Write register column ``k`` at runtime via a static-unrolled cascade.  See ``_get_col`` for rationale."""
            for kk in qd.static(range(N)):
                if k == kk:
                    self.r[kk] = val

        @qd.func
        def _trsm(self, L):
            """In-place triangular solve: solve self @ L^T = B (original self).

            L is a TileNxN holding the lower-triangular Cholesky factor (from cholesky_).  On return, self holds the
            solution X.

            The outer loop uses ``range(N)`` (runtime), not ``qd.static(range(N))``, so LLVM can pick the unroll
            factor: fully unrolling the N*N body fully explodes the live set and pushes ~37% more registers into the
            kernel, causing measurable perf loss on the blocked Cholesky benchmark (e.g. ~9% slower on
            ``misc/demos/cholesky_blocked.py`` for N=92).  The inner ``j`` loop is also ``range(N)`` for the same
            reason.  Runtime access into the unpacked-vector field goes through ``_get_col`` / ``_set_col`` which emit
            explicit cascades over ``self.r[kk]`` for static ``kk``.
            """
            for c in range(N):
                dot = qd.cast(0.0, dtype)
                for j in range(N):
                    if c > j:
                        Lkj = qd.simt.subgroup.shuffle(L._get_col(j), qd.u32(c))
                        dot += self._get_col(j) * Lkj  # type: ignore[reportOperatorIssue]

                diag_c = qd.simt.subgroup.shuffle(L._get_col(c), qd.u32(c))
                new_val = (self._get_col(c) - dot) / diag_c  # type: ignore[reportOperatorIssue]
                self._set_col(c, new_val)

        def solve_triangular_(self, B: Any, lower: bool = True) -> None:
            """Triangular solve: X @ self^T = B, storing result X in B in-place.

            self must be lower-triangular and non-singular (all diagonal elements non-zero).  Passing a singular matrix
            causes division by zero, producing inf/NaN without warning.  Only lower=True is supported.
            """
            if not lower:
                raise TypeError(f"{name}.solve_triangular_: only lower=True is supported")
            B._trsm(self)

        @qd.func
        def _resolve_vec2d(self, arr: qd.template(), row_start, row_stop, col):
            """Load one scalar per thread from a 2D array column, clamped to array bounds."""
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            arr_row_stop = arr.shape[0]
            if arr_row_stop < row_stop:
                row_stop = arr_row_stop
            # Use qd.cast, not dtype(0.0): the AST transformer only treats a call as a type construction when id(dtype)
            # is in primitive_types.type_ids, but a dtype resolved from a deep-copied default_fp (e.g. after
            # qd.init(default_fp=qd.f32)) has a different id and falls through to a raw call, raising "Quadrants data
            # types cannot be called outside Quadrants kernels".  qd.cast is identity-independent and folds to the same
            # typed constant.
            v = qd.cast(0.0, dtype)
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
            v = qd.cast(0.0, dtype)  # see _resolve_vec2d for why qd.cast (not dtype(0.0))
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
                    raise TypeError(f"{name}: unsupported augmented assignment op '{op}' with outer product")
            else:
                raise TypeError(f"{name}: unsupported augmented assignment with {type(other)}")

    _Tile.__name__ = f"_{name}"
    _Tile.__qualname__ = f"_make_tile_class.<locals>._{name}"

    # StructType.__call__ already defaults missing args to 0, so Tile() produces a zero-initialized tile without needing
    # default values in the class definition (which @qd.dataclass doesn't support).
    result = qd.dataclass(_Tile)
    result.SIZE = N  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t.eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]
    return result


class _TileProxy:
    """Proxy for dtype-at-point-of-use tile creation.

    Use as ``qd.simt.Tile16x16.zeros(dtype=qd.f32)`` or ``qd.simt.Tile32x32.zeros(dtype=qd.f32)`` inside a kernel.
    The dtype is resolved at kernel compilation time, defaulting to the compile config's ``default_fp`` if omitted.
    """

    def __init__(self, N: int) -> None:
        self._N = N
        self.SIZE = N

    def _resolve(self, dtype):
        from quadrants.lang import impl  # pylint: disable=import-outside-toplevel
        from quadrants.lang.exception import (  # pylint: disable=import-outside-toplevel
            QuadrantsSyntaxError,
        )

        arch = impl.current_cfg().arch
        if arch in (qd.cpu, qd.x64, getattr(qd, "arm64", None)):
            raise QuadrantsSyntaxError(
                f"Tile{self._N}x{self._N} requires a GPU backend (cuda, metal, vulkan, amdgpu). "
                f"Current arch is {arch}."
            )
        if dtype is None:
            dtype = impl.get_runtime().default_fp
        return _make_tile(self._N, dtype)

    def zeros(self, *, dtype=None):
        """Zero-initialized tile."""
        return self._resolve(dtype)()

    def eye(self, *, dtype=None):
        """Identity tile (diagonal = 1, rest = 0)."""
        return self._resolve(dtype).eye()


Tile16x16Proxy = _TileProxy(16)
Tile32x32Proxy = _TileProxy(32)
