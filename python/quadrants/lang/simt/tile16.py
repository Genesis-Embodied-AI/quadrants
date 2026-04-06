# type: ignore

"""
Register-resident 16x16 tile operations using subgroup (warp) shuffles.

Each tile is a 16x16 matrix distributed across 16 threads in a subgroup,
one row per thread, with each row stored in 16 scalar registers (r0-r15).
Cross-thread communication uses warp shuffles — no shared memory needed.

The thread's lane index (tid) is obtained internally via subgroup.invocation_id(),
so callers never need to pass it. Load/store methods take a row offset (row0);
each thread accesses row = row0 + tid.

Usage example::

    Tile16x16 = make_tile16x16(qd.f32)              # create f32 tile class (or qd.f64 for double precision)
    t = Tile16x16.zeros()                        # zero-initialized tile
    t = Tile16x16.eye()                          # identity tile
    t[:] = arr[r0:r0+16, c0:c0+n]             # load from 2D array (slice syntax)
    t[:] = arr[i0, r0:r0+16, c0:c0+n]        # load from 3D array (slice syntax)
    t.eye_()                                  # set to identity matrix (in-place)
    t -= qd.outer(a, b)                       # rank-1 subtract: t -= a @ b^T
    t -= qd.outer(v, v)                       # symmetric rank-1 subtract
    t.cholesky_(eps)                           # in-place Cholesky factorization
    L.solve_triangular_(B)                     # triangular solve: X @ L^T = B, result in B
    arr[r0:r0+16, c0:c0+n] = t               # store to 2D array (slice syntax)
    arr[i0, r0:r0+16, c0:c0+n] = t           # store to 3D array (slice syntax)
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING

import quadrants as qd

if _TYPE_CHECKING:
    from typing import Any

    class _Tile16x16Proto:  # noqa: E303
        """Static type stub so pyright sees Tile16x16 methods correctly."""

        SIZE: int

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
        @classmethod
        def zeros(cls) -> _Tile16x16Proto: ...  # noqa: E704
        @classmethod
        def eye(cls) -> _Tile16x16Proto: ...  # noqa: E704
        def eye_(self) -> None: ...  # noqa: E704
        def cholesky_(self, eps: Any) -> None: ...  # noqa: E704
        def solve_triangular_(self, B: _Tile16x16Proto, lower: bool = True) -> None: ...  # noqa: E704
        def load(self, arr: Any, row0: Any, col0: Any, n_cols: Any) -> None: ...  # noqa: E704
        def store(self, arr: Any, row0: Any, col0: Any, n_cols: Any) -> None: ...  # noqa: E704
        def load3d(self, arr: Any, i0: Any, row0: Any, col0: Any, n_cols: Any) -> None: ...  # noqa: E704
        def store3d(self, arr: Any, i0: Any, row0: Any, col0: Any, n_cols: Any) -> None: ...  # noqa: E704
        def syr_sub(self, v: Any) -> None: ...  # noqa: E704
        def ger_sub(self, a: Any, b: Any) -> None: ...  # noqa: E704
        def trsm(self, L: _Tile16x16Proto) -> None: ...  # noqa: E704
        def __isub__(self, other: Any) -> _Tile16x16Proto: ...  # noqa: E704
        def __getitem__(self, key: Any) -> Any: ...  # noqa: E704
        def __setitem__(self, key: Any, value: Any) -> None: ...  # noqa: E704


_TILE = 16


class _OuterProduct:
    """Deferred outer product proxy for use with augmented assignment on Tile16x16.

    Created by qd.outer(a, b). Not a quadrants expression — only valid as the
    RHS of ``tile -= qd.outer(a, b)``.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __add__(self, other):
        raise TypeError("OuterProduct does not support composition; apply each update separately")

    def __radd__(self, other):
        raise TypeError("OuterProduct does not support composition; apply each update separately")


def outer(a, b):
    """Create a deferred outer product for use with Tile16x16 augmented assignment.

    Usage::

        t -= qd.outer(a, b)   # equivalent to t.ger_sub(a, b)
        t -= qd.outer(v, v)   # equivalent to t.syr_sub(v)
    """
    return _OuterProduct(a, b)


class _TileSliceProxy:
    """Deferred 2D/3D array slice for tile load/store.

    Created by subscripting a Field or ndarray with 2D slices, e.g.
    ``arr[k0:k0+16, k0:k0+16]``.  Not a quadrants expression — only valid
    as the RHS of a tile assignment (load) or as the LHS target (store).
    """

    _is_deferred = True

    def __init__(self, arr, row_start, col_start, col_stop, batch_idx=None):
        self.arr = arr
        self.row_start = row_start
        self.col_start = col_start
        self.col_stop = col_stop
        self.batch_idx = batch_idx

    def _assign(self, tile):
        """Store path: arr[r:r+16, c:c+n] = tile."""
        if self.batch_idx is not None:
            tile.store3d(self.arr, self.batch_idx, self.row_start, self.col_start, self.col_stop)
        else:
            tile.store(self.arr, self.row_start, self.col_start, self.col_stop)


class _TileRefProxy:
    """Proxy returned by tile[:] for the LHS of a load assignment.

    Enables ``tile[:] = arr[r:r+16, c:n]``.  The ``[:]`` is required to
    distinguish in-place tile loads from variable rebinding.
    """

    _is_deferred = True

    def __init__(self, tile):
        self.tile = tile

    def _assign(self, value):
        if isinstance(value, _TileSliceProxy):
            if value.batch_idx is not None:
                self.tile.load3d(value.arr, value.batch_idx, value.row_start, value.col_start, value.col_stop)
            else:
                self.tile.load(value.arr, value.row_start, value.col_start, value.col_stop)
        else:
            raise TypeError(f"Tile16x16[:] can only be assigned from an array slice, got {type(value)}")


# =============================================================================
# Tile16x16 factory — creates a Tile16x16 class for the given scalar dtype
# =============================================================================

_tile16_cache = {}


def make_tile16x16(dtype=qd.f32) -> "type[_Tile16x16Proto]":
    """Create a Tile16x16 dataclass whose registers use the given scalar dtype (qd.f32 or qd.f64)."""
    if dtype in _tile16_cache:
        return _tile16_cache[dtype]
    cls = _make_tile16x16_class(dtype)
    _tile16_cache[dtype] = cls
    return cls


def _make_tile16x16_class(dtype):
    class _Tile16x16:
        """A 16x16 tile distributed one row per subgroup thread, held in 16 scalar registers.

        All fields default to 0.0 when omitted: ``Tile16x16.zeros()`` creates a zero tile.
        """

        r0: dtype
        r1: dtype
        r2: dtype
        r3: dtype
        r4: dtype
        r5: dtype
        r6: dtype
        r7: dtype
        r8: dtype
        r9: dtype
        r10: dtype
        r11: dtype
        r12: dtype
        r13: dtype
        r14: dtype
        r15: dtype

        @qd.func
        def load(self, arr: qd.template(), row0, col0, n_cols):
            """Load one row from a 2D array with column bounds checking.

            Each thread loads arr[row0 + tid, col0:col0+16].
            n_cols is clamped to arr.shape[1] to prevent out-of-bounds access.
            """
            arr_n_cols = arr.shape[1]
            if arr_n_cols < n_cols:
                n_cols = arr_n_cols
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if col0 + 0 < n_cols:
                self.r0 = arr[row, col0 + 0]
            if col0 + 1 < n_cols:
                self.r1 = arr[row, col0 + 1]
            if col0 + 2 < n_cols:
                self.r2 = arr[row, col0 + 2]
            if col0 + 3 < n_cols:
                self.r3 = arr[row, col0 + 3]
            if col0 + 4 < n_cols:
                self.r4 = arr[row, col0 + 4]
            if col0 + 5 < n_cols:
                self.r5 = arr[row, col0 + 5]
            if col0 + 6 < n_cols:
                self.r6 = arr[row, col0 + 6]
            if col0 + 7 < n_cols:
                self.r7 = arr[row, col0 + 7]
            if col0 + 8 < n_cols:
                self.r8 = arr[row, col0 + 8]
            if col0 + 9 < n_cols:
                self.r9 = arr[row, col0 + 9]
            if col0 + 10 < n_cols:
                self.r10 = arr[row, col0 + 10]
            if col0 + 11 < n_cols:
                self.r11 = arr[row, col0 + 11]
            if col0 + 12 < n_cols:
                self.r12 = arr[row, col0 + 12]
            if col0 + 13 < n_cols:
                self.r13 = arr[row, col0 + 13]
            if col0 + 14 < n_cols:
                self.r14 = arr[row, col0 + 14]
            if col0 + 15 < n_cols:
                self.r15 = arr[row, col0 + 15]

        @qd.func
        def load3d(self, arr: qd.template(), i0, row0, col0, n_cols):
            """Load one row from a 3D array: arr[i0, row0+tid, col0+c] with column bounds checking.

            n_cols is clamped to arr.shape[2] to prevent out-of-bounds access.
            """
            arr_n_cols = arr.shape[2]
            if arr_n_cols < n_cols:
                n_cols = arr_n_cols
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if col0 + 0 < n_cols:
                self.r0 = arr[i0, row, col0 + 0]
            if col0 + 1 < n_cols:
                self.r1 = arr[i0, row, col0 + 1]
            if col0 + 2 < n_cols:
                self.r2 = arr[i0, row, col0 + 2]
            if col0 + 3 < n_cols:
                self.r3 = arr[i0, row, col0 + 3]
            if col0 + 4 < n_cols:
                self.r4 = arr[i0, row, col0 + 4]
            if col0 + 5 < n_cols:
                self.r5 = arr[i0, row, col0 + 5]
            if col0 + 6 < n_cols:
                self.r6 = arr[i0, row, col0 + 6]
            if col0 + 7 < n_cols:
                self.r7 = arr[i0, row, col0 + 7]
            if col0 + 8 < n_cols:
                self.r8 = arr[i0, row, col0 + 8]
            if col0 + 9 < n_cols:
                self.r9 = arr[i0, row, col0 + 9]
            if col0 + 10 < n_cols:
                self.r10 = arr[i0, row, col0 + 10]
            if col0 + 11 < n_cols:
                self.r11 = arr[i0, row, col0 + 11]
            if col0 + 12 < n_cols:
                self.r12 = arr[i0, row, col0 + 12]
            if col0 + 13 < n_cols:
                self.r13 = arr[i0, row, col0 + 13]
            if col0 + 14 < n_cols:
                self.r14 = arr[i0, row, col0 + 14]
            if col0 + 15 < n_cols:
                self.r15 = arr[i0, row, col0 + 15]

        @qd.func
        def store(self, arr: qd.template(), row0, col0, n_cols):
            """Store one row to a 2D array with column bounds checking.

            Each thread stores to arr[row0 + tid, col0:col0+16].
            n_cols is clamped to arr.shape[1] to prevent out-of-bounds access.
            """
            arr_n_cols = arr.shape[1]
            if arr_n_cols < n_cols:
                n_cols = arr_n_cols
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if col0 + 0 < n_cols:
                arr[row, col0 + 0] = self.r0
            if col0 + 1 < n_cols:
                arr[row, col0 + 1] = self.r1
            if col0 + 2 < n_cols:
                arr[row, col0 + 2] = self.r2
            if col0 + 3 < n_cols:
                arr[row, col0 + 3] = self.r3
            if col0 + 4 < n_cols:
                arr[row, col0 + 4] = self.r4
            if col0 + 5 < n_cols:
                arr[row, col0 + 5] = self.r5
            if col0 + 6 < n_cols:
                arr[row, col0 + 6] = self.r6
            if col0 + 7 < n_cols:
                arr[row, col0 + 7] = self.r7
            if col0 + 8 < n_cols:
                arr[row, col0 + 8] = self.r8
            if col0 + 9 < n_cols:
                arr[row, col0 + 9] = self.r9
            if col0 + 10 < n_cols:
                arr[row, col0 + 10] = self.r10
            if col0 + 11 < n_cols:
                arr[row, col0 + 11] = self.r11
            if col0 + 12 < n_cols:
                arr[row, col0 + 12] = self.r12
            if col0 + 13 < n_cols:
                arr[row, col0 + 13] = self.r13
            if col0 + 14 < n_cols:
                arr[row, col0 + 14] = self.r14
            if col0 + 15 < n_cols:
                arr[row, col0 + 15] = self.r15

        @qd.func
        def store3d(self, arr: qd.template(), i0, row0, col0, n_cols):
            """Store one row to a 3D array: arr[i0, row0+tid, col0+c] with column bounds checking.

            n_cols is clamped to arr.shape[2] to prevent out-of-bounds access.
            """
            arr_n_cols = arr.shape[2]
            if arr_n_cols < n_cols:
                n_cols = arr_n_cols
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if col0 + 0 < n_cols:
                arr[i0, row, col0 + 0] = self.r0
            if col0 + 1 < n_cols:
                arr[i0, row, col0 + 1] = self.r1
            if col0 + 2 < n_cols:
                arr[i0, row, col0 + 2] = self.r2
            if col0 + 3 < n_cols:
                arr[i0, row, col0 + 3] = self.r3
            if col0 + 4 < n_cols:
                arr[i0, row, col0 + 4] = self.r4
            if col0 + 5 < n_cols:
                arr[i0, row, col0 + 5] = self.r5
            if col0 + 6 < n_cols:
                arr[i0, row, col0 + 6] = self.r6
            if col0 + 7 < n_cols:
                arr[i0, row, col0 + 7] = self.r7
            if col0 + 8 < n_cols:
                arr[i0, row, col0 + 8] = self.r8
            if col0 + 9 < n_cols:
                arr[i0, row, col0 + 9] = self.r9
            if col0 + 10 < n_cols:
                arr[i0, row, col0 + 10] = self.r10
            if col0 + 11 < n_cols:
                arr[i0, row, col0 + 11] = self.r11
            if col0 + 12 < n_cols:
                arr[i0, row, col0 + 12] = self.r12
            if col0 + 13 < n_cols:
                arr[i0, row, col0 + 13] = self.r13
            if col0 + 14 < n_cols:
                arr[i0, row, col0 + 14] = self.r14
            if col0 + 15 < n_cols:
                arr[i0, row, col0 + 15] = self.r15

        @qd.func
        def eye_(self):
            """Set this tile to the 16x16 identity matrix.

            Each thread sets its diagonal element to 1.0 and all others to 0.0.
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            self.r0 = 0.0
            self.r1 = 0.0
            self.r2 = 0.0
            self.r3 = 0.0
            self.r4 = 0.0
            self.r5 = 0.0
            self.r6 = 0.0
            self.r7 = 0.0
            self.r8 = 0.0
            self.r9 = 0.0
            self.r10 = 0.0
            self.r11 = 0.0
            self.r12 = 0.0
            self.r13 = 0.0
            self.r14 = 0.0
            self.r15 = 0.0
            if tid == 0:
                self.r0 = 1.0
            if tid == 1:
                self.r1 = 1.0
            if tid == 2:
                self.r2 = 1.0
            if tid == 3:
                self.r3 = 1.0
            if tid == 4:
                self.r4 = 1.0
            if tid == 5:
                self.r5 = 1.0
            if tid == 6:
                self.r6 = 1.0
            if tid == 7:
                self.r7 = 1.0
            if tid == 8:
                self.r8 = 1.0
            if tid == 9:
                self.r9 = 1.0
            if tid == 10:
                self.r10 = 1.0
            if tid == 11:
                self.r11 = 1.0
            if tid == 12:
                self.r12 = 1.0
            if tid == 13:
                self.r13 = 1.0
            if tid == 14:
                self.r14 = 1.0
            if tid == 15:
                self.r15 = 1.0

        @qd.func
        def syr_sub(self, v):
            """Symmetric rank-1 subtract in-place: self -= v @ v^T."""
            vc = qd.simt.subgroup.shuffle(v, qd.u32(0))
            self.r0 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(1))
            self.r1 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(2))
            self.r2 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(3))
            self.r3 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(4))
            self.r4 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(5))
            self.r5 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(6))
            self.r6 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(7))
            self.r7 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(8))
            self.r8 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(9))
            self.r9 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(10))
            self.r10 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(11))
            self.r11 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(12))
            self.r12 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(13))
            self.r13 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(14))
            self.r14 -= v * vc
            vc = qd.simt.subgroup.shuffle(v, qd.u32(15))
            self.r15 -= v * vc

        @qd.func
        def ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T."""
            bc = qd.simt.subgroup.shuffle(b, qd.u32(0))
            self.r0 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(1))
            self.r1 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(2))
            self.r2 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(3))
            self.r3 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(4))
            self.r4 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(5))
            self.r5 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(6))
            self.r6 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(7))
            self.r7 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(8))
            self.r8 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(9))
            self.r9 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(10))
            self.r10 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(11))
            self.r11 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(12))
            self.r12 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(13))
            self.r13 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(14))
            self.r14 -= a * bc
            bc = qd.simt.subgroup.shuffle(b, qd.u32(15))
            self.r15 -= a * bc

        @qd.func
        def cholesky_(self, eps):
            """In-place 16x16 Cholesky factorization via subgroup shuffles.

            On return, the lower triangle holds L such that A = L @ L^T.
            Diagonal clamped to sqrt(max(value, eps)) for numerical stability.
            """
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            for k in range(_TILE):
                diag_val = qd.cast(0.0, dtype)
                if tid == k:
                    s = qd.cast(0.0, dtype)
                    if k > 0:
                        s += self.r0 * self.r0
                    if k > 1:
                        s += self.r1 * self.r1
                    if k > 2:
                        s += self.r2 * self.r2
                    if k > 3:
                        s += self.r3 * self.r3
                    if k > 4:
                        s += self.r4 * self.r4
                    if k > 5:
                        s += self.r5 * self.r5
                    if k > 6:
                        s += self.r6 * self.r6
                    if k > 7:
                        s += self.r7 * self.r7
                    if k > 8:
                        s += self.r8 * self.r8
                    if k > 9:
                        s += self.r9 * self.r9
                    if k > 10:
                        s += self.r10 * self.r10
                    if k > 11:
                        s += self.r11 * self.r11
                    if k > 12:
                        s += self.r12 * self.r12
                    if k > 13:
                        s += self.r13 * self.r13
                    if k > 14:
                        s += self.r14 * self.r14
                    cur = qd.cast(0.0, dtype)
                    if k == 0:
                        cur = self.r0
                    if k == 1:
                        cur = self.r1
                    if k == 2:
                        cur = self.r2
                    if k == 3:
                        cur = self.r3
                    if k == 4:
                        cur = self.r4
                    if k == 5:
                        cur = self.r5
                    if k == 6:
                        cur = self.r6
                    if k == 7:
                        cur = self.r7
                    if k == 8:
                        cur = self.r8
                    if k == 9:
                        cur = self.r9
                    if k == 10:
                        cur = self.r10
                    if k == 11:
                        cur = self.r11
                    if k == 12:
                        cur = self.r12
                    if k == 13:
                        cur = self.r13
                    if k == 14:
                        cur = self.r14
                    if k == 15:
                        cur = self.r15
                    diag_val = qd.sqrt(qd.max(cur - s, eps))
                    if k == 0:
                        self.r0 = diag_val
                    if k == 1:
                        self.r1 = diag_val
                    if k == 2:
                        self.r2 = diag_val
                    if k == 3:
                        self.r3 = diag_val
                    if k == 4:
                        self.r4 = diag_val
                    if k == 5:
                        self.r5 = diag_val
                    if k == 6:
                        self.r6 = diag_val
                    if k == 7:
                        self.r7 = diag_val
                    if k == 8:
                        self.r8 = diag_val
                    if k == 9:
                        self.r9 = diag_val
                    if k == 10:
                        self.r10 = diag_val
                    if k == 11:
                        self.r11 = diag_val
                    if k == 12:
                        self.r12 = diag_val
                    if k == 13:
                        self.r13 = diag_val
                    if k == 14:
                        self.r14 = diag_val
                    if k == 15:
                        self.r15 = diag_val

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

                dot = qd.cast(0.0, dtype)
                if k > 0:
                    Lkj = qd.simt.subgroup.shuffle(self.r0, qd.u32(k))
                    dot += Lkj * self.r0
                if k > 1:
                    Lkj = qd.simt.subgroup.shuffle(self.r1, qd.u32(k))
                    dot += Lkj * self.r1
                if k > 2:
                    Lkj = qd.simt.subgroup.shuffle(self.r2, qd.u32(k))
                    dot += Lkj * self.r2
                if k > 3:
                    Lkj = qd.simt.subgroup.shuffle(self.r3, qd.u32(k))
                    dot += Lkj * self.r3
                if k > 4:
                    Lkj = qd.simt.subgroup.shuffle(self.r4, qd.u32(k))
                    dot += Lkj * self.r4
                if k > 5:
                    Lkj = qd.simt.subgroup.shuffle(self.r5, qd.u32(k))
                    dot += Lkj * self.r5
                if k > 6:
                    Lkj = qd.simt.subgroup.shuffle(self.r6, qd.u32(k))
                    dot += Lkj * self.r6
                if k > 7:
                    Lkj = qd.simt.subgroup.shuffle(self.r7, qd.u32(k))
                    dot += Lkj * self.r7
                if k > 8:
                    Lkj = qd.simt.subgroup.shuffle(self.r8, qd.u32(k))
                    dot += Lkj * self.r8
                if k > 9:
                    Lkj = qd.simt.subgroup.shuffle(self.r9, qd.u32(k))
                    dot += Lkj * self.r9
                if k > 10:
                    Lkj = qd.simt.subgroup.shuffle(self.r10, qd.u32(k))
                    dot += Lkj * self.r10
                if k > 11:
                    Lkj = qd.simt.subgroup.shuffle(self.r11, qd.u32(k))
                    dot += Lkj * self.r11
                if k > 12:
                    Lkj = qd.simt.subgroup.shuffle(self.r12, qd.u32(k))
                    dot += Lkj * self.r12
                if k > 13:
                    Lkj = qd.simt.subgroup.shuffle(self.r13, qd.u32(k))
                    dot += Lkj * self.r13
                if k > 14:
                    Lkj = qd.simt.subgroup.shuffle(self.r14, qd.u32(k))
                    dot += Lkj * self.r14

                if tid > k:
                    cur = qd.cast(0.0, dtype)
                    if k == 0:
                        cur = self.r0
                    if k == 1:
                        cur = self.r1
                    if k == 2:
                        cur = self.r2
                    if k == 3:
                        cur = self.r3
                    if k == 4:
                        cur = self.r4
                    if k == 5:
                        cur = self.r5
                    if k == 6:
                        cur = self.r6
                    if k == 7:
                        cur = self.r7
                    if k == 8:
                        cur = self.r8
                    if k == 9:
                        cur = self.r9
                    if k == 10:
                        cur = self.r10
                    if k == 11:
                        cur = self.r11
                    if k == 12:
                        cur = self.r12
                    if k == 13:
                        cur = self.r13
                    if k == 14:
                        cur = self.r14
                    if k == 15:
                        cur = self.r15
                    new_val = (cur - dot) / diag_k
                    if k == 0:
                        self.r0 = new_val
                    if k == 1:
                        self.r1 = new_val
                    if k == 2:
                        self.r2 = new_val
                    if k == 3:
                        self.r3 = new_val
                    if k == 4:
                        self.r4 = new_val
                    if k == 5:
                        self.r5 = new_val
                    if k == 6:
                        self.r6 = new_val
                    if k == 7:
                        self.r7 = new_val
                    if k == 8:
                        self.r8 = new_val
                    if k == 9:
                        self.r9 = new_val
                    if k == 10:
                        self.r10 = new_val
                    if k == 11:
                        self.r11 = new_val
                    if k == 12:
                        self.r12 = new_val
                    if k == 13:
                        self.r13 = new_val
                    if k == 14:
                        self.r14 = new_val
                    if k == 15:
                        self.r15 = new_val

        @qd.func
        def trsm(self, L):
            """In-place triangular solve: solve self @ L^T = B (original self).

            L is a Tile16x16 holding the lower-triangular Cholesky factor (from potrf).
            On return, self holds the solution X.
            """
            for c in range(_TILE):
                dot = qd.cast(0.0, dtype)
                if c > 0:
                    Lkj = qd.simt.subgroup.shuffle(L.r0, qd.u32(c))
                    dot += self.r0 * Lkj
                if c > 1:
                    Lkj = qd.simt.subgroup.shuffle(L.r1, qd.u32(c))
                    dot += self.r1 * Lkj
                if c > 2:
                    Lkj = qd.simt.subgroup.shuffle(L.r2, qd.u32(c))
                    dot += self.r2 * Lkj
                if c > 3:
                    Lkj = qd.simt.subgroup.shuffle(L.r3, qd.u32(c))
                    dot += self.r3 * Lkj
                if c > 4:
                    Lkj = qd.simt.subgroup.shuffle(L.r4, qd.u32(c))
                    dot += self.r4 * Lkj
                if c > 5:
                    Lkj = qd.simt.subgroup.shuffle(L.r5, qd.u32(c))
                    dot += self.r5 * Lkj
                if c > 6:
                    Lkj = qd.simt.subgroup.shuffle(L.r6, qd.u32(c))
                    dot += self.r6 * Lkj
                if c > 7:
                    Lkj = qd.simt.subgroup.shuffle(L.r7, qd.u32(c))
                    dot += self.r7 * Lkj
                if c > 8:
                    Lkj = qd.simt.subgroup.shuffle(L.r8, qd.u32(c))
                    dot += self.r8 * Lkj
                if c > 9:
                    Lkj = qd.simt.subgroup.shuffle(L.r9, qd.u32(c))
                    dot += self.r9 * Lkj
                if c > 10:
                    Lkj = qd.simt.subgroup.shuffle(L.r10, qd.u32(c))
                    dot += self.r10 * Lkj
                if c > 11:
                    Lkj = qd.simt.subgroup.shuffle(L.r11, qd.u32(c))
                    dot += self.r11 * Lkj
                if c > 12:
                    Lkj = qd.simt.subgroup.shuffle(L.r12, qd.u32(c))
                    dot += self.r12 * Lkj
                if c > 13:
                    Lkj = qd.simt.subgroup.shuffle(L.r13, qd.u32(c))
                    dot += self.r13 * Lkj
                if c > 14:
                    Lkj = qd.simt.subgroup.shuffle(L.r14, qd.u32(c))
                    dot += self.r14 * Lkj

                diag_reg = qd.cast(0.0, dtype)
                if c == 0:
                    diag_reg = L.r0
                if c == 1:
                    diag_reg = L.r1
                if c == 2:
                    diag_reg = L.r2
                if c == 3:
                    diag_reg = L.r3
                if c == 4:
                    diag_reg = L.r4
                if c == 5:
                    diag_reg = L.r5
                if c == 6:
                    diag_reg = L.r6
                if c == 7:
                    diag_reg = L.r7
                if c == 8:
                    diag_reg = L.r8
                if c == 9:
                    diag_reg = L.r9
                if c == 10:
                    diag_reg = L.r10
                if c == 11:
                    diag_reg = L.r11
                if c == 12:
                    diag_reg = L.r12
                if c == 13:
                    diag_reg = L.r13
                if c == 14:
                    diag_reg = L.r14
                if c == 15:
                    diag_reg = L.r15
                diag_c = qd.simt.subgroup.shuffle(diag_reg, qd.u32(c))

                cur = qd.cast(0.0, dtype)
                if c == 0:
                    cur = self.r0
                if c == 1:
                    cur = self.r1
                if c == 2:
                    cur = self.r2
                if c == 3:
                    cur = self.r3
                if c == 4:
                    cur = self.r4
                if c == 5:
                    cur = self.r5
                if c == 6:
                    cur = self.r6
                if c == 7:
                    cur = self.r7
                if c == 8:
                    cur = self.r8
                if c == 9:
                    cur = self.r9
                if c == 10:
                    cur = self.r10
                if c == 11:
                    cur = self.r11
                if c == 12:
                    cur = self.r12
                if c == 13:
                    cur = self.r13
                if c == 14:
                    cur = self.r14
                if c == 15:
                    cur = self.r15

                new_val = (cur - dot) / diag_c

                if c == 0:
                    self.r0 = new_val
                if c == 1:
                    self.r1 = new_val
                if c == 2:
                    self.r2 = new_val
                if c == 3:
                    self.r3 = new_val
                if c == 4:
                    self.r4 = new_val
                if c == 5:
                    self.r5 = new_val
                if c == 6:
                    self.r6 = new_val
                if c == 7:
                    self.r7 = new_val
                if c == 8:
                    self.r8 = new_val
                if c == 9:
                    self.r9 = new_val
                if c == 10:
                    self.r10 = new_val
                if c == 11:
                    self.r11 = new_val
                if c == 12:
                    self.r12 = new_val
                if c == 13:
                    self.r13 = new_val
                if c == 14:
                    self.r14 = new_val
                if c == 15:
                    self.r15 = new_val

        def _augassign(self, other, op):
            if isinstance(other, _OuterProduct):
                if op == "Sub":
                    self.ger_sub(other.a, other.b)
                else:
                    raise TypeError(f"Tile16x16: unsupported augmented assignment op '{op}' with outer product")
            else:
                raise TypeError(f"Tile16x16: unsupported augmented assignment with {type(other)}")

        def solve_triangular_(self, B, lower=True):
            """Triangular solve: X @ self^T = B, storing result X in B in-place.

            self must be lower-triangular (e.g. from cholesky_()).
            Only lower=True is supported.
            """
            if not lower:
                raise TypeError("Tile16x16.solve_triangular_: only lower=True is supported")
            B.trsm(self)

    # StructType.__call__ already defaults missing args to 0, so Tile16x16.zeros()
    # produces a zero-initialized tile without needing default values in the
    # class definition (which @qd.dataclass doesn't support).
    result = qd.dataclass(_Tile16x16)
    result.SIZE = _TILE
    result._quadrants_internal = True
    result.zeros = result

    @qd.func
    def _eye():
        t = result()
        t.eye_()
        return t

    result.eye = _eye
    return result


Tile16x16 = make_tile16x16(qd.f32)
