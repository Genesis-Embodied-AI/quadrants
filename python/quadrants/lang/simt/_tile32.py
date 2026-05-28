# pyright: reportInvalidTypeForm=false

"""
Register-resident 32x32 tile operations.

Each tile is a 32x32 matrix distributed across 32 threads in a subgroup, one row per thread, with each row stored
in 32 scalar registers (r0-r31).  Cross-thread communication uses subgroup shuffles -- no shared memory needed.

Surface mirrors :mod:`quadrants.lang.simt._tile16`: use ``qd.simt.Tile32x32.zeros(dtype=...)`` /
``qd.simt.Tile32x32.eye(dtype=...)`` inside a kernel, then load / store via slice syntax
(``tile[:] = arr[r0:r1, c0:c1]`` / ``arr[r0:r1, c0:c1] = tile``) and update via ``tile -= qd.outer(a, b)``.

The slice / outer-product proxy classes are imported from :mod:`._tile16` so a single set of dispatch helpers serves
both tile sizes (a tile instance is duck-typed against ``_load`` / ``_store`` / ``_ger_sub``).
"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any, NoReturn

import quadrants as qd

from quadrants.lang.simt._tile16 import (
    _OuterProduct,
    _TileRefProxy,
    _TileSliceProxy,
    _VecSliceProxy,
)

if _TYPE_CHECKING:

    class _Tile32x32Proto:  # noqa: E303
        """Static type stub so pyright sees Tile32x32 methods correctly."""

        SIZE: int

        def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: E704
        @classmethod
        def zeros(cls) -> "_Tile32x32Proto": ...  # noqa: E704
        @classmethod
        def eye(cls) -> "_Tile32x32Proto": ...  # noqa: E704
        def eye_(self) -> None: ...  # noqa: E704
        def cholesky_(self, eps: Any) -> None: ...  # noqa: E704
        def solve_triangular_(self, B: "_Tile32x32Proto", lower: bool = True) -> None: ...  # noqa: E704
        def _load(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store(self, arr: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _load3d(self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any) -> None: ...  # noqa: E704
        def _store3d(
            self, arr: Any, batch: Any, row_start: Any, row_end: Any, col_start: Any, col_end: Any
        ) -> None: ...  # noqa: E704
        def _get_col(self, k: Any) -> Any: ...  # noqa: E704
        def _set_col(self, k: Any, val: Any) -> None: ...  # noqa: E704
        def _ger_sub(self, a: Any, b: Any) -> None: ...  # noqa: E704
        def _trsm(self, L: "_Tile32x32Proto") -> None: ...  # noqa: E704
        def __isub__(self, other: Any) -> "_Tile32x32Proto": ...  # noqa: E704
        def __getitem__(self, key: Any) -> Any: ...  # noqa: E704
        def __setitem__(self, key: Any, value: Any) -> None: ...  # noqa: E704


_TILE = 32

# Field-name lookup table for direct register access in qd.static-unrolled loops. See ``_tile16._REGS`` for the
# rationale; the same trick applies here, just with 32 fields instead of 16.
_REGS = tuple(f"r{i}" for i in range(_TILE))


_tile32_cache: dict[Any, type] = {}


def _make_tile32x32(dtype=None) -> "type[_Tile32x32Proto]":
    """Create a Tile32x32 dataclass whose registers use the given scalar dtype (qd.f32 or qd.f64).

    This is an internal factory. Use ``qd.simt.Tile32x32`` (the proxy) instead.
    """
    if dtype is None:
        dtype = qd.f32
    if dtype in _tile32_cache:
        return _tile32_cache[dtype]  # pyright: ignore[reportReturnType]
    cls = _make_tile32x32_class(dtype)
    _tile32_cache[dtype] = cls
    return cls  # pyright: ignore[reportReturnType]


def _make_tile32x32_class(dtype):
    class _Tile32x32:
        """A 32x32 tile distributed one row per subgroup thread, held in 32 scalar registers.  All fields default to
        0.0 when omitted: ``Tile32x32()`` creates a zero tile."""

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
        r16: dtype
        r17: dtype
        r18: dtype
        r19: dtype
        r20: dtype
        r21: dtype
        r22: dtype
        r23: dtype
        r24: dtype
        r25: dtype
        r26: dtype
        r27: dtype
        r28: dtype
        r29: dtype
        r30: dtype
        r31: dtype

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
                # Inline cascade: with j a python-int from qd.static, only the matching branch is emitted into the AST.
                # Avoids the 32x duplication that calling _set_col(j) through the @qd.func boundary would force.
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        val = arr[row, col_start + j]
                        if j == 0:
                            self.r0 = val
                        if j == 1:
                            self.r1 = val
                        if j == 2:
                            self.r2 = val
                        if j == 3:
                            self.r3 = val
                        if j == 4:
                            self.r4 = val
                        if j == 5:
                            self.r5 = val
                        if j == 6:
                            self.r6 = val
                        if j == 7:
                            self.r7 = val
                        if j == 8:
                            self.r8 = val
                        if j == 9:
                            self.r9 = val
                        if j == 10:
                            self.r10 = val
                        if j == 11:
                            self.r11 = val
                        if j == 12:
                            self.r12 = val
                        if j == 13:
                            self.r13 = val
                        if j == 14:
                            self.r14 = val
                        if j == 15:
                            self.r15 = val
                        if j == 16:
                            self.r16 = val
                        if j == 17:
                            self.r17 = val
                        if j == 18:
                            self.r18 = val
                        if j == 19:
                            self.r19 = val
                        if j == 20:
                            self.r20 = val
                        if j == 21:
                            self.r21 = val
                        if j == 22:
                            self.r22 = val
                        if j == 23:
                            self.r23 = val
                        if j == 24:
                            self.r24 = val
                        if j == 25:
                            self.r25 = val
                        if j == 26:
                            self.r26 = val
                        if j == 27:
                            self.r27 = val
                        if j == 28:
                            self.r28 = val
                        if j == 29:
                            self.r29 = val
                        if j == 30:
                            self.r30 = val
                        if j == 31:
                            self.r31 = val

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
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        val = arr[batch, row, col_start + j]
                        if j == 0:
                            self.r0 = val
                        if j == 1:
                            self.r1 = val
                        if j == 2:
                            self.r2 = val
                        if j == 3:
                            self.r3 = val
                        if j == 4:
                            self.r4 = val
                        if j == 5:
                            self.r5 = val
                        if j == 6:
                            self.r6 = val
                        if j == 7:
                            self.r7 = val
                        if j == 8:
                            self.r8 = val
                        if j == 9:
                            self.r9 = val
                        if j == 10:
                            self.r10 = val
                        if j == 11:
                            self.r11 = val
                        if j == 12:
                            self.r12 = val
                        if j == 13:
                            self.r13 = val
                        if j == 14:
                            self.r14 = val
                        if j == 15:
                            self.r15 = val
                        if j == 16:
                            self.r16 = val
                        if j == 17:
                            self.r17 = val
                        if j == 18:
                            self.r18 = val
                        if j == 19:
                            self.r19 = val
                        if j == 20:
                            self.r20 = val
                        if j == 21:
                            self.r21 = val
                        if j == 22:
                            self.r22 = val
                        if j == 23:
                            self.r23 = val
                        if j == 24:
                            self.r24 = val
                        if j == 25:
                            self.r25 = val
                        if j == 26:
                            self.r26 = val
                        if j == 27:
                            self.r27 = val
                        if j == 28:
                            self.r28 = val
                        if j == 29:
                            self.r29 = val
                        if j == 30:
                            self.r30 = val
                        if j == 31:
                            self.r31 = val

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
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        arr[row, col_start + j] = self._r(j)

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
                for j in qd.static(range(32)):
                    if col_start + j < col_stop:
                        arr[batch, row, col_start + j] = self._r(j)

        @qd.func
        def eye_(self):
            """Set this tile to the 32x32 identity matrix.  Each thread sets its diagonal element to 1.0 and all
            others to 0.0."""
            tid = qd.simt.subgroup.invocation_id()
            for j in qd.static(range(32)):
                val = 1.0 if tid == j else 0.0
                if j == 0:
                    self.r0 = val
                if j == 1:
                    self.r1 = val
                if j == 2:
                    self.r2 = val
                if j == 3:
                    self.r3 = val
                if j == 4:
                    self.r4 = val
                if j == 5:
                    self.r5 = val
                if j == 6:
                    self.r6 = val
                if j == 7:
                    self.r7 = val
                if j == 8:
                    self.r8 = val
                if j == 9:
                    self.r9 = val
                if j == 10:
                    self.r10 = val
                if j == 11:
                    self.r11 = val
                if j == 12:
                    self.r12 = val
                if j == 13:
                    self.r13 = val
                if j == 14:
                    self.r14 = val
                if j == 15:
                    self.r15 = val
                if j == 16:
                    self.r16 = val
                if j == 17:
                    self.r17 = val
                if j == 18:
                    self.r18 = val
                if j == 19:
                    self.r19 = val
                if j == 20:
                    self.r20 = val
                if j == 21:
                    self.r21 = val
                if j == 22:
                    self.r22 = val
                if j == 23:
                    self.r23 = val
                if j == 24:
                    self.r24 = val
                if j == 25:
                    self.r25 = val
                if j == 26:
                    self.r26 = val
                if j == 27:
                    self.r27 = val
                if j == 28:
                    self.r28 = val
                if j == 29:
                    self.r29 = val
                if j == 30:
                    self.r30 = val
                if j == 31:
                    self.r31 = val

        @qd.func
        def _get_col(self, k):
            """Return the value of register (column) k."""
            val = qd.cast(0.0, dtype)
            if k == 0:
                val = self.r0
            if k == 1:
                val = self.r1
            if k == 2:
                val = self.r2
            if k == 3:
                val = self.r3
            if k == 4:
                val = self.r4
            if k == 5:
                val = self.r5
            if k == 6:
                val = self.r6
            if k == 7:
                val = self.r7
            if k == 8:
                val = self.r8
            if k == 9:
                val = self.r9
            if k == 10:
                val = self.r10
            if k == 11:
                val = self.r11
            if k == 12:
                val = self.r12
            if k == 13:
                val = self.r13
            if k == 14:
                val = self.r14
            if k == 15:
                val = self.r15
            if k == 16:
                val = self.r16
            if k == 17:
                val = self.r17
            if k == 18:
                val = self.r18
            if k == 19:
                val = self.r19
            if k == 20:
                val = self.r20
            if k == 21:
                val = self.r21
            if k == 22:
                val = self.r22
            if k == 23:
                val = self.r23
            if k == 24:
                val = self.r24
            if k == 25:
                val = self.r25
            if k == 26:
                val = self.r26
            if k == 27:
                val = self.r27
            if k == 28:
                val = self.r28
            if k == 29:
                val = self.r29
            if k == 30:
                val = self.r30
            if k == 31:
                val = self.r31
            return val

        @qd.func
        def _set_col(self, k, val):
            """Set register (column) k to val."""
            if k == 0:
                self.r0 = val
            if k == 1:
                self.r1 = val
            if k == 2:
                self.r2 = val
            if k == 3:
                self.r3 = val
            if k == 4:
                self.r4 = val
            if k == 5:
                self.r5 = val
            if k == 6:
                self.r6 = val
            if k == 7:
                self.r7 = val
            if k == 8:
                self.r8 = val
            if k == 9:
                self.r9 = val
            if k == 10:
                self.r10 = val
            if k == 11:
                self.r11 = val
            if k == 12:
                self.r12 = val
            if k == 13:
                self.r13 = val
            if k == 14:
                self.r14 = val
            if k == 15:
                self.r15 = val
            if k == 16:
                self.r16 = val
            if k == 17:
                self.r17 = val
            if k == 18:
                self.r18 = val
            if k == 19:
                self.r19 = val
            if k == 20:
                self.r20 = val
            if k == 21:
                self.r21 = val
            if k == 22:
                self.r22 = val
            if k == 23:
                self.r23 = val
            if k == 24:
                self.r24 = val
            if k == 25:
                self.r25 = val
            if k == 26:
                self.r26 = val
            if k == 27:
                self.r27 = val
            if k == 28:
                self.r28 = val
            if k == 29:
                self.r29 = val
            if k == 30:
                self.r30 = val
            if k == 31:
                self.r31 = val

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T."""
            for j in qd.static(range(32)):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(j))
                val = self._r(j) - a * bc
                if j == 0:
                    self.r0 = val
                if j == 1:
                    self.r1 = val
                if j == 2:
                    self.r2 = val
                if j == 3:
                    self.r3 = val
                if j == 4:
                    self.r4 = val
                if j == 5:
                    self.r5 = val
                if j == 6:
                    self.r6 = val
                if j == 7:
                    self.r7 = val
                if j == 8:
                    self.r8 = val
                if j == 9:
                    self.r9 = val
                if j == 10:
                    self.r10 = val
                if j == 11:
                    self.r11 = val
                if j == 12:
                    self.r12 = val
                if j == 13:
                    self.r13 = val
                if j == 14:
                    self.r14 = val
                if j == 15:
                    self.r15 = val
                if j == 16:
                    self.r16 = val
                if j == 17:
                    self.r17 = val
                if j == 18:
                    self.r18 = val
                if j == 19:
                    self.r19 = val
                if j == 20:
                    self.r20 = val
                if j == 21:
                    self.r21 = val
                if j == 22:
                    self.r22 = val
                if j == 23:
                    self.r23 = val
                if j == 24:
                    self.r24 = val
                if j == 25:
                    self.r25 = val
                if j == 26:
                    self.r26 = val
                if j == 27:
                    self.r27 = val
                if j == 28:
                    self.r28 = val
                if j == 29:
                    self.r29 = val
                if j == 30:
                    self.r30 = val
                if j == 31:
                    self.r31 = val

        @qd.func
        def cholesky_(self, eps):
            """In-place 32x32 Cholesky factorization via subgroup shuffles.

            On return, the lower triangle holds L such that A = L @ L^T.  Diagonal clamped to
            sqrt(max(value, eps)) for numerical stability.
            """
            # `k` and `j` are wrapped in qd.static so the `if k > j` predicates fold at compile time and register access
            # on the outer `k` and inner `j` collapses to a single field reference via `self._r(<py_int>)` (a thin
            # getattr wrapper) rather than a 32-deep register-indexing cascade. Writes use an inline `if k == N:
            # self.rN = ...` chain (setattr is rejected by the quadrants AST builder) which the AST transformer folds at
            # build time when `k` is a python int. The per-lane row-norm used for the diagonal update is carried in
            # `my_norm_sq`, so each diagonal step is O(1) rather than O(k). The off-diagonal `dot` is split into two
            # interleaved partial sums (`dot0`/`dot1`) so the back-to-back FMA dependency chain is cut in half,
            # exposing more instruction-level parallelism.
            tid = qd.i32(qd.simt.subgroup.invocation_id())
            my_norm_sq = qd.cast(0.0, dtype)
            for k in qd.static(range(32)):
                diag_val = qd.cast(0.0, dtype)
                if tid == k:
                    diag_val = qd.sqrt(qd.max(self._r(k) - my_norm_sq, eps))
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
                    if k == 16:
                        self.r16 = diag_val
                    if k == 17:
                        self.r17 = diag_val
                    if k == 18:
                        self.r18 = diag_val
                    if k == 19:
                        self.r19 = diag_val
                    if k == 20:
                        self.r20 = diag_val
                    if k == 21:
                        self.r21 = diag_val
                    if k == 22:
                        self.r22 = diag_val
                    if k == 23:
                        self.r23 = diag_val
                    if k == 24:
                        self.r24 = diag_val
                    if k == 25:
                        self.r25 = diag_val
                    if k == 26:
                        self.r26 = diag_val
                    if k == 27:
                        self.r27 = diag_val
                    if k == 28:
                        self.r28 = diag_val
                    if k == 29:
                        self.r29 = diag_val
                    if k == 30:
                        self.r30 = diag_val
                    if k == 31:
                        self.r31 = diag_val

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

                dot0 = qd.cast(0.0, dtype)
                dot1 = qd.cast(0.0, dtype)
                for j in qd.static(range(32)):
                    if k > j:
                        my_col = self._r(j)
                        Lkj = qd.simt.subgroup.shuffle(my_col, qd.u32(k))
                        if j % 2 == 0:
                            dot0 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                        else:
                            dot1 += Lkj * my_col  # type: ignore[reportOperatorIssue]
                dot = dot0 + dot1

                new_val = qd.cast(0.0, dtype)
                if tid > k:  # type: ignore[reportOperatorIssue]
                    new_val = (self._r(k) - dot) / diag_k  # type: ignore[reportOperatorIssue]
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
                    if k == 16:
                        self.r16 = new_val
                    if k == 17:
                        self.r17 = new_val
                    if k == 18:
                        self.r18 = new_val
                    if k == 19:
                        self.r19 = new_val
                    if k == 20:
                        self.r20 = new_val
                    if k == 21:
                        self.r21 = new_val
                    if k == 22:
                        self.r22 = new_val
                    if k == 23:
                        self.r23 = new_val
                    if k == 24:
                        self.r24 = new_val
                    if k == 25:
                        self.r25 = new_val
                    if k == 26:
                        self.r26 = new_val
                    if k == 27:
                        self.r27 = new_val
                    if k == 28:
                        self.r28 = new_val
                    if k == 29:
                        self.r29 = new_val
                    if k == 30:
                        self.r30 = new_val
                    if k == 31:
                        self.r31 = new_val
                if tid > k:  # type: ignore[reportOperatorIssue]
                    my_norm_sq += new_val * new_val

        @qd.func
        def _trsm(self, L):
            """In-place triangular solve: solve self @ L^T = B (original self).

            L is a Tile32x32 holding the lower-triangular Cholesky factor (from cholesky_).  On return, self holds
            the solution X.
            """
            for c in range(32):
                dot = qd.cast(0.0, dtype)
                for j in range(32):
                    if c > j:
                        Lkj = qd.simt.subgroup.shuffle(L._get_col(j), qd.u32(c))
                        dot += self._get_col(j) * Lkj  # type: ignore[reportOperatorIssue]

                diag_c = qd.simt.subgroup.shuffle(L._get_col(c), qd.u32(c))
                new_val = (self._get_col(c) - dot) / diag_c  # type: ignore[reportOperatorIssue]
                self._set_col(c, new_val)

        def solve_triangular_(self, B: Any, lower: bool = True) -> None:
            """Triangular solve: X @ self^T = B, storing result X in B in-place.

            self must be lower-triangular and non-singular (all diagonal elements non-zero).  Passing a singular
            matrix causes division by zero, producing inf/NaN without warning.  Only lower=True is supported.
            """
            if not lower:
                raise TypeError("Tile32x32.solve_triangular_: only lower=True is supported")
            B._trsm(self)

        def _r(self, k):
            """Direct field read by python-int index. Used at qd.static-unrolled call sites to bypass the _TILE-way
            ``_get_col(k)`` cascade: with ``k`` a python int (from ``qd.static(range(_TILE))``),
            ``getattr(self, _REGS[k])`` is evaluated by the AST transformer at build time and returns a single
            field-reference expression."""
            return getattr(self, _REGS[k])

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
                    raise TypeError(f"Tile32x32: unsupported augmented assignment op '{op}' with outer product")
            else:
                raise TypeError(f"Tile32x32: unsupported augmented assignment with {type(other)}")

    # StructType.__call__ already defaults missing args to 0, so Tile() produces a zero-initialized tile
    # without needing default values in the class definition (which @qd.dataclass doesn't support).
    result = qd.dataclass(_Tile32x32)
    result.SIZE = _TILE  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t.eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]
    return result


class _Tile32x32Proxy:
    """Proxy for dtype-at-point-of-use tile creation.

    Use as ``qd.simt.Tile32x32.zeros(dtype=qd.f32)`` inside a kernel. The dtype is resolved at kernel compilation
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
                "Tile32x32 requires a GPU backend (cuda, metal, vulkan, amdgpu). " f"Current arch is {arch}."
            )
        if dtype is None:
            dtype = impl.get_runtime().default_fp
        if dtype in _tile32_cache:
            return _tile32_cache[dtype]
        return _make_tile32x32(dtype)

    def zeros(self, *, dtype=None):
        """Zero-initialized tile."""
        return self._resolve(dtype)()

    def eye(self, *, dtype=None):
        """Identity tile (diagonal = 1, rest = 0)."""
        return self._resolve(dtype).eye()


Tile32x32Proxy = _Tile32x32Proxy()
