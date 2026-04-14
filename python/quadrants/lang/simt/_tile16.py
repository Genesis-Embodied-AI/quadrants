# pyright: reportInvalidTypeForm=false

"""
Internal implementation of register-resident 16x16 tile operations.

Everything in this module is private to keep the user-facing API simple.
The public API will be added in later PRs.

Each tile is a 16x16 matrix distributed across 16 threads in a subgroup,
one row per thread, with each row stored in 16 scalar registers (r0-r15).
Cross-thread communication uses warp shuffles -- no shared memory needed.

The thread's lane index (tid) is obtained internally via
``subgroup.invocation_id()``, so callers never need to pass it.
"""

import quadrants as qd

_TILE = 16

_tile16_cache = {}


def _make_tile16x16(dtype=qd.f32):
    """Create a Tile16x16 dataclass whose registers use the given scalar dtype (qd.f32 or qd.f64).

    Returns a qd.dataclass type with 16 fields (r0-r15), zeros/eye factories, and
    _load/_store/_eye_ methods.
    """
    if dtype in _tile16_cache:
        return _tile16_cache[dtype]
    cls = _make_tile16x16_class(dtype)
    _tile16_cache[dtype] = cls
    return cls


def _make_tile16x16_class(dtype):
    class _Tile16x16:
        """A 16x16 tile distributed one row per subgroup thread, held in 16 scalar registers.

        All fields default to 0.0 when omitted: ``Tile16x16()`` creates a zero tile.
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
        def _load(self, arr: qd.template(), row_start, row_end, col_start, col_end):
            """Load from a 2D array within [row_start, row_end) x [col_start, col_end).

            Each thread loads arr[row_start + tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the load (tile row unchanged).
            """
            arr_row_end = arr.shape[0]
            if arr_row_end < row_end:
                row_end = arr_row_end
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_end:
                arr_col_end = arr.shape[1]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                for j in range(_TILE):
                    if col_start + j < col_end:
                        self._set_col(j, arr[row, col_start + j])

        @qd.func
        def _load3d(self, arr: qd.template(), batch, row_start, row_end, col_start, col_end):
            """Load from a 3D array within [row_start, row_end) x [col_start, col_end).

            Each thread loads arr[batch, row_start+tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the load (tile row unchanged).
            """
            arr_row_end = arr.shape[1]
            if arr_row_end < row_end:
                row_end = arr_row_end
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_end:
                arr_col_end = arr.shape[2]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                for j in range(_TILE):
                    if col_start + j < col_end:
                        self._set_col(j, arr[batch, row, col_start + j])

        @qd.func
        def _store(self, arr: qd.template(), row_start, row_end, col_start, col_end):
            """Store to a 2D array within [row_start, row_end) x [col_start, col_end).

            Each thread stores to arr[row_start + tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the store.
            """
            arr_row_end = arr.shape[0]
            if arr_row_end < row_end:
                row_end = arr_row_end
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_end:
                arr_col_end = arr.shape[1]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                for j in range(_TILE):
                    if col_start + j < col_end:
                        arr[row, col_start + j] = self._get_col(j)

        @qd.func
        def _store3d(self, arr: qd.template(), batch, row_start, row_end, col_start, col_end):
            """Store to a 3D array within [row_start, row_end) x [col_start, col_end).

            Each thread stores to arr[batch, row_start+tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the store.
            """
            arr_row_end = arr.shape[1]
            if arr_row_end < row_end:
                row_end = arr_row_end
            row = row_start + qd.simt.subgroup.invocation_id()
            if row < row_end:
                arr_col_end = arr.shape[2]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                for j in range(_TILE):
                    if col_start + j < col_end:
                        arr[batch, row, col_start + j] = self._get_col(j)

        @qd.func
        def _eye_(self):
            """Set this tile to the 16x16 identity matrix.

            Each thread sets its diagonal element to 1.0 and all others to 0.0.
            """
            tid = qd.simt.subgroup.invocation_id()
            for j in range(_TILE):
                self._set_col(j, 0.0)
            for j in range(_TILE):
                if tid == j:
                    self._set_col(j, 1.0)

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

        @qd.func
        def _ger_sub(self, a, b):
            """General rank-1 subtract in-place: self -= a @ b^T."""
            for j in range(_TILE):
                bc = qd.simt.subgroup.shuffle(b, qd.u32(j))
                self._set_col(j, self._get_col(j) - a * bc)

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
                    for j in range(_TILE):
                        if k > j:
                            c = self._get_col(j)
                            s += c * c
                    diag_val = qd.sqrt(qd.max(self._get_col(k) - s, eps))
                    self._set_col(k, diag_val)

                diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

                dot = qd.cast(0.0, dtype)
                for j in range(_TILE):
                    if k > j:
                        my_col = self._get_col(j)
                        Lkj = qd.simt.subgroup.shuffle(my_col, qd.u32(k))
                        dot += Lkj * my_col  # type: ignore[reportOperatorIssue]

                if tid > k:  # type: ignore[reportOperatorIssue]
                    new_val = (self._get_col(k) - dot) / diag_k  # type: ignore[reportOperatorIssue]
                    self._set_col(k, new_val)

        @qd.func
        def _trsm(self, L):
            """In-place triangular solve: solve self @ L^T = B (original self).

            L is a Tile16x16 holding the lower-triangular Cholesky factor (from cholesky_).
            On return, self holds the solution X.
            """
            for c in range(_TILE):
                dot = qd.cast(0.0, dtype)
                for j in range(_TILE):
                    if c > j:
                        Lkj = qd.simt.subgroup.shuffle(L._get_col(j), qd.u32(c))
                        dot += self._get_col(j) * Lkj  # type: ignore[reportOperatorIssue]

                diag_c = qd.simt.subgroup.shuffle(L._get_col(c), qd.u32(c))
                new_val = (self._get_col(c) - dot) / diag_c  # type: ignore[reportOperatorIssue]
                self._set_col(c, new_val)

        def solve_triangular_(self, B, lower=True):
            """Triangular solve: X @ self^T = B, storing result X in B in-place.

            self must be lower-triangular (e.g. from cholesky_()).
            Only lower=True is supported.
            """
            if not lower:
                raise TypeError("Tile16x16.solve_triangular_: only lower=True is supported")
            B._trsm(self)

    # StructType.__call__ already defaults missing args to 0, so Tile()
    # produces a zero-initialized tile without needing default values in the
    # class definition (which @qd.dataclass doesn't support).
    result = qd.dataclass(_Tile16x16)
    result.SIZE = _TILE  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t._eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]
    return result
