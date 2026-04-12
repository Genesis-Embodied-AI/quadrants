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
                if col_start < col_end:
                    self.r0 = arr[row, col_start]
                if col_start + 1 < col_end:
                    self.r1 = arr[row, col_start + 1]
                if col_start + 2 < col_end:
                    self.r2 = arr[row, col_start + 2]
                if col_start + 3 < col_end:
                    self.r3 = arr[row, col_start + 3]
                if col_start + 4 < col_end:
                    self.r4 = arr[row, col_start + 4]
                if col_start + 5 < col_end:
                    self.r5 = arr[row, col_start + 5]
                if col_start + 6 < col_end:
                    self.r6 = arr[row, col_start + 6]
                if col_start + 7 < col_end:
                    self.r7 = arr[row, col_start + 7]
                if col_start + 8 < col_end:
                    self.r8 = arr[row, col_start + 8]
                if col_start + 9 < col_end:
                    self.r9 = arr[row, col_start + 9]
                if col_start + 10 < col_end:
                    self.r10 = arr[row, col_start + 10]
                if col_start + 11 < col_end:
                    self.r11 = arr[row, col_start + 11]
                if col_start + 12 < col_end:
                    self.r12 = arr[row, col_start + 12]
                if col_start + 13 < col_end:
                    self.r13 = arr[row, col_start + 13]
                if col_start + 14 < col_end:
                    self.r14 = arr[row, col_start + 14]
                if col_start + 15 < col_end:
                    self.r15 = arr[row, col_start + 15]

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
                if col_start < col_end:
                    self.r0 = arr[batch, row, col_start]
                if col_start + 1 < col_end:
                    self.r1 = arr[batch, row, col_start + 1]
                if col_start + 2 < col_end:
                    self.r2 = arr[batch, row, col_start + 2]
                if col_start + 3 < col_end:
                    self.r3 = arr[batch, row, col_start + 3]
                if col_start + 4 < col_end:
                    self.r4 = arr[batch, row, col_start + 4]
                if col_start + 5 < col_end:
                    self.r5 = arr[batch, row, col_start + 5]
                if col_start + 6 < col_end:
                    self.r6 = arr[batch, row, col_start + 6]
                if col_start + 7 < col_end:
                    self.r7 = arr[batch, row, col_start + 7]
                if col_start + 8 < col_end:
                    self.r8 = arr[batch, row, col_start + 8]
                if col_start + 9 < col_end:
                    self.r9 = arr[batch, row, col_start + 9]
                if col_start + 10 < col_end:
                    self.r10 = arr[batch, row, col_start + 10]
                if col_start + 11 < col_end:
                    self.r11 = arr[batch, row, col_start + 11]
                if col_start + 12 < col_end:
                    self.r12 = arr[batch, row, col_start + 12]
                if col_start + 13 < col_end:
                    self.r13 = arr[batch, row, col_start + 13]
                if col_start + 14 < col_end:
                    self.r14 = arr[batch, row, col_start + 14]
                if col_start + 15 < col_end:
                    self.r15 = arr[batch, row, col_start + 15]

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
                if col_start < col_end:
                    arr[row, col_start] = self.r0
                if col_start + 1 < col_end:
                    arr[row, col_start + 1] = self.r1
                if col_start + 2 < col_end:
                    arr[row, col_start + 2] = self.r2
                if col_start + 3 < col_end:
                    arr[row, col_start + 3] = self.r3
                if col_start + 4 < col_end:
                    arr[row, col_start + 4] = self.r4
                if col_start + 5 < col_end:
                    arr[row, col_start + 5] = self.r5
                if col_start + 6 < col_end:
                    arr[row, col_start + 6] = self.r6
                if col_start + 7 < col_end:
                    arr[row, col_start + 7] = self.r7
                if col_start + 8 < col_end:
                    arr[row, col_start + 8] = self.r8
                if col_start + 9 < col_end:
                    arr[row, col_start + 9] = self.r9
                if col_start + 10 < col_end:
                    arr[row, col_start + 10] = self.r10
                if col_start + 11 < col_end:
                    arr[row, col_start + 11] = self.r11
                if col_start + 12 < col_end:
                    arr[row, col_start + 12] = self.r12
                if col_start + 13 < col_end:
                    arr[row, col_start + 13] = self.r13
                if col_start + 14 < col_end:
                    arr[row, col_start + 14] = self.r14
                if col_start + 15 < col_end:
                    arr[row, col_start + 15] = self.r15

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
                if col_start < col_end:
                    arr[batch, row, col_start] = self.r0
                if col_start + 1 < col_end:
                    arr[batch, row, col_start + 1] = self.r1
                if col_start + 2 < col_end:
                    arr[batch, row, col_start + 2] = self.r2
                if col_start + 3 < col_end:
                    arr[batch, row, col_start + 3] = self.r3
                if col_start + 4 < col_end:
                    arr[batch, row, col_start + 4] = self.r4
                if col_start + 5 < col_end:
                    arr[batch, row, col_start + 5] = self.r5
                if col_start + 6 < col_end:
                    arr[batch, row, col_start + 6] = self.r6
                if col_start + 7 < col_end:
                    arr[batch, row, col_start + 7] = self.r7
                if col_start + 8 < col_end:
                    arr[batch, row, col_start + 8] = self.r8
                if col_start + 9 < col_end:
                    arr[batch, row, col_start + 9] = self.r9
                if col_start + 10 < col_end:
                    arr[batch, row, col_start + 10] = self.r10
                if col_start + 11 < col_end:
                    arr[batch, row, col_start + 11] = self.r11
                if col_start + 12 < col_end:
                    arr[batch, row, col_start + 12] = self.r12
                if col_start + 13 < col_end:
                    arr[batch, row, col_start + 13] = self.r13
                if col_start + 14 < col_end:
                    arr[batch, row, col_start + 14] = self.r14
                if col_start + 15 < col_end:
                    arr[batch, row, col_start + 15] = self.r15

        @qd.func
        def _eye_(self):
            """Set this tile to the 16x16 identity matrix.

            Each thread sets its diagonal element to 1.0 and all others to 0.0.
            """
            tid = qd.simt.subgroup.invocation_id()
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
        def _ger_sub(self, a, b):
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
