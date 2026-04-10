# pyright: reportInvalidTypeForm=false
"""
Internal implementation of register-resident 16x16 tile operations.

Everything in this module is private to keep the user-facing API simple.
The public API will be added in later PRs.

Each tile is a 16x16 matrix distributed across 16 threads in a subgroup,
one row per thread, with each row stored in 16 scalar registers (r0-r15).
Cross-thread communication uses warp shuffles — no shared memory needed.

The thread's lane index (tid) is obtained internally via
``subgroup.invocation_id()``, so callers never need to pass it.
"""

import quadrants as qd

_TILE = 16


# =============================================================================
# Tile16x16 factory — creates a Tile16x16 class for the given scalar dtype
# =============================================================================

_tile16_cache = {}


def _make_tile16x16(dtype=qd.f32):
    """Create a Tile16x16 dataclass whose registers use the given scalar dtype (qd.f32 or qd.f64).

    Returns a @qd.dataclass type with 16 fields (r0–r15), zeros/eye factories, and
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
        def _load(self, arr: qd.template(), row_start, col_start, col_end, row_end):
            """Load from a 2D array within [row_start, row_end) x [col_start, col_end).

            Each thread loads arr[row_start + tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the load (tile row unchanged).
            """
            row = row_start + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_end:
                arr_col_end = arr.shape[1]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                if col_start + 0 < col_end:
                    self.r0 = arr[row, col_start + 0]
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
        def _load3d(self, arr: qd.template(), batch, row_start, col_start, col_end, row_end):
            """Load from a 3D array within [row_start, row_end) x [col_start, col_end).

            Each thread loads arr[batch, row_start+tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the load (tile row unchanged).
            """
            row = row_start + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_end:
                arr_col_end = arr.shape[2]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                if col_start + 0 < col_end:
                    self.r0 = arr[batch, row, col_start + 0]
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
        def _store(self, arr: qd.template(), row_start, col_start, col_end, row_end):
            """Store to a 2D array within [row_start, row_end) x [col_start, col_end).

            Each thread stores to arr[row_start + tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the store.
            """
            row = row_start + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_end:
                arr_col_end = arr.shape[1]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                if col_start + 0 < col_end:
                    arr[row, col_start + 0] = self.r0
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
        def _store3d(self, arr: qd.template(), batch, row_start, col_start, col_end, row_end):
            """Store to a 3D array within [row_start, row_end) x [col_start, col_end).

            Each thread stores to arr[batch, row_start+tid, col_start:col_end].
            Threads where row_start + tid >= row_end skip the store.
            """
            row = row_start + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_end:
                arr_col_end = arr.shape[2]
                if arr_col_end < col_end:
                    col_end = arr_col_end
                if col_start + 0 < col_end:
                    arr[batch, row, col_start + 0] = self.r0
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

    # StructType.__call__ already defaults missing args to 0, so Tile()
    # produces a zero-initialized tile without needing default values in the
    # class definition (which @qd.dataclass doesn't support).
    result = qd.dataclass(_Tile16x16)
    result.SIZE = _TILE  # type: ignore[reportAttributeAccessIssue]
    result._quadrants_internal = True  # type: ignore[reportAttributeAccessIssue]
    result.zeros = result  # type: ignore[reportAttributeAccessIssue]

    @qd.func
    def _eye():
        t = result()
        t._eye_()  # type: ignore[reportAttributeAccessIssue]
        return t

    result.eye = _eye  # type: ignore[reportAttributeAccessIssue]
    return result
