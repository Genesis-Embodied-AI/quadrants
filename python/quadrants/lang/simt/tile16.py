# type: ignore

"""
Internal implementation of register-resident 16x16 tile operations.

Everything in this module is private — the public API is provided by
``_Tile16x16Proxy`` (exposed as ``qd.simt.Tile16x16``), which wraps
the raw tile class with dtype-at-point-of-use and slice syntax.

The only public attribute on the raw tile is ``.SIZE`` (= 16), which is
re-exported by the proxy.

Each tile is a 16x16 matrix distributed across 16 threads in a subgroup,
one row per thread, with each row stored in 16 scalar registers (r0-r15).
Cross-thread communication uses warp shuffles — no shared memory needed.

The thread's lane index (tid) is obtained internally via subgroup.invocation_id(),
so callers never need to pass it. Load/store methods take a row offset (row0);
each thread accesses row = row0 + tid.
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
    load/store/_eye_ methods.
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
        def _load(self, arr: qd.template(), row0, col0, n_cols, row_stop):
            """Load from a 2D array with row and column bounds checking.

            Each thread loads arr[row0 + tid, col0:col0+n_cols].
            Threads where row0 + tid >= row_stop skip the load (tile row unchanged).
            """
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_stop:
                arr_n_cols = arr.shape[1]
                if arr_n_cols < n_cols:
                    n_cols = arr_n_cols
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
        def _load3d(self, arr: qd.template(), i0, row0, col0, n_cols, row_stop):
            """Load from a 3D array with row and column bounds checking.

            Each thread loads arr[i0, row0+tid, col0:col0+n_cols].
            Threads where row0 + tid >= row_stop skip the load (tile row unchanged).
            """
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_stop:
                arr_n_cols = arr.shape[2]
                if arr_n_cols < n_cols:
                    n_cols = arr_n_cols
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
        def _store(self, arr: qd.template(), row0, col0, n_cols, row_stop):
            """Store to a 2D array with row and column bounds checking.

            Each thread stores to arr[row0 + tid, col0:col0+n_cols].
            Threads where row0 + tid >= row_stop skip the store.
            """
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_stop:
                arr_n_cols = arr.shape[1]
                if arr_n_cols < n_cols:
                    n_cols = arr_n_cols
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
        def _store3d(self, arr: qd.template(), i0, row0, col0, n_cols, row_stop):
            """Store to a 3D array with row and column bounds checking.

            Each thread stores to arr[i0, row0+tid, col0:col0+n_cols].
            Threads where row0 + tid >= row_stop skip the store.
            """
            row = row0 + qd.i32(qd.simt.subgroup.invocation_id())
            if row < row_stop:
                arr_n_cols = arr.shape[2]
                if arr_n_cols < n_cols:
                    n_cols = arr_n_cols
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
    result.SIZE = _TILE
    result._quadrants_internal = True
    result.zeros = result

    @qd.func
    def _eye():
        t = result()
        t._eye_()
        return t

    result.eye = _eye
    return result
