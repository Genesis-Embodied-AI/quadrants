# type: ignore

"""
Register-resident 16x16 tile operations using subgroup (warp) shuffles.

Each tile is a 16x16 matrix distributed across 16 threads in a subgroup,
one row per thread, with each row stored in 16 scalar registers (r0-r15).
Cross-thread communication uses warp shuffles — no shared memory needed.

Operations:
    load    — load a tile row from a 2D array with column bounds checking
    store   — store a tile row to a 2D array with column bounds checking
    syr_sub — symmetric rank-1 subtract:  R -= v @ v^T
    ger_sub — general rank-1 subtract:    R -= a @ b^T
    potrf   — in-place Cholesky factorization of a 16x16 tile
    trsm    — triangular solve: given L (from potrf), solve L @ X^T = B^T
"""

import quadrants as qd

_TILE = 16


@qd.func
def load(arr, row, col0, n_cols, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """Load one row of a 16x16 tile from a 2D array with column bounds checking.

    Columns [col0, col0+16) are loaded into r0-r15. Out-of-bounds columns
    (col0+i >= n_cols) leave the corresponding register unchanged.
    """
    if col0 + 0 < n_cols: r0 = arr[row, col0 + 0]
    if col0 + 1 < n_cols: r1 = arr[row, col0 + 1]
    if col0 + 2 < n_cols: r2 = arr[row, col0 + 2]
    if col0 + 3 < n_cols: r3 = arr[row, col0 + 3]
    if col0 + 4 < n_cols: r4 = arr[row, col0 + 4]
    if col0 + 5 < n_cols: r5 = arr[row, col0 + 5]
    if col0 + 6 < n_cols: r6 = arr[row, col0 + 6]
    if col0 + 7 < n_cols: r7 = arr[row, col0 + 7]
    if col0 + 8 < n_cols: r8 = arr[row, col0 + 8]
    if col0 + 9 < n_cols: r9 = arr[row, col0 + 9]
    if col0 + 10 < n_cols: r10 = arr[row, col0 + 10]
    if col0 + 11 < n_cols: r11 = arr[row, col0 + 11]
    if col0 + 12 < n_cols: r12 = arr[row, col0 + 12]
    if col0 + 13 < n_cols: r13 = arr[row, col0 + 13]
    if col0 + 14 < n_cols: r14 = arr[row, col0 + 14]
    if col0 + 15 < n_cols: r15 = arr[row, col0 + 15]
    return r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15


@qd.func
def store(arr, row, col0, n_cols, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """Store one row of a 16x16 tile to a 2D array with column bounds checking.

    Registers r0-r15 are written to columns [col0, col0+16). Out-of-bounds
    columns (col0+i >= n_cols) are skipped.
    """
    if col0 + 0 < n_cols: arr[row, col0 + 0] = r0
    if col0 + 1 < n_cols: arr[row, col0 + 1] = r1
    if col0 + 2 < n_cols: arr[row, col0 + 2] = r2
    if col0 + 3 < n_cols: arr[row, col0 + 3] = r3
    if col0 + 4 < n_cols: arr[row, col0 + 4] = r4
    if col0 + 5 < n_cols: arr[row, col0 + 5] = r5
    if col0 + 6 < n_cols: arr[row, col0 + 6] = r6
    if col0 + 7 < n_cols: arr[row, col0 + 7] = r7
    if col0 + 8 < n_cols: arr[row, col0 + 8] = r8
    if col0 + 9 < n_cols: arr[row, col0 + 9] = r9
    if col0 + 10 < n_cols: arr[row, col0 + 10] = r10
    if col0 + 11 < n_cols: arr[row, col0 + 11] = r11
    if col0 + 12 < n_cols: arr[row, col0 + 12] = r12
    if col0 + 13 < n_cols: arr[row, col0 + 13] = r13
    if col0 + 14 < n_cols: arr[row, col0 + 14] = r14
    if col0 + 15 < n_cols: arr[row, col0 + 15] = r15


@qd.func
def syr_sub(v, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """Symmetric rank-1 subtract: R -= v @ v^T via subgroup shuffles.

    Each thread holds one element of column vector v and one row of tile R.
    After the call, R[tid, c] -= v[tid] * v[c] for all c in [0, 16).
    """
    vc = qd.simt.subgroup.shuffle(v, qd.u32(0));  r0 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(1));  r1 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(2));  r2 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(3));  r3 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(4));  r4 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(5));  r5 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(6));  r6 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(7));  r7 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(8));  r8 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(9));  r9 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(10)); r10 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(11)); r11 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(12)); r12 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(13)); r13 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(14)); r14 -= v * vc
    vc = qd.simt.subgroup.shuffle(v, qd.u32(15)); r15 -= v * vc
    return r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15


@qd.func
def ger_sub(a, b, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """General rank-1 subtract: R -= a @ b^T via subgroup shuffles.

    Each thread holds one element of vectors a and b, and one row of tile R.
    After the call, R[tid, c] -= a[tid] * b[c] for all c in [0, 16).
    """
    bc = qd.simt.subgroup.shuffle(b, qd.u32(0));  r0 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(1));  r1 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(2));  r2 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(3));  r3 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(4));  r4 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(5));  r5 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(6));  r6 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(7));  r7 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(8));  r8 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(9));  r9 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(10)); r10 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(11)); r11 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(12)); r12 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(13)); r13 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(14)); r14 -= a * bc
    bc = qd.simt.subgroup.shuffle(b, qd.u32(15)); r15 -= a * bc
    return r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15


@qd.func
def potrf(tid, eps, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15):
    """In-register 16x16 Cholesky factorization (POTRF) via subgroup shuffles.

    Factorizes a symmetric positive-definite 16x16 tile in-place: on return,
    the lower triangle of R holds L such that A = L @ L^T. The diagonal is
    clamped to sqrt(max(value, eps)) for numerical stability.

    Args:
        tid: subgroup invocation index (0-15), identifying which row this thread owns.
        eps: minimum diagonal value (clamping threshold for numerical stability).
        r0-r15: the tile row owned by this thread (modified in-place).
    """
    for k in range(_TILE):
        diag_val = 0.0
        if tid == k:
            s = 0.0
            if k > 0: s += r0 * r0
            if k > 1: s += r1 * r1
            if k > 2: s += r2 * r2
            if k > 3: s += r3 * r3
            if k > 4: s += r4 * r4
            if k > 5: s += r5 * r5
            if k > 6: s += r6 * r6
            if k > 7: s += r7 * r7
            if k > 8: s += r8 * r8
            if k > 9: s += r9 * r9
            if k > 10: s += r10 * r10
            if k > 11: s += r11 * r11
            if k > 12: s += r12 * r12
            if k > 13: s += r13 * r13
            if k > 14: s += r14 * r14
            cur = 0.0
            if k == 0: cur = r0
            if k == 1: cur = r1
            if k == 2: cur = r2
            if k == 3: cur = r3
            if k == 4: cur = r4
            if k == 5: cur = r5
            if k == 6: cur = r6
            if k == 7: cur = r7
            if k == 8: cur = r8
            if k == 9: cur = r9
            if k == 10: cur = r10
            if k == 11: cur = r11
            if k == 12: cur = r12
            if k == 13: cur = r13
            if k == 14: cur = r14
            if k == 15: cur = r15
            diag_val = qd.sqrt(qd.max(cur - s, eps))
            if k == 0: r0 = diag_val
            if k == 1: r1 = diag_val
            if k == 2: r2 = diag_val
            if k == 3: r3 = diag_val
            if k == 4: r4 = diag_val
            if k == 5: r5 = diag_val
            if k == 6: r6 = diag_val
            if k == 7: r7 = diag_val
            if k == 8: r8 = diag_val
            if k == 9: r9 = diag_val
            if k == 10: r10 = diag_val
            if k == 11: r11 = diag_val
            if k == 12: r12 = diag_val
            if k == 13: r13 = diag_val
            if k == 14: r14 = diag_val
            if k == 15: r15 = diag_val

        diag_k = qd.simt.subgroup.shuffle(diag_val, qd.u32(k))

        dot = 0.0
        if k > 0:
            Lkj = qd.simt.subgroup.shuffle(r0, qd.u32(k)); dot += Lkj * r0
        if k > 1:
            Lkj = qd.simt.subgroup.shuffle(r1, qd.u32(k)); dot += Lkj * r1
        if k > 2:
            Lkj = qd.simt.subgroup.shuffle(r2, qd.u32(k)); dot += Lkj * r2
        if k > 3:
            Lkj = qd.simt.subgroup.shuffle(r3, qd.u32(k)); dot += Lkj * r3
        if k > 4:
            Lkj = qd.simt.subgroup.shuffle(r4, qd.u32(k)); dot += Lkj * r4
        if k > 5:
            Lkj = qd.simt.subgroup.shuffle(r5, qd.u32(k)); dot += Lkj * r5
        if k > 6:
            Lkj = qd.simt.subgroup.shuffle(r6, qd.u32(k)); dot += Lkj * r6
        if k > 7:
            Lkj = qd.simt.subgroup.shuffle(r7, qd.u32(k)); dot += Lkj * r7
        if k > 8:
            Lkj = qd.simt.subgroup.shuffle(r8, qd.u32(k)); dot += Lkj * r8
        if k > 9:
            Lkj = qd.simt.subgroup.shuffle(r9, qd.u32(k)); dot += Lkj * r9
        if k > 10:
            Lkj = qd.simt.subgroup.shuffle(r10, qd.u32(k)); dot += Lkj * r10
        if k > 11:
            Lkj = qd.simt.subgroup.shuffle(r11, qd.u32(k)); dot += Lkj * r11
        if k > 12:
            Lkj = qd.simt.subgroup.shuffle(r12, qd.u32(k)); dot += Lkj * r12
        if k > 13:
            Lkj = qd.simt.subgroup.shuffle(r13, qd.u32(k)); dot += Lkj * r13
        if k > 14:
            Lkj = qd.simt.subgroup.shuffle(r14, qd.u32(k)); dot += Lkj * r14

        if tid > k:
            cur = 0.0
            if k == 0: cur = r0
            if k == 1: cur = r1
            if k == 2: cur = r2
            if k == 3: cur = r3
            if k == 4: cur = r4
            if k == 5: cur = r5
            if k == 6: cur = r6
            if k == 7: cur = r7
            if k == 8: cur = r8
            if k == 9: cur = r9
            if k == 10: cur = r10
            if k == 11: cur = r11
            if k == 12: cur = r12
            if k == 13: cur = r13
            if k == 14: cur = r14
            if k == 15: cur = r15
            new_val = (cur - dot) / diag_k
            if k == 0: r0 = new_val
            if k == 1: r1 = new_val
            if k == 2: r2 = new_val
            if k == 3: r3 = new_val
            if k == 4: r4 = new_val
            if k == 5: r5 = new_val
            if k == 6: r6 = new_val
            if k == 7: r7 = new_val
            if k == 8: r8 = new_val
            if k == 9: r9 = new_val
            if k == 10: r10 = new_val
            if k == 11: r11 = new_val
            if k == 12: r12 = new_val
            if k == 13: r13 = new_val
            if k == 14: r14 = new_val
            if k == 15: r15 = new_val

    return r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15


@qd.func
def trsm(
    r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15,
):
    """In-register 16x16 triangular solve (TRSM): solve L @ X^T = B^T.

    r0-r15 hold the lower-triangular Cholesky factor L (from potrf, read-only).
    q0-q15 hold the right-hand side B on entry, and the solution X on exit.
    """
    for c in range(_TILE):
        dot = 0.0
        if c > 0:
            Lkj = qd.simt.subgroup.shuffle(r0, qd.u32(c)); dot += q0 * Lkj
        if c > 1:
            Lkj = qd.simt.subgroup.shuffle(r1, qd.u32(c)); dot += q1 * Lkj
        if c > 2:
            Lkj = qd.simt.subgroup.shuffle(r2, qd.u32(c)); dot += q2 * Lkj
        if c > 3:
            Lkj = qd.simt.subgroup.shuffle(r3, qd.u32(c)); dot += q3 * Lkj
        if c > 4:
            Lkj = qd.simt.subgroup.shuffle(r4, qd.u32(c)); dot += q4 * Lkj
        if c > 5:
            Lkj = qd.simt.subgroup.shuffle(r5, qd.u32(c)); dot += q5 * Lkj
        if c > 6:
            Lkj = qd.simt.subgroup.shuffle(r6, qd.u32(c)); dot += q6 * Lkj
        if c > 7:
            Lkj = qd.simt.subgroup.shuffle(r7, qd.u32(c)); dot += q7 * Lkj
        if c > 8:
            Lkj = qd.simt.subgroup.shuffle(r8, qd.u32(c)); dot += q8 * Lkj
        if c > 9:
            Lkj = qd.simt.subgroup.shuffle(r9, qd.u32(c)); dot += q9 * Lkj
        if c > 10:
            Lkj = qd.simt.subgroup.shuffle(r10, qd.u32(c)); dot += q10 * Lkj
        if c > 11:
            Lkj = qd.simt.subgroup.shuffle(r11, qd.u32(c)); dot += q11 * Lkj
        if c > 12:
            Lkj = qd.simt.subgroup.shuffle(r12, qd.u32(c)); dot += q12 * Lkj
        if c > 13:
            Lkj = qd.simt.subgroup.shuffle(r13, qd.u32(c)); dot += q13 * Lkj
        if c > 14:
            Lkj = qd.simt.subgroup.shuffle(r14, qd.u32(c)); dot += q14 * Lkj

        diag_reg = 0.0
        if c == 0: diag_reg = r0
        if c == 1: diag_reg = r1
        if c == 2: diag_reg = r2
        if c == 3: diag_reg = r3
        if c == 4: diag_reg = r4
        if c == 5: diag_reg = r5
        if c == 6: diag_reg = r6
        if c == 7: diag_reg = r7
        if c == 8: diag_reg = r8
        if c == 9: diag_reg = r9
        if c == 10: diag_reg = r10
        if c == 11: diag_reg = r11
        if c == 12: diag_reg = r12
        if c == 13: diag_reg = r13
        if c == 14: diag_reg = r14
        if c == 15: diag_reg = r15
        diag_c = qd.simt.subgroup.shuffle(diag_reg, qd.u32(c))

        cur = 0.0
        if c == 0: cur = q0
        if c == 1: cur = q1
        if c == 2: cur = q2
        if c == 3: cur = q3
        if c == 4: cur = q4
        if c == 5: cur = q5
        if c == 6: cur = q6
        if c == 7: cur = q7
        if c == 8: cur = q8
        if c == 9: cur = q9
        if c == 10: cur = q10
        if c == 11: cur = q11
        if c == 12: cur = q12
        if c == 13: cur = q13
        if c == 14: cur = q14
        if c == 15: cur = q15

        new_val = (cur - dot) / diag_c

        if c == 0: q0 = new_val
        if c == 1: q1 = new_val
        if c == 2: q2 = new_val
        if c == 3: q3 = new_val
        if c == 4: q4 = new_val
        if c == 5: q5 = new_val
        if c == 6: q6 = new_val
        if c == 7: q7 = new_val
        if c == 8: q8 = new_val
        if c == 9: q9 = new_val
        if c == 10: q10 = new_val
        if c == 11: q11 = new_val
        if c == 12: q12 = new_val
        if c == 13: q13 = new_val
        if c == 14: q14 = new_val
        if c == 15: q15 = new_val

    return q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15
