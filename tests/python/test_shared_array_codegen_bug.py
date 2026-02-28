"""Regression test for a code generation bug where changing operand order in
dead code inside a kernel with SharedArray causes CUDA_ERROR_ILLEGAL_ADDRESS.
"""

import numpy as np
import quadrants as qd


BLOCK_DIM = 64
MAX_DOFS = 64
MAX_CONSTRAINTS = 32
N_DOFS = 36
_B = 4096


@qd.func
def func_hessian_add(
    nt_H: qd.template(),
    mass_mat: qd.template(),
    n_constraints: qd.template(),
    improved: qd.template(),
):
    n_lower_tri = N_DOFS * (N_DOFS + 1) // 2

    qd.loop_config(block_dim=BLOCK_DIM)
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        if i_b >= _B:
            continue
        if n_constraints[i_b] == 0 or not improved[i_b]:
            continue

        jac_row = qd.simt.block.SharedArray((MAX_CONSTRAINTS, MAX_DOFS), qd.f32)
        efc_D = qd.simt.block.SharedArray((MAX_CONSTRAINTS,), qd.f32)

        n_c = n_constraints[i_b]
        n_conts_tile = qd.min(MAX_CONSTRAINTS, n_c)

        i_c_ = tid
        while i_c_ < n_conts_tile:
            efc_D[i_c_] = qd.f32(1.0)
            i_c_ = i_c_ + BLOCK_DIM

        i_c_ = tid
        while i_c_ < n_conts_tile:
            for i_d_ in range(N_DOFS):
                jac_row[i_c_, i_d_] = qd.f32(1.0)
            i_c_ = i_c_ + BLOCK_DIM
        qd.simt.block.sync()

        pid = tid
        numel = N_DOFS * N_DOFS
        while pid < numel:
            i_d1 = pid // N_DOFS
            i_d2 = pid % N_DOFS
            if i_d1 >= i_d2:
                coef = mass_mat[i_d1, i_d2, i_b]
                for j_c_ in range(n_conts_tile):
                    coef = coef + jac_row[j_c_, i_d1] * jac_row[j_c_, i_d2] * efc_D[j_c_]
                nt_H[i_b, i_d1, i_d2] = coef
            pid = pid + BLOCK_DIM
        qd.simt.block.sync()

        if n_c == 0:
            i_pair = tid
            while i_pair < n_lower_tri:
                i_d1 = qd.cast(
                    qd.floor((-1.0 + qd.sqrt(1.0 + 8.0 * i_pair)) / 2.0), qd.i32
                )
                i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
                nt_H[i_b, i_d1, i_d2] = mass_mat[i_d1, i_d2, i_b]
                i_pair = i_pair + BLOCK_DIM


@qd.func
def func_cholesky(
    nt_H: qd.template(),
    n_constraints: qd.template(),
    improved: qd.template(),
):
    n_lower_tri = N_DOFS * (N_DOFS + 1) // 2
    EPS = 1e-10

    qd.loop_config(block_dim=BLOCK_DIM)
    for i in range(_B * BLOCK_DIM):
        tid = i % BLOCK_DIM
        i_b = i // BLOCK_DIM
        if i_b >= _B:
            continue
        if n_constraints[i_b] == 0 or not improved[i_b]:
            continue

        H = qd.simt.block.SharedArray((MAX_DOFS, MAX_DOFS + 1), qd.f32)

        i_pair = tid
        while i_pair < n_lower_tri:
            i_d1 = qd.cast((qd.sqrt(8 * i_pair + 1) - 1) // 2, qd.i32)
            i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
            H[i_d1, i_d2] = nt_H[i_b, i_d1, i_d2]
            i_pair = i_pair + BLOCK_DIM
        qd.simt.block.sync()

        for i_d in range(N_DOFS):
            if tid == 0:
                tmp = H[i_d, i_d]
                for j_d in range(i_d):
                    tmp = tmp - H[i_d, j_d] ** 2
                H[i_d, i_d] = qd.sqrt(qd.max(tmp, EPS))
            qd.simt.block.sync()

            inv_diag = 1.0 / H[i_d, i_d]
            j_d = i_d + 1 + tid
            while j_d < N_DOFS:
                dot = qd.f32(0.0)
                for k_d in range(i_d):
                    dot = dot + H[j_d, k_d] * H[i_d, k_d]
                H[j_d, i_d] = (H[j_d, i_d] - dot) * inv_diag
                j_d = j_d + BLOCK_DIM
            qd.simt.block.sync()

        i_pair = tid
        while i_pair < n_lower_tri:
            i_d1 = qd.cast((qd.sqrt(8 * i_pair + 1) - 1) // 2, qd.i32)
            i_d2 = i_pair - i_d1 * (i_d1 + 1) // 2
            nt_H[i_b, i_d1, i_d2] = H[i_d1, i_d2]
            i_pair = i_pair + BLOCK_DIM


def test_combined():
    """Both functions inlined into one kernel — crashes."""
    qd.init(arch=qd.cuda)

    nt_H = qd.field(qd.f32, shape=(_B, N_DOFS, N_DOFS))
    mass_mat = qd.field(qd.f32, shape=(N_DOFS, N_DOFS, _B))
    n_constraints = qd.field(qd.i32, shape=(_B,))
    improved = qd.field(qd.i32, shape=(_B,))
    n_constraints.from_numpy(np.full(_B, 10, dtype=np.int32))
    improved.from_numpy(np.ones(_B, dtype=np.int32))

    @qd.kernel
    def kernel():
        func_hessian_add(nt_H, mass_mat, n_constraints, improved)
        func_cholesky(nt_H, n_constraints, improved)

    kernel()
    qd.sync()
    qd.reset()
