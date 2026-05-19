#!/usr/bin/env python3
"""Benchmark NxN blocked Cholesky factorization using Tile16x16.

Three kernels compared:

1. Baseline: scalar Cholesky-Crout, 64 threads, shared memory, 2*N+1 sequential syncs. Thread 0 computes each
   diagonal, remaining threads parallelize off-diagonal updates.

2. Blocked: ceil(N/16) x ceil(N/16) grid of 16x16 tiles, 16 threads, shared memory, scalar Crout for diagonal
   blocks. Same blocking structure as Tile16x16 but all data lives in shared memory with block.sync() between
   every step.

3. Tile16x16: same blocked structure but fully register-resident via Tile16x16. No shared memory, zero syncs.
   Prior tiles read from global memory (L2).

Results on RTX PRO 6000 Blackwell, 4096 environments, N=92, f32:

    Kernel                                        Threads  Time (us)  vs baseline
    baseline (scalar Crout, shared mem)                64       2766        1.00x
    blocked  (scalar Crout, shared mem)                16       2556        1.08x
    tile16   (Tile16x16, no shared memory)             16        533        5.19x

Usage:
    python misc/demos/cholesky_blocked.py [--n N] [--n-envs N_ENVS] [--num-warmup WARMUP] [--num-iters ITERS]
"""

import argparse
import time

import numpy as np

import quadrants as qd


def _parse_args():
    p = argparse.ArgumentParser(
        description="Blocked Cholesky NxN benchmark (3 kernels: baseline / blocked / tile16).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n", type=int, default=92, help="Matrix dimension N (NxN SPD).")
    p.add_argument("--n-envs", type=int, default=4096, help="Number of independent environments.")
    p.add_argument("--num-warmup", type=int, default=50, help="Warmup iterations per kernel.")
    p.add_argument("--num-iters", type=int, default=200, help="Timed iterations per kernel.")
    return p.parse_args()


_args = _parse_args()
N = _args.n
TILE = 16
N_BLOCKS = (N + TILE - 1) // TILE
N_PADDED = N_BLOCKS * TILE  # rounded up for blocked kernel SharedArrays
N_ENVS = _args.n_envs
WARMUP = _args.num_warmup
ITERS = _args.num_iters

qd.init(arch=qd.gpu)

A_field = qd.field(dtype=qd.f32, shape=(N_ENVS, N, N))
L_baseline_field = qd.field(dtype=qd.f32, shape=(N_ENVS, N, N))
L_blocked_field = qd.field(dtype=qd.f32, shape=(N_ENVS, N, N))
L_tile16_field = qd.field(dtype=qd.f32, shape=(N_ENVS, N, N))


def make_spd_matrices(n_envs, n):
    rng = np.random.default_rng(42)
    R = rng.standard_normal((n_envs, n, n)).astype(np.float32)
    A = np.einsum("bij,bik->bjk", R, R) + np.eye(n, dtype=np.float32)
    return A


# ---------------------------------------------------------------------------
# Kernel 1: scalar Cholesky-Crout (64 threads, shared memory)
# ---------------------------------------------------------------------------


@qd.kernel
def cholesky_baseline():
    qd.loop_config(name="chol_baseline", block_dim=64)
    for idx in range(N_ENVS * 64):
        tid = idx % 64
        env = idx // 64

        H = qd.simt.block.SharedArray((N, N + 1), qd.f32)

        # Load lower triangle into shared memory
        for row in range(N):
            if row % 64 == tid:
                for col in range(row + 1):
                    H[row, col] = A_field[env, row, col]
        qd.simt.block.sync()

        for i_d in range(N):
            # Thread 0 computes diagonal: sqrt(A_ii - sum(L_ij^2))
            if tid == 0:
                tmp = H[i_d, i_d]
                for j_d in range(i_d):
                    tmp -= H[i_d, j_d] * H[i_d, j_d]
                H[i_d, i_d] = qd.sqrt(qd.max(tmp, qd.f32(1e-12)))
            qd.simt.block.sync()

            # All threads update off-diagonal entries in parallel
            inv_diag = qd.f32(1.0) / H[i_d, i_d]
            j_d = i_d + 1 + tid
            while j_d < N:
                dot = qd.f32(0.0)
                for k_d in range(i_d):
                    dot += H[j_d, k_d] * H[i_d, k_d]
                H[j_d, i_d] = (H[j_d, i_d] - dot) * inv_diag
                j_d += 64
            qd.simt.block.sync()

        # Write result to global memory
        for row in range(N):
            if row % 64 == tid:
                for col in range(row + 1):
                    L_baseline_field[env, row, col] = H[row, col]


# ---------------------------------------------------------------------------
# Kernel 2: blocked Cholesky (16 threads, shared memory, scalar Crout)
# ---------------------------------------------------------------------------


@qd.kernel
def cholesky_blocked():
    qd.loop_config(name="chol_blocked", block_dim=TILE)
    for idx in range(N_ENVS * TILE):
        tid = idx % TILE
        env = idx // TILE

        H = qd.simt.block.SharedArray((N_PADDED, N_PADDED + 1), qd.f32)

        # Load lower triangle into shared memory
        for row in range(N):
            c = tid
            while c <= row:
                H[row, c] = A_field[env, row, c]
                c += TILE
        qd.simt.block.sync()

        for kb in range(N_BLOCKS):
            k0 = kb * TILE

            # Subtract prior-block contributions from diagonal block
            for r in range(TILE):
                c = tid
                if c <= r:
                    s = qd.f32(0.0)
                    for jb in range(kb):
                        j0 = jb * TILE
                        for t in range(TILE):
                            s += H[k0 + r, j0 + t] * H[k0 + c, j0 + t]
                    H[k0 + r, k0 + c] -= s
            qd.simt.block.sync()

            # Scalar Crout factorization of 16x16 diagonal block
            for col in range(TILE):
                if tid == 0:
                    tmp = H[k0 + col, k0 + col]
                    for j in range(col):
                        tmp -= H[k0 + col, k0 + j] * H[k0 + col, k0 + j]
                    H[k0 + col, k0 + col] = qd.sqrt(qd.max(tmp, qd.f32(1e-12)))
                qd.simt.block.sync()

                row = col + 1 + tid
                if row < TILE:
                    inv_d = qd.f32(1.0) / H[k0 + col, k0 + col]
                    dot = qd.f32(0.0)
                    for j in range(col):
                        dot += H[k0 + row, k0 + j] * H[k0 + col, k0 + j]
                    H[k0 + row, k0 + col] = (H[k0 + row, k0 + col] - dot) * inv_d
                qd.simt.block.sync()

            # Update off-diagonal blocks below the diagonal
            for ib in range(kb + 1, N_BLOCKS):
                i0 = ib * TILE

                # Subtract prior-block contributions from off-diagonal block
                for r in range(TILE):
                    c = tid
                    s = qd.f32(0.0)
                    for jb in range(kb):
                        j0 = jb * TILE
                        for t in range(TILE):
                            s += H[i0 + r, j0 + t] * H[k0 + c, j0 + t]
                    H[i0 + r, k0 + c] -= s
                qd.simt.block.sync()

                # Triangular solve against diagonal block
                for c in range(TILE):
                    r = tid
                    if r < TILE:
                        dot = qd.f32(0.0)
                        for j in range(c):
                            dot += H[i0 + r, k0 + j] * H[k0 + c, k0 + j]
                        H[i0 + r, k0 + c] = (H[i0 + r, k0 + c] - dot) / H[k0 + c, k0 + c]
                    qd.simt.block.sync()

        # Write result to global memory
        for row in range(N):
            c = tid
            while c <= row:
                L_blocked_field[env, row, c] = H[row, c]
                c += TILE


# ---------------------------------------------------------------------------
# Kernel 3: Tile16x16 blocked Cholesky (16 threads, no shared memory)
# ---------------------------------------------------------------------------


@qd.kernel
def cholesky_tile16():
    qd.loop_config(name="chol_tile16", block_dim=TILE)
    for idx in range(N_ENVS * TILE):
        env = idx // TILE

        for kb in range(N_BLOCKS):
            k0 = kb * TILE
            k1 = qd.min(k0 + TILE, N)

            # Load diagonal block (identity for padding rows beyond N)
            L_kk = qd.simt.Tile16x16.eye(dtype=qd.f32)
            L_kk[:] = A_field[env, k0:k1, k0:k1]

            # Subtract rank-1 contributions from prior column-blocks. Each vector load reads from global memory,
            # but only data this subgroup itself wrote in an earlier iteration (same thread reads its own prior
            # store), so no cross-thread sync is needed.
            for jb in range(kb):
                j0 = jb * TILE
                for t in range(TILE):
                    v = L_tile16_field[env, k0:k1, j0 + t]
                    L_kk -= qd.outer(v, v)

            # Factor the diagonal block
            L_kk.cholesky_(qd.f32(1e-12))

            # Process off-diagonal blocks below the diagonal
            for ib in range(kb + 1, N_BLOCKS):
                i0 = ib * TILE
                i1 = qd.min(i0 + TILE, N)

                # Load off-diagonal block (zeros for padding rows beyond N)
                L_ik = qd.simt.Tile16x16.zeros(dtype=qd.f32)
                L_ik[:] = A_field[env, i0:i1, k0:k1]

                # Subtract rank-1 contributions from prior column-blocks
                # (same self-read-after-self-write pattern as above)
                for jb in range(kb):
                    j0 = jb * TILE
                    for t in range(TILE):
                        v_own = L_tile16_field[env, i0:i1, j0 + t]
                        v_diag = L_tile16_field[env, k0:k1, j0 + t]
                        L_ik -= qd.outer(v_own, v_diag)

                # Triangular solve: L_ik @ L_kk^T = A_ik
                L_kk.solve_triangular_(L_ik)

                # Store off-diagonal result
                L_tile16_field[env, i0:i1, k0:k1] = L_ik

            # Store diagonal result
            L_tile16_field[env, k0:k1, k0:k1] = L_kk


# ---------------------------------------------------------------------------
# Benchmark + verification
# ---------------------------------------------------------------------------


def benchmark(kernel_fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        kernel_fn()
    qd.sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        kernel_fn()
    qd.sync()
    elapsed = time.perf_counter() - t0
    return elapsed / n_iters * 1e6


def verify(name, L_field, A_np):
    L_np = L_field.to_numpy()
    max_err = 0.0
    n_checked = min(8, N_ENVS)
    for i in range(n_checked):
        L_i = np.tril(L_np[i])
        recon = L_i @ L_i.T
        err = np.max(np.abs(recon - A_np[i]))
        if np.isnan(err) or np.isinf(err):
            print(f"  {name} env {i}: FAILED (nan/inf)")
            return float("inf")
        max_err = max(max_err, err)
    print(f"  {name} max reconstruction error (first {n_checked} envs): {max_err:.2e}")
    return max_err


def main():
    print(f"Blocked Cholesky {N}x{N} benchmark")
    print(f"  {N_ENVS} environments, {WARMUP} warmup, {ITERS} measured iterations")
    print()

    A_np = make_spd_matrices(N_ENVS, N)
    A_field.from_numpy(A_np)

    print("Compiling baseline (scalar Crout, 64 threads)...")
    cholesky_baseline()
    qd.sync()
    verify("baseline", L_baseline_field, A_np)

    print("Compiling blocked (scalar Crout, 16 threads, shared mem)...")
    cholesky_blocked()
    qd.sync()
    verify("blocked", L_blocked_field, A_np)

    print("Compiling Tile16x16 (blocked, 16 threads, no shared memory)...")
    cholesky_tile16()
    qd.sync()
    verify("tile16", L_tile16_field, A_np)
    print()

    t_baseline = benchmark(cholesky_baseline, WARMUP, ITERS)
    t_blocked = benchmark(cholesky_blocked, WARMUP, ITERS)
    t_tile16 = benchmark(cholesky_tile16, WARMUP, ITERS)

    print(f"{'Kernel':<45s} {'Threads':>7s} {'Time (us)':>10s} {'vs baseline':>12s}")
    rows = [
        ("baseline (scalar Crout, shared mem)", 64, t_baseline),
        ("blocked  (scalar Crout, shared mem)", 16, t_blocked),
        ("tile16   (Tile16x16, no shared memory)", 16, t_tile16),
    ]
    for name, threads, t in rows:
        speedup = t_baseline / t
        print(f"{name:<45s} {threads:>7d} {t:>10.0f} {speedup:>11.2f}x")


if __name__ == "__main__":
    main()
