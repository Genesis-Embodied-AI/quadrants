#!/usr/bin/env python3
"""Benchmark 64x64 blocked Cholesky factorization using Tile16.

Compares a baseline Genesis-style scalar Cholesky-Crout (64 threads, shared
memory, 64 sequential syncs) against a fully register-resident blocked
Cholesky using Tile16 (16 threads, no shared memory, zero syncs).

The blocked kernel reads initial values from global memory, writes factorized
tiles to a separate output field, and reads prior tiles from that field for
GEMM lookback (served by L2 cache). Each thread only reads/writes its own
rows, so program order guarantees visibility without any fencing.

Tested on RTX 5090, 4096 environments, f32:

    Kernel                          Threads   Time (us)    vs baseline
    baseline (Genesis Crout)             64       569         1.00x
    blocked  (Tile16, no shmem)          16       179         3.17x

Usage:
    python misc/demos/cholesky_blocked.py
"""

import time

import numpy as np
import quadrants as qd
from quadrants.lang.simt.tile16 import Tile16

N = 64
TILE = 16
N_BLOCKS = N // TILE  # 4
N_ENVS = 4096
WARMUP = 50
ITERS = 200

qd.init(arch=qd.cuda)

A_field = qd.field(dtype=qd.f32, shape=(N_ENVS, N, N))
L_baseline_field = qd.field(dtype=qd.f32, shape=(N_ENVS, N, N))
L_tile16_field = qd.field(dtype=qd.f32, shape=(N_ENVS, N, N))


def make_spd_matrices(n_envs, n):
    rng = np.random.default_rng(42)
    R = rng.standard_normal((n_envs, n, n)).astype(np.float32)
    A = np.einsum("bij,bik->bjk", R, R) + np.eye(n, dtype=np.float32)
    return A


# ---------------------------------------------------------------------------
# Baseline: Genesis-style scalar Cholesky-Crout (64 threads, shared memory)
# ---------------------------------------------------------------------------

@qd.kernel
def cholesky_baseline():
    qd.loop_config(name="chol_baseline", block_dim=N)
    for idx in range(N_ENVS * N):
        tid = idx % N
        env = idx // N

        H = qd.simt.block.SharedArray(N * (N + 1), qd.f32)

        for c in range(N):
            if tid <= c:
                H[c * (N + 1) + tid] = A_field[env, c, tid]
        qd.simt.block.sync()

        for j in range(N):
            if tid == 0:
                s = qd.f32(0.0)
                for p in range(j):
                    s += H[j * (N + 1) + p] * H[j * (N + 1) + p]
                H[j * (N + 1) + j] = qd.sqrt(qd.max(H[j * (N + 1) + j] - s, qd.f32(1e-12)))
            qd.simt.block.sync()

            row = tid
            if row > j:
                s = qd.f32(0.0)
                for p in range(j):
                    s += H[row * (N + 1) + p] * H[j * (N + 1) + p]
                H[row * (N + 1) + j] = (H[row * (N + 1) + j] - s) / H[j * (N + 1) + j]
            qd.simt.block.sync()

        for c in range(N):
            if tid <= c:
                L_baseline_field[env, c, tid] = H[c * (N + 1) + tid]


# ---------------------------------------------------------------------------
# Tile16: fully register-resident blocked Cholesky (16 threads, no shmem)
# ---------------------------------------------------------------------------

@qd.kernel
def cholesky_tile16():
    qd.loop_config(name="chol_tile16", block_dim=TILE)
    for idx in range(N_ENVS * TILE):
        tid = idx % TILE
        env = idx // TILE

        for kb in range(N_BLOCKS):
            k0 = kb * TILE

            # Load diagonal block from A
            L_kk = Tile16()
            L_kk.load3d(A_field, env, k0 + tid, k0, N)

            # Diagonal syr subtract: A_kk -= L_k* @ L_k*^T (lookback from L)
            for jb in range(kb):
                j0 = jb * TILE
                for t in range(TILE):
                    v = L_tile16_field[env, k0 + tid, j0 + t]
                    L_kk.syr_sub(v)

            # POTRF: factorize diagonal tile
            L_kk.potrf(tid, qd.f32(1e-12))

            # Off-diagonal blocks
            for ib in range(kb + 1, N_BLOCKS):
                i0 = ib * TILE

                # Load off-diagonal block from A
                L_ik = Tile16()
                L_ik.load3d(A_field, env, i0 + tid, k0, N)

                # Off-diagonal ger subtract: A_ik -= L_i* @ L_k*^T
                for jb in range(kb):
                    j0 = jb * TILE
                    for t in range(TILE):
                        v_own = L_tile16_field[env, i0 + tid, j0 + t]
                        v_diag = L_tile16_field[env, k0 + tid, j0 + t]
                        L_ik.ger_sub(v_own, v_diag)

                # TRSM: solve L_kk @ X^T = L_ik^T
                L_ik.trsm(L_kk)

                # Store off-diagonal result
                L_ik.store3d(L_tile16_field, env, i0 + tid, k0, N)

            # Store diagonal result
            L_kk.store3d(L_tile16_field, env, k0 + tid, k0, N)


# ---------------------------------------------------------------------------
# Benchmark + verification
# ---------------------------------------------------------------------------

def benchmark(name, kernel_fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        kernel_fn()
    qd.sync()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        kernel_fn()
    qd.sync()
    elapsed = time.perf_counter() - t0
    us_per_call = elapsed / n_iters * 1e6
    print(f"  {name:40s}  {us_per_call:8.1f} us/call  ({n_iters} iters)")
    return us_per_call


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
    print(f"Blocked Cholesky {N}x{N} benchmark (dex_hand dimensions)")
    print(f"  {N_ENVS} environments, {WARMUP} warmup, {ITERS} measured iterations")
    print()

    A_np = make_spd_matrices(N_ENVS, N)
    A_field.from_numpy(A_np)

    print("Compiling baseline (Genesis-style Crout, 64 threads)...")
    cholesky_baseline()
    qd.sync()
    verify("baseline", L_baseline_field, A_np)

    print("Compiling Tile16 (blocked, 16 threads, no shared memory)...")
    cholesky_tile16()
    qd.sync()
    verify("tile16", L_tile16_field, A_np)
    print()

    print("Benchmarking:")
    t_baseline = benchmark("baseline (Genesis Crout, 64 threads)", cholesky_baseline, WARMUP, ITERS)
    t_tile16 = benchmark("tile16 (blocked, no shmem, 16 threads)", cholesky_tile16, WARMUP, ITERS)
    print()
    print(f"  Tile16 vs baseline: {t_baseline / t_tile16:.2f}x")


if __name__ == "__main__":
    main()
