# Tile16: register-resident 16x16 tiles

`Tile16` provides a 16x16 matrix tile that lives entirely in registers, distributed across 16 threads in a subgroup (warp). Each thread holds one row as 16 scalar registers. Cross-thread communication uses warp shuffles — no shared memory needed.

This is useful for implementing blocked linear algebra kernels (Cholesky, triangular solve, etc.) where you want to keep working data in registers for maximum throughput.

## Quick start

```python
from quadrants.lang.simt.tile16 import Tile16

@qd.func
def my_blocked_op(A, tid, row, col0, n_cols, eps):
    t = Tile16()
    t.load(A, row, col0, n_cols)
    t.potrf(tid, eps)
    t.store(A, row, col0, n_cols)
```

## Creating a tile

`Tile16()` creates a zero-initialized tile. You can also pass 16 initial values:

```python
t = Tile16()                                    # all zeros
t = Tile16(a0, a1, a2, ..., a15)                # explicit values
```

## Loading and storing

Load/store transfer data between a tile and device memory arrays.

### 2D arrays

```python
t.load(arr, row, col0, n_cols)       # arr[row, col0+0..15]
t.store(arr, row, col0, n_cols)
```

### 3D arrays

For arrays with a leading batch dimension (e.g. `H[batch, row, col]`):

```python
t.load3d(arr, i0, row, col0, n_cols)   # arr[i0, row, col0+0..15]
t.store3d(arr, i0, row, col0, n_cols)
```

All load/store methods perform column bounds checking against `n_cols`. Out-of-bounds columns are left as zero (load) or skipped (store).

## Identity initialization

For padding partial tiles in blocked algorithms:

```python
t.set_identity(tid)
```

Sets the tile to the identity matrix row for thread `tid`: column `tid` gets 1.0, all others 0.0. Across the full subgroup, this produces a distributed 16x16 identity matrix.

## Rank-1 updates

### Symmetric rank-1 subtract (syr)

```python
t.syr_sub(v)    # t -= v @ v^T
```

Each thread provides its element of the vector `v`. The outer product `v @ v^T` is computed via shuffles and subtracted from the tile in-place. Used for diagonal block updates in blocked Cholesky.

### General rank-1 subtract (ger)

```python
t.ger_sub(a, b)   # t -= a @ b^T
```

Like `syr_sub` but with two different vectors. Used for off-diagonal block updates.

## Cholesky factorization (potrf)

```python
t.potrf(tid, eps)
```

Factorizes the tile in-place: replaces the lower triangle with `L` such that `L @ L^T ≈ A`. The `eps` parameter clamps the diagonal to avoid numerical issues with near-singular matrices. After this call, the lower triangle of `t` contains `L`.

## Triangular solve (trsm)

```python
B.trsm(L)
```

Solves `L @ X^T = B^T` in-place, replacing `B` with `X`. `L` must be a lower-triangular tile (e.g. from `potrf`). Used for off-diagonal blocks in blocked Cholesky.

## Full example: blocked Cholesky

A simplified blocked Cholesky factorization using `Tile16`:

```python
from quadrants.lang.simt.tile16 import Tile16

TILE = 16

@qd.func
def blocked_cholesky(H, tid, n_dofs, eps):
    N_BLOCKS = (n_dofs + TILE - 1) // TILE

    for kb in range(N_BLOCKS):
        k0 = kb * TILE

        # Load diagonal block, pad with identity if out of bounds
        L_kk = Tile16()
        if k0 + tid < n_dofs:
            L_kk.load(H, k0 + tid, k0, n_dofs)
        else:
            L_kk.set_identity(tid)

        # Subtract contributions from previous blocks
        for jb in range(kb):
            j0 = jb * TILE
            for t in range(TILE):
                v = 0.0
                if k0 + tid < n_dofs:
                    v = H[k0 + tid, j0 + t]
                L_kk.syr_sub(v)

        # Factorize diagonal block
        L_kk.potrf(tid, eps)

        # Process off-diagonal blocks
        for ib in range(kb + 1, N_BLOCKS):
            i0 = ib * TILE

            L_ik = Tile16()
            if i0 + tid < n_dofs:
                L_ik.load(H, i0 + tid, k0, n_dofs)

            for jb in range(kb):
                j0 = jb * TILE
                for t in range(TILE):
                    v_own = 0.0
                    v_diag = 0.0
                    if i0 + tid < n_dofs:
                        v_own = H[i0 + tid, j0 + t]
                    if k0 + tid < n_dofs:
                        v_diag = H[k0 + tid, j0 + t]
                    L_ik.ger_sub(v_own, v_diag)

            L_ik.trsm(L_kk)

            if i0 + tid < n_dofs:
                L_ik.store(H, i0 + tid, k0, n_dofs)

        if k0 + tid < n_dofs:
            L_kk.store(H, k0 + tid, k0, n_dofs)
```

## Method reference

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `(arr, row, col0, n_cols)` | Load from 2D array |
| `load3d` | `(arr, i0, row, col0, n_cols)` | Load from 3D array |
| `store` | `(arr, row, col0, n_cols)` | Store to 2D array |
| `store3d` | `(arr, i0, row, col0, n_cols)` | Store to 3D array |
| `set_identity` | `(tid)` | Set to identity matrix row |
| `syr_sub` | `(v)` | Symmetric rank-1 subtract |
| `ger_sub` | `(a, b)` | General rank-1 subtract |
| `potrf` | `(tid, eps)` | Cholesky factorization |
| `trsm` | `(L)` | Triangular solve |

## Experiment: 64x64 blocked Cholesky (dex_hand dimensions)

Benchmark script: `misc/demos/cholesky_blocked.py`

Setup: Quadrants 0.6.0b1, RTX 5090, 4096 environments, f32. The 64x64 matrix (62 DOFs padded to 64) matches the constraint-space Hessian in the Genesis dex_hand scene.

### Kernels

**Baseline (Genesis-style)**: flat scalar Cholesky-Crout, 64 threads per env, 64 sequential column steps with `block.sync()` after each. Thread 0 computes diagonal, remaining threads parallelize off-diagonal updates. Matches `func_cholesky_factor_direct_tiled` in Genesis.

**Tile16 (no shared memory)**: 4x4 grid of 16x16 tiles, 16 threads per env. Diagonal blocks loaded into `Tile16`, updated via `syr_sub`, factorized via `potrf`. Off-diagonal blocks loaded into a second `Tile16`, updated via `ger_sub`, solved via `trsm`. All operations are register-resident with zero shared memory and zero syncs. Prior factorized tiles are read back from global memory (served by L2 cache).

### Results

```
Kernel                                      Threads   Time (us)    vs baseline
baseline (Genesis Crout)                         64       569         1.00x
blocked  (scalar POTRF, shared mem)              16       482         1.18x
blocked  (shuffle POTRF, shared mem)             16       503         1.13x
fused    (GEMM+POTRF in regs)                    16       473         1.20x
fused2   (+ reg TRSM)                            16       411         1.39x
fused3   (+ shuffle off-diag GEMM)               16       373         1.53x
tile16   (Tile16, no shared memory)              16       179         3.17x
```

### Why it's fast

The key insight is that shared memory, despite being "fast", is the bottleneck — not as a slow memory, but as a **scarce resource limiting occupancy**.

With shared memory, each thread block needs ~17 KB for the 64x65 matrix (padded for bank conflicts). On an SM with 128 KB shared, this allows ~7 concurrent warps. The `Tile16` kernel uses zero shared memory, so occupancy is limited only by registers (~50 per thread x 16 threads = 800 per block), allowing ~32 concurrent warps — a 4.5x increase in schedulable warps that provides dramatically better latency hiding.

Despite non-coalesced global reads (stride-64 row-major), the L2 cache (96 MB on Blackwell) easily holds the working set (~16 KB per env x 4096 envs = 64 MB). The massive occupancy gain dominates.

### Optimization progression

The path from baseline to `Tile16` involved six incremental steps:

1. **Blocking (1.18x)**: 4x4 grid of 16x16 tiles reduces sequential depth from 64 to 4 block-column steps. Smaller `block_dim` (16 vs 64) improves occupancy.

2. **Fused GEMM+POTRF (1.20x)**: loads diagonal block into registers first, runs GEMM subtract via shuffles, then POTRF — no intermediate sync or shared memory round-trip.

3. **Register-resident TRSM (1.39x)**: keeps `L_kk` in registers after POTRF, loads each off-diagonal block into a second register set. TRSM solves entirely via shuffles, eliminating ~120 shared memory reads of `L_kk` per off-diagonal block.

4. **Shuffle off-diagonal GEMM (1.53x)**: converts the last shared-memory-heavy operation to shuffles. Reduces shared memory reads from 32 to 2 per iteration — a 16x reduction. Total syncs drop from ~235 to 5.

5. **Eliminate shared memory (3.17x)**: replaces shared memory with global memory lookback (L2 cache). Occupancy jumps from ~7 to ~32 warps. The 2.08x gain over the previous step confirms shared memory scarcity was the primary bottleneck.

All of these steps are encapsulated in the `Tile16` API — the user writes `L_kk.potrf(tid, eps)` and gets the fully optimized register-resident implementation.
