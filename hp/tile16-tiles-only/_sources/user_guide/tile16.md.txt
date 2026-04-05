# Tile16: register-resident 16x16 tiles

`Tile16` provides a 16x16 matrix tile that lives entirely in registers, distributed across 16 threads in a subgroup (warp). Each thread holds one row as 16 scalar registers. Cross-thread communication uses warp shuffles — no shared memory needed.

This is useful for implementing blocked linear algebra kernels (Cholesky, triangular solve, etc.) where you want to keep working data in registers for maximum throughput.

Tile16 runs on all GPU backends supported by Quadrants: CUDA, AMD, Metal, and Vulkan. It builds on `qd.simt.subgroup.shuffle`, which is cross-platform — no vendor-specific libraries required.

## Quick start

```python
from quadrants.lang.simt.tile16 import Tile16

@qd.func
def my_blocked_op(A, row0, col0, n_cols, eps):
    t = Tile16()
    t.load(A, row0, col0, n_cols)
    t.potrf(eps)
    t.store(A, row0, col0, n_cols)
```

## Creating a tile

`Tile16()` creates a zero-initialized tile. You can also pass 16 initial values:

```python
t = Tile16()                                    # all zeros
t = Tile16(a0, a1, a2, ..., a15)                # explicit values
```

## Loading and storing

Load/store transfer data between a tile and device memory arrays. Each thread accesses row `row0 + tid`, where `tid` is the thread's subgroup lane index (obtained internally via `subgroup.invocation_id()`).

### 2D arrays

```python
t.load(arr, row0, col0, n_cols)       # arr[row0 + tid, col0+0..15]
t.store(arr, row0, col0, n_cols)
```

### 3D arrays

For arrays with a leading batch dimension (e.g. `H[batch, row, col]`):

```python
t.load3d(arr, i0, row0, col0, n_cols)   # arr[i0, row0 + tid, col0+0..15]
t.store3d(arr, i0, row0, col0, n_cols)
```

All load/store methods perform column bounds checking against `n_cols`. Out-of-bounds columns are left as zero (load) or skipped (store).

## Identity initialization

For padding partial tiles in blocked algorithms:

```python
t.set_identity()
```

Sets the tile to the 16x16 identity matrix. Each thread sets its diagonal element to 1.0 and all others to 0.0.

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
t.potrf(eps)
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
            L_kk.load(H, k0, k0, n_dofs)
        else:
            L_kk.set_identity()

        # Subtract contributions from previous blocks
        for jb in range(kb):
            j0 = jb * TILE
            for t in range(TILE):
                v = 0.0
                if k0 + tid < n_dofs:
                    v = H[k0 + tid, j0 + t]
                L_kk.syr_sub(v)

        # Factorize diagonal block
        L_kk.potrf(eps)

        # Process off-diagonal blocks
        for ib in range(kb + 1, N_BLOCKS):
            i0 = ib * TILE

            L_ik = Tile16()
            if i0 + tid < n_dofs:
                L_ik.load(H, i0, k0, n_dofs)

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
                L_ik.store(H, i0, k0, n_dofs)

        if k0 + tid < n_dofs:
            L_kk.store(H, k0, k0, n_dofs)
```

## Method reference

| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `(arr, row0, col0, n_cols)` | Load from 2D array (row = row0 + tid) |
| `load3d` | `(arr, i0, row0, col0, n_cols)` | Load from 3D array (row = row0 + tid) |
| `store` | `(arr, row0, col0, n_cols)` | Store to 2D array (row = row0 + tid) |
| `store3d` | `(arr, i0, row0, col0, n_cols)` | Store to 3D array (row = row0 + tid) |
| `set_identity` | `()` | Set to 16x16 identity matrix |
| `syr_sub` | `(v)` | Symmetric rank-1 subtract |
| `ger_sub` | `(a, b)` | General rank-1 subtract |
| `potrf` | `(eps)` | Cholesky factorization |
| `trsm` | `(L)` | Triangular solve |

## Experiment: 64x64 blocked Cholesky (dex_hand dimensions)

Benchmark script: `misc/demos/cholesky_blocked.py`

RTX PRO 6000 Blackwell, 4096 environments, f32, 64x64 matrices (dex_hand constraint-space Hessian).

```
Kernel                                      Threads   Time (us)    vs baseline
baseline (scalar Crout, shared mem)              64       611         1.00x
blocked  (scalar Crout, shared mem)              16       515         1.19x
tile16   (Tile16, no shared memory)              16       200         3.05x
```
