# Tile16x16: register-resident 16x16 tiles

`Tile16x16` provides a 16x16 matrix tile that lives entirely in registers, distributed across 16 threads in a subgroup (warp). Each thread holds one row as 16 scalar registers. Cross-thread communication uses warp shuffles — no shared memory needed.

This is useful for implementing blocked linear algebra kernels (Cholesky, triangular solve, etc.) where you want to keep working data in registers for maximum throughput.

Tile16x16 runs on all GPU backends supported by Quadrants: CUDA, AMD, Metal, and Vulkan. It builds on `qd.simt.subgroup.shuffle`, which is cross-platform — no vendor-specific libraries required.

## Quick start

```python
from quadrants.lang.simt.tile16 import Tile16x16

@qd.func
def my_blocked_op(A, row0, col0, eps):
    t = Tile16x16()
    t[:] = A[row0:row0+16, col0:col0+16]
    t.cholesky_(eps)
    A[row0:row0+16, col0:col0+16] = t
```

## Creating a tile

`Tile16x16.zeros()` creates a zero-initialized tile. `Tile16x16.eye()` creates an identity tile. You can also pass 16 initial values:

```python
t = Tile16x16.zeros()                              # all zeros
t = Tile16x16.eye()                                # 16x16 identity
t = Tile16x16(a0, a1, a2, ..., a15)                # explicit values
```

## Loading and storing

Load/store transfer data between a tile and device memory arrays using slice syntax. Each thread accesses row `row0 + tid`, where `tid` is the thread's subgroup lane index (obtained internally via `subgroup.invocation_id()`).

### 2D arrays

```python
t = Tile16x16()
t[:] = arr[row0:row0+16, col0:col0+16]    # load
arr[row0:row0+16, col0:col0+16] = t       # store
```

### 3D arrays

For arrays with a leading batch dimension (e.g. `H[batch, row, col]`):

```python
t = Tile16x16()
t[:] = arr[i0, row0:row0+16, col0:col0+16]    # load
arr[i0, row0:row0+16, col0:col0+16] = t       # store
```

Standard NumPy-style slicing. The load/store automatically clamps column indices to the array's shape, so out-of-bounds columns are left as zero (load) or skipped (store).

The `[:]` on the load LHS is required — it distinguishes an in-place tile load from a variable rebinding. The store side does not need `[:]` because the array subscript on the LHS already triggers the correct assignment path.

## Identity initialization

For padding partial tiles in blocked algorithms:

```python
t = Tile16x16.eye()   # create a new identity tile
t.eye_()               # or reset an existing tile to identity in-place
```

Each thread sets its diagonal element to 1.0 and all others to 0.0.

## Rank-1 updates

```python
t -= qd.outer(v, v)    # t -= v @ v^T  (symmetric)
t -= qd.outer(a, b)    # t -= a @ b^T  (general)
```

Each thread provides its element(s) of the vector(s). The outer product is computed via warp shuffles and subtracted from the tile in-place. `qd.outer(a, b)` returns a deferred proxy — it is only valid as the RHS of `-=` on a Tile16x16. Composition like `qd.outer(a, b) + qd.outer(c, d)` raises `TypeError`.

Used for diagonal block updates (symmetric case) and off-diagonal block updates (general case) in blocked Cholesky.

## Cholesky factorization (cholesky_)

```python
t.cholesky_(eps)
```

Factorizes the tile in-place: replaces the lower triangle with `L` such that `L @ L^T ≈ A`. The `eps` parameter clamps the diagonal to avoid numerical issues with near-singular matrices. After this call, the lower triangle of `t` contains `L`.

## Triangular solve (solve_triangular_)

```python
L.solve_triangular_(B)
```

Solves `X @ L^T = B` in-place, replacing `B` with `X`. `L` (self) must be a lower-triangular tile (e.g. from `cholesky_()`). Only `lower=True` is supported; passing `lower=False` raises `TypeError`. Used for off-diagonal blocks in blocked Cholesky.

## Full example: blocked Cholesky

See the `cholesky_tile16` kernel in [`misc/demos/cholesky_blocked.py`](../../../misc/demos/cholesky_blocked.py) for a complete blocked Cholesky factorization using `Tile16x16`.

## Method reference

| Operation | Description |
|-----------|-------------|
| `Tile16x16.zeros()` | Create a zero-initialized tile |
| `Tile16x16.eye()` | Create an identity tile |
| `t[:] = arr[r0:r0+16, c0:c_end]` | Load from 2D array |
| `t[:] = arr[i, r0:r0+16, c0:c_end]` | Load from 3D array |
| `arr[r0:r0+16, c0:c_end] = t` | Store to 2D array |
| `arr[i, r0:r0+16, c0:c_end] = t` | Store to 3D array |
| `t.eye_()` | Set to 16x16 identity matrix (in-place) |
| `t -= qd.outer(v, v)` | Symmetric rank-1 subtract |
| `t -= qd.outer(a, b)` | General rank-1 subtract |
| `t.cholesky_(eps)` | In-place Cholesky factorization |
| `L.solve_triangular_(B)` | Triangular solve (in-place on B) |

## Experiment: 92x92 blocked Cholesky (dex_hand dimensions)

Benchmark script: `misc/demos/cholesky_blocked.py`

RTX PRO 6000 Blackwell, 4096 environments, f32, 92x92 matrices (dex_hand constraint-space Hessian).

```
Kernel                                        Threads  Time (us)  vs baseline
baseline (scalar Crout, shared mem)                64       2766        1.00x
blocked  (scalar Crout, shared mem)                16       2556        1.08x
tile16   (Tile16x16, no shared memory)             16        533        5.19x
```
