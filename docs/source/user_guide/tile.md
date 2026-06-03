# Register-resident tiles: `Tile16x16` and `Tile32x32`

Quadrants provides two register-resident matrix tile types:

- `qd.simt.Tile16x16` — a 16x16 tile distributed across 16 threads in a subgroup (one row per thread, 16 scalar registers per thread).
- `qd.simt.Tile32x32` — a 32x32 tile distributed across 32 threads in a subgroup (one row per thread, 32 scalar registers per thread).

Both have identical APIs (creation, slice-syntax load/store, `qd.outer` rank-1 updates, `cholesky_`, `solve_triangular_`, SharedArray interop) and use subgroup shuffles for cross-thread communication — no shared memory needed. The rest of this page documents the API in terms of `Tile16x16`; everything carries over to `Tile32x32` by swapping the class name and using `SIZE == 32` / `block_dim=32`. The [`Tile32x32` section](#tile32x32) below has guidance on when to pick 32x32 vs 16x16 and a short example.

Tiles are useful for implementing blocked linear algebra kernels (Cholesky, triangular solve, etc.) where you want to keep working data in registers for maximum throughput.

Both tiles run on all GPU backends supported by Quadrants: CUDA, AMD, Metal, and Vulkan. They build on `qd.simt.subgroup.shuffle`, which is cross-platform — no vendor-specific libraries required. Using either tile on a CPU backend raises `QuadrantsSyntaxError`.

## Quick start

```python
import quadrants as qd

@qd.func
def my_blocked_op(A, row0, col0, eps):
    N = qd.simt.Tile16x16.SIZE
    t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
    t[:] = A[row0:row0+N, col0:col0+N]
    t.cholesky_(eps)
    A[row0:row0+N, col0:col0+N] = t
```

## Creating a tile

Tiles are created inside kernels or `@qd.func` functions:

```python
# inside a kernel or @qd.func:
t = qd.simt.Tile16x16.zeros(dtype=qd.f32)    # all zeros, f32
t = qd.simt.Tile16x16.zeros(dtype=qd.f64)    # all zeros, f64
t = qd.simt.Tile16x16.eye(dtype=qd.f32)      # 16x16 identity, f32
t = qd.simt.Tile16x16.eye(dtype=qd.f64)      # 16x16 identity, f64
```

The `dtype` argument is optional — if omitted it defaults to the runtime's `default_fp` (usually `qd.f32`). The underlying tile dataclass has 16 fields (`r0`–`r15`) — the scalar registers for one row.

## Loading and storing

Load/store transfers data between a tile and device memory arrays using slice syntax. Both `qd.ndarray` and `qd.field` are supported. Each thread accesses row `row0 + tid`, where `tid` is the thread's subgroup lane index (obtained internally via `subgroup.invocation_id()`).

### Slice syntax

```python
t[:] = arr[row0:row1, col0:col1]    # load from 2D array
arr[row0:row1, col0:col1] = t       # store to 2D array

t[:] = arr[batch, row0:row1, col0:col1]    # load from 3D array
arr[batch, row0:row1, col0:col1] = t       # store to 3D array
```

### Slice value rules

- **Both start and stop indices are required.** `arr[:N, :N]` and `arr[0:, 0:]` are not allowed; write `arr[0:N, 0:N]`.
- **Row range** `[row0, row1)`: thread `tid` accesses row `row0 + tid`. Threads where `row0 + tid >= row1` are skipped. Additionally, rows beyond the array's shape are skipped. Typically `row1 = row0 + qd.simt.Tile16x16.SIZE`, but smaller ranges work for partial tiles.
- **Column range** `[col0, col1)`: each active thread loads/stores columns `col0` through `min(col1, arr.shape[-1]) - 1`. Tile columns beyond this range are left as zero (load) or skipped (store). At most `qd.simt.Tile16x16.SIZE` columns are accessed (tile registers `r0`–`r15` map to `col0`, `col0+1`, …, `col0+15`).
- **Batch index** (3D only): a scalar integer indexing the leading dimension.

### Notes

The `[:]` on the load LHS is required — it distinguishes an in-place tile load from a variable rebinding. The store side does not need `[:]` because the array subscript on the LHS already triggers the correct assignment path.

## Identity initialization

For padding partial tiles in blocked algorithms:

```python
t = qd.simt.Tile16x16.eye(dtype=qd.f32)    # create a new identity tile
t.eye_()                                     # or reset an existing tile to identity in-place
```

Each thread sets its diagonal element to 1.0 and all others to 0.0.

## Rank-1 updates

```python
t -= qd.outer(v, v)    # t -= v @ v^T  (symmetric)
t -= qd.outer(a, b)    # t -= a @ b^T  (general)
```

Each thread provides its element(s) of the vector(s). The outer product is computed via subgroup shuffles and subtracted from the tile in-place. `qd.outer(a, b)` returns a deferred proxy — it is only valid as the RHS of `-=` on a Tile16x16. Composition like `qd.outer(a, b) + qd.outer(c, d)` raises `TypeError`.

### Loading column vectors for outer products

Column vectors can be loaded from arrays using slice syntax:

```python
v = arr[row0:row1, col]          # 2D: one element per thread
v = arr[batch, row0:row1, col]   # 3D: same, with batch index
t -= qd.outer(v, v)
```

The row slice follows the same rules as tile slices: both start and stop are required. `col` is a scalar column index. Each thread loads `arr[row0 + tid, col]`; threads where `row0 + tid >= row1` get zero.

## Cholesky factorization

```python
t.cholesky_(eps)
```

Factorizes the tile in-place: replaces the lower triangle with `L` such that `L @ L^T ≈ A`. The `eps` parameter clamps the diagonal to avoid numerical issues with near-singular matrices. After this call, the lower triangle of `t` contains `L`.

## Triangular solve

```python
L.solve_triangular_(B)
```

Solves `X @ L^T = B` in-place, replacing `B` with `X`. `L` must be a lower-triangular, non-singular tile (all diagonal elements non-zero, e.g. from `cholesky_()`). Only `lower=True` is supported; passing `lower=False` raises `TypeError`.

### Combined Cholesky + triangular solve

A common pattern is to factorize a tile and immediately solve against it:

```python
L = qd.simt.Tile16x16.zeros(dtype=qd.f32)
L[:] = A[0:N, 0:N]
L.cholesky_(eps)
B = qd.simt.Tile16x16.zeros(dtype=qd.f32)
B[:] = rhs[0:N, 0:N]
L.solve_triangular_(B)
rhs[0:N, 0:N] = B
```

## SharedArray support

Tiles can load from and store to `qd.simt.block.SharedArray` using the same slice syntax as device arrays:

```python
sh = qd.simt.block.SharedArray((qd.simt.Tile16x16.SIZE, qd.simt.Tile16x16.SIZE), qd.f32)
t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
t[:] = src[0:N, 0:N]
sh[0:N, 0:N] = t          # store tile to shared memory
qd.simt.block.sync()
t2 = qd.simt.Tile16x16.zeros(dtype=qd.f32)
t2[:] = sh[0:N, 0:N]      # load tile from shared memory
```

Column clamping applies the same way as for device arrays — columns beyond the SharedArray width are left as zero on load or skipped on store. Column vector slices (`v = sh[K0:K1, col]`) also work with SharedArray.

## Kernel structure

### Block size

Set `block_dim=qd.simt.Tile16x16.SIZE` so that each thread block contains exactly 16 threads — one per tile row:

```python
@qd.kernel
def my_kernel(A: qd.types.NDArray[qd.f32, 3]):
    N = qd.simt.Tile16x16.SIZE
    qd.loop_config(block_dim=N)
    for i in range(A.shape[0]):
        t = qd.simt.Tile16x16.zeros(dtype=qd.f32)
        t[:] = A[i, 0:N, 0:N]
        t.cholesky_(1e-6)
        A[i, 0:N, 0:N] = t
```

### Subgroup size

Tile operations communicate between threads using `qd.simt.subgroup.shuffle`. The hardware subgroup (warp) size is typically larger than 16 (e.g. 32 on NVIDIA). Quadrants handles this internally — tile operations only use the first 16 lanes, and the remaining lanes are idle.

## f64 support

Pass `dtype=qd.f64` for double precision:

```python
t = qd.simt.Tile16x16.zeros(dtype=qd.f64)
```

## `Tile32x32`

The 32x32 sibling is used the same way as `Tile16x16`, just with `block_dim=32` and `SIZE == 32`:

```python
import quadrants as qd

@qd.func
def my_blocked_op(A, row0, col0, eps):
    N = qd.simt.Tile32x32.SIZE   # == 32
    t = qd.simt.Tile32x32.zeros(dtype=qd.f32)
    t[:] = A[row0:row0+N, col0:col0+N]
    t.cholesky_(eps)
    A[row0:row0+N, col0:col0+N] = t
```

`Tile16x16` and `Tile32x32` can be mixed within the same kernel — their slice-dispatch caches are independent.

### When to pick 32x32 vs 16x16

| Size  | Threads per block | Registers per thread | Best for |
|-------|------------------:|---------------------:|----------|
| 16x16 | 16                | 16                   | Small problems where occupancy from many narrow blocks matters; very small N (e.g. N≤16 or N≈48) where the 32-tile would waste lanes |
| 32x32 | 32                | 32                   | Larger problems (N ≳ 32) where bigger tiles cut the number of blocked passes and the FMA chain inside `cholesky_` amortizes the larger register file |

## Method reference

| Operation | Description |
|-----------|-------------|
| `qd.simt.Tile16x16.zeros(dtype=...)` | Create a zero-initialized tile |
| `qd.simt.Tile16x16.eye(dtype=...)` | Create an identity tile |
| `qd.simt.Tile16x16.SIZE` | Tile dimension constant (16) |
| `t[:] = arr[r0:r1, c0:c1]` | Load from 2D array |
| `t[:] = arr[i, r0:r1, c0:c1]` | Load from 3D array |
| `arr[r0:r1, c0:c1] = t` | Store to 2D array |
| `arr[i, r0:r1, c0:c1] = t` | Store to 3D array |
| `t.eye_()` | Set to identity matrix (in-place) |
| `t -= qd.outer(v, v)` | Symmetric rank-1 subtract |
| `t -= qd.outer(a, b)` | General rank-1 subtract |
| `t.cholesky_(eps)` | In-place Cholesky factorization |
| `L.solve_triangular_(B)` | Triangular solve (in-place on B) |

## Example: blocked Cholesky

See [`misc/demos/cholesky_blocked.py`](../../../misc/demos/cholesky_blocked.py) for a complete blocked Cholesky factorization using Tile16x16, benchmarked against scalar-Crout baselines (shared memory with 64 threads, and blocked shared memory with 16 threads).

Results on RTX PRO 6000 Blackwell, 4096 environments, N=92, f32:

| Kernel | Threads | Time (us) | vs baseline |
|--------|--------:|----------:|------------:|
| baseline (scalar Crout, shared mem) | 64 | 2766 | 1.00x |
| blocked (scalar Crout, shared mem) | 16 | 2556 | 1.08x |
| **tile16 (Tile16x16, no shared memory)** | 16 | 533 | **5.19x** |
