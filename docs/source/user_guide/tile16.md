# Tile16x16: register-resident 16x16 tiles

`Tile16x16` provides a 16x16 matrix tile that lives entirely in registers, distributed across 16 threads in a subgroup (warp). Each thread holds one row as 16 scalar registers. Cross-thread communication uses subgroup shuffles — no shared memory needed.

This is useful for implementing blocked linear algebra kernels (Cholesky, triangular solve, etc.) where you want to keep working data in registers for maximum throughput.

Tile16x16 runs on all GPU backends supported by Quadrants: CUDA, AMD, Metal, and Vulkan. It builds on `qd.simt.subgroup.shuffle`, which is cross-platform — no vendor-specific libraries required.

## Quick start

```python
import quadrants as qd
from quadrants.lang.simt._tile16 import _make_tile16x16

Tile = _make_tile16x16(qd.f32)
N = Tile.SIZE  # 16

@qd.func
def my_blocked_op(A, row0, col0, eps):
    t = Tile.zeros()
    t[:] = A[row0:row0+N, col0:col0+N]
    t.cholesky_(eps)
    A[row0:row0+N, col0:col0+N] = t
```

## Creating a tile

```python
Tile = _make_tile16x16(qd.f32)   # or qd.f64

t = Tile.zeros()       # all zeros
t = Tile.eye()         # 16x16 identity
```

`_make_tile16x16` returns a `qd.dataclass` type whose 16 fields (`r0`–`r15`) are the scalar registers for one row. The result is cached per dtype.

## Loading and storing

Load/store transfers data between a tile and device memory arrays using slice syntax. Each thread accesses row `row0 + tid`, where `tid` is the thread's subgroup lane index (obtained internally via `subgroup.invocation_id()`).

### Slice syntax

```python
t[:] = arr[row0:row1, col0:col1]    # load from 2D array
arr[row0:row1, col0:col1] = t       # store to 2D array

t[:] = arr[batch, row0:row1, col0:col1]    # load from 3D array
arr[batch, row0:row1, col0:col1] = t       # store to 3D array
```

### Slice value rules

- **Both start and stop indices are required.** `arr[:N, :N]` and `arr[0:, 0:]` are not allowed; write `arr[0:N, 0:N]`.
- **Row range** `[row0, row1)`: thread `tid` accesses row `row0 + tid`. Threads where `row0 + tid >= row1` are skipped. Additionally, rows beyond the array's shape are skipped. Typically `row1 = row0 + Tile.SIZE`, but smaller ranges work for partial tiles.
- **Column range** `[col0, col1)`: each active thread loads/stores columns `col0` through `min(col1, arr.shape[-1]) - 1`. Tile columns beyond this range are left as zero (load) or skipped (store). At most `Tile.SIZE` columns are accessed (tile registers `r0`–`r15` map to `col0`, `col0+1`, …, `col0+15`).
- **Batch index** (3D only): a scalar integer indexing the leading dimension.

### Notes

The `[:]` on the load LHS is required — it distinguishes an in-place tile load from a variable rebinding. The store side does not need `[:]` because the array subscript on the LHS already triggers the correct assignment path.

## Identity initialization

For padding partial tiles in blocked algorithms:

```python
t = Tile.eye()    # create a new identity tile
t._eye_()         # or reset an existing tile to identity in-place
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

Solves `X @ L^T = B` in-place, replacing `B` with `X`. `L` must be a lower-triangular tile (e.g. from `cholesky_()`). Only `lower=True` is supported; passing `lower=False` raises `TypeError`.

## Kernel structure

Each tile operation uses 16 lanes of a subgroup. Set `block_dim=Tile.SIZE` so that each thread block is one 16-thread group:

```python
N = Tile.SIZE

@qd.kernel
def my_kernel(A: qd.types.NDArray[qd.f32, 3]):
    qd.loop_config(block_dim=N)
    for i in range(A.shape[0]):
        t = Tile.zeros()
        t[:] = A[i, 0:N, 0:N]
        t.cholesky_(1e-6)
        A[i, 0:N, 0:N] = t
```

## f64 support

Pass `qd.f64` to the factory for double precision:

```python
Tile64 = _make_tile16x16(qd.f64)
```

Not all GPU backends support f64. Use `test_utils.skip_if_f64_unsupported()` in tests.

## Method reference

| Operation | Description |
|-----------|-------------|
| `Tile.zeros()` | Create a zero-initialized tile |
| `Tile.eye()` | Create an identity tile |
| `Tile.SIZE` | Tile dimension constant (16) |
| `t[:] = arr[r0:r1, c0:c1]` | Load from 2D array |
| `t[:] = arr[i, r0:r1, c0:c1]` | Load from 3D array |
| `arr[r0:r1, c0:c1] = t` | Store to 2D array |
| `arr[i, r0:r1, c0:c1] = t` | Store to 3D array |
| `t._eye_()` | Set to identity matrix (in-place) |
| `t -= qd.outer(v, v)` | Symmetric rank-1 subtract |
| `t -= qd.outer(a, b)` | General rank-1 subtract |
| `t.cholesky_(eps)` | In-place Cholesky factorization |
| `L.solve_triangular_(B)` | Triangular solve (in-place on B) |
