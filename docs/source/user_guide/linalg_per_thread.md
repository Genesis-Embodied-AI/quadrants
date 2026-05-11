# Per-thread linear algebra

Small matrix decompositions and linear solvers that run **per thread, in registers, with no cross-thread cooperation** — every thread independently decomposes / solves its own 2×2 .. 12×12 matrix. No shared memory, no syncs, no atomics on shared state, no warp / subgroup primitives. A 1000-element kernel runs 1000 copies of the algorithm in parallel, each on its own data.

This is a different category from the element-wise / arithmetic matrix operations covered in [matrix_vector_per_thread](matrix_vector_per_thread.md) (also per-thread, but closed-form rather than iterative), and from the cross-thread / sparse linear algebra under `qd.linalg.*` (CG, sparse direct solvers — covered separately).

Each entry point here implements an *iterative or multi-step numerical algorithm* (Jacobi sweeps, Gauss elimination, Givens rotations) rather than a single closed-form formula.

All ops live at the top level (`qd.svd`, `qd.sym_eig`, `qd.make_spd`, `qd.polar_decompose`, `qd.eig`, `qd.solve`) and are intended to be called from inside a `@qd.kernel` or `@qd.func`.

## What's available

| Op                              | Operates on            | Shapes               | Returns                                                |
|---------------------------------|------------------------|----------------------|--------------------------------------------------------|
| `qd.svd(A)`                     | square real matrix     | 2×2, 3×3             | `(U, S, V)` such that `A = U @ S @ V.transpose()`      |
| `qd.sym_eig(A)`                 | symmetric real matrix  | 2×2 .. 12×12         | `(eigenvalues, eigenvectors)` (real)                   |
| `qd.make_spd(A)`                | symmetric real matrix  | 4×4 .. 12×12         | `M` — the closest positive semi-definite matrix to `A` |
| `qd.polar_decompose(A)`         | square real matrix     | 2×2, 3×3             | `(R, S)` such that `A = R @ S`, `R` orthogonal, `S` SPD |
| `qd.eig(A)`                     | square real matrix     | 2×2                  | `(eigenvalues, eigenvectors)` (complex, packed)        |
| `qd.solve(A, b)`                | square `A` + vector `b`| 2×2, 3×3             | `x` such that `A @ x = b`                              |

A few patterns to note:

- **Shapes are fixed.** Calling any of these on a matrix outside the supported shapes raises an exception at trace time (e.g. `"SVD only supports 2×2 and 3×3 matrices."`). There is no fallback for larger shapes.
- **All ops accept an optional `dt` argument.** When unspecified, it defaults to `impl.get_runtime().default_fp` — usually `qd.f32` unless overridden in `qd.init()`. Pass `dt=qd.f64` for the high-precision variant.
- **Output shape matches the input shape.** A 3×3 input yields 3×3 outputs (and a length-3 vector for `solve` / eigenvalues); a 2×2 input yields 2×2 outputs.
- **Real matrices only.** `qd.eig` returns complex results in a packed real layout (see below); the others all assume real-valued input and return real-valued output.

## Semantics

### `qd.svd(A, dt=None)`

Singular value decomposition — produces `(U, S, V)` such that `A = U @ S @ V.transpose()`, with:

- `U` orthogonal (`U @ U.transpose() = I`).
- `V` orthogonal.
- `S` diagonal, with non-negative singular values.

Shapes 2×2 and 3×3 use closed-form / Jacobi-style implementations specialized per shape.

Singular values come back sorted **descending** (`S[0,0] >= S[1,1] >= ...`) for both 2×2 and 3×3. The 3×3 path uses the Sifakis algorithm and absorbs `sign(det(A))` into σ rather than into `U` / `V`, so the smallest entry of `S` may be negative when `det(A) < 0`; the descending sort is on direct numeric value (matches the 2×2 path and NumPy / LAPACK conventions).

For 3×3 the implementation enforces `det(U) = det(V) = +1` regardless of input — useful for ARAP-style rotations `R = U @ V.transpose()` (which is then guaranteed to be a proper rotation). The 2×2 path does not enforce this convention; if you need a particular handedness on 2×2, check it explicitly and flip a column if needed.

### `qd.sym_eig(A, dt=None)`

Symmetric eigendecomposition — for a real symmetric `A`, returns `(eigenvalues, eigenvectors)` with:

- `eigenvalues`: a `Vector(n)` of real eigenvalues.
- `eigenvectors`: a `Matrix(n, n)` whose columns are the corresponding orthonormal eigenvectors.

Eigenvalues come back sorted **ascending** (`eigvals[0] <= eigvals[1] <= ...`) for every shape, with column `i` of `eigenvectors` being the eigenvector for `eigvals[i]`. This matches NumPy / LAPACK's `eigh` (note that `qd.svd` sorts σ *descending*, also matching its NumPy / LAPACK counterpart — the cross-op disagreement is the standard convention, not an inconsistency).

Three implementations dispatch by size:

- **2×2** — closed-form via the trace / determinant identity.
- **3×3** — closed-form Cardano method (Eigen3 `computeDirect`).
- **4×4 .. 12×12** — cyclic Jacobi: 12 sweeps of Givens rotations zeroing every off-diagonal `(p, q)` pair, with `Q := Q · J` accumulated as eigenvectors. ~6 digits of accuracy in `f32`, ~12 digits in `f64`.

Calling at `N >= 13` raises (`"Symmetric eigen solver currently supports sizes up to 12×12."`).

`A` is *assumed* symmetric; the implementation does not symmetrize first. If your matrix is only approximately symmetric (e.g. accumulated floating-point error), explicitly compute `(A + A.transpose()) * 0.5` before calling.

### `qd.make_spd(A, dt=None)`

Project a symmetric matrix `A` to the closest positive semi-definite matrix in the Frobenius-norm sense. Implemented as `Q · diag(max(λ, 0)) · Qᵀ` where `A = Q · diag(λ) · Qᵀ` is the symmetric eigendecomposition computed by `qd.sym_eig`.

Available for shapes 4×4 .. 12×12 (it shares the cyclic-Jacobi path with `qd.sym_eig`; for 2×2 / 3×3 you can write the same projection by hand using `qd.sym_eig` directly — see the example below).

Use cases:

- Projecting an indefinite contact / element Hessian to its closest SPD approximation before assembly (qipc-style).
- Regularizing a covariance estimate that may have small negative eigenvalues from rounding.
- Producing a usable preconditioner from a not-quite-SPD matrix.

`make_spd` is a Frobenius-projector onto the SPD cone: `make_spd(make_spd(A)) == make_spd(A)`. If `A` is already SPD it is returned essentially unchanged (up to `sym_eig` round-trip error); if `A` is negative-definite the result is the zero matrix.

### `qd.polar_decompose(A, dt=None)`

Polar decomposition — produces `(R, S)` such that `A = R @ S`, with:

- `R` orthogonal (the closest-rotation factor, modulo handedness).
- `S` symmetric positive semi-definite (the stretch factor).

Built on top of SVD: `A = U @ Σ @ Vᵀ` ⇒ `R = U @ Vᵀ`, `S = V @ Σ @ Vᵀ`. The same caveats about sign convention as `qd.svd` apply — `R` is the closest orthogonal factor to `A` but is not guaranteed to be a proper rotation (positive determinant) without a manual fix-up.

### `qd.eig(A, dt=None)`

General eigendecomposition for **2×2 only**. Returns:

- `eigenvalues`: a `Matrix(n, 2)` where row `i` is `(real_part, imaginary_part)` of the i-th eigenvalue.
- `eigenvectors`: a `Matrix(n*2, n)` where each column is an eigenvector with each entry expanded into two scalars (real, imaginary). For a 2×2 input, the eigenvector matrix is 4×2.

Eigenvalues of a real 2×2 matrix come in complex-conjugate pairs when the discriminant is negative; the packed layout keeps the function single-return-shape. Calling with `n=3` raises — for 3×3 general (non-symmetric) eigendecomposition, there is no Quadrants entry point today; the canonical path is `qd.sym_eig` if your matrix happens to be symmetric.

### `qd.solve(A, b, dt=None)`

Direct solve of `A @ x = b` via Gauss elimination with partial pivoting. Returns the solution vector `x`.

- Shapes 2×2 and 3×3.
- The implementation asserts `A.n == A.m` and `A.m == b.n`.
- Singular `A` is checked by a kernel `assert` (`"Matrix is singular in linear solve."`) inside the Gauss-elimination path. Kernel asserts only fire when the runtime is initialised with `qd.init(debug=True)` (see [debug](debug.md)) — under the default `debug=False` a singular input silently produces a divide-by-zero / NaN result with no diagnostic. If you need a signal in production, check singularity explicitly before calling `qd.solve` (e.g. `abs(A.determinant())` against a tolerance), or run development workloads with `debug=True` to catch the case.
- Each call factorises `A` from scratch and back-substitutes for the given `b`.

## Examples

### Closest rotation to a 3×3 matrix (ARAP)

```python
@qd.func
def closest_rotation(A: qd.types.matrix(3, 3, qd.f64)) -> qd.types.matrix(3, 3, qd.f64):
    U, S, V = qd.svd(A, dt=qd.f64)
    R = U @ V.transpose()
    if R.determinant() < 0.0:
        V[:, 2] *= -1.0
        R = U @ V.transpose()
    return R
```

The `R.determinant() < 0.0` branch fixes the handedness when SVD's sign convention produces a reflection rather than a rotation.

### Project to symmetric positive semi-definite

For shapes 4×4 .. 12×12 (a typical qipc 12×12 contact Hessian), use `qd.make_spd` directly:

```python
mat12 = qd.types.matrix(12, 12, qd.f64)


@qd.kernel
def project_each(
    H_field: qd.types.NDArray[mat12, 1],
    H_spd_field: qd.types.NDArray[mat12, 1],
) -> None:
    for i in range(H_field.shape[0]):
        H_spd_field[i] = qd.make_spd(H_field[i], dt=qd.f64)
```

Each thread eigen-decomposes its own 12×12 Hessian, clamps negative eigenvalues to zero, and reconstructs.

For 2×2 / 3×3 there is no `qd.make_spd` entry point (it shares the cyclic-Jacobi path that only kicks in at N≥4). Inline the projection using `qd.sym_eig` directly:

```python
@qd.func
def make_spd_3x3(H: qd.types.matrix(3, 3, qd.f64)) -> qd.types.matrix(3, 3, qd.f64):
    eigvals, Q = qd.sym_eig(H, dt=qd.f64)
    for i in qd.static(range(3)):
        if eigvals[i] < 0.0:
            eigvals[i] = 0.0
    Lambda = qd.Matrix.zero(qd.f64, 3, 3)
    for i in qd.static(range(3)):
        Lambda[i, i] = eigvals[i]
    return Q @ Lambda @ Q.transpose()
```

### Per-thread linear solve

```python
@qd.kernel
def solve_each(A_field: qd.types.NDArray[qd.types.matrix(2, 2, qd.f32), 1],
               b_field: qd.types.NDArray[qd.types.vector(2, qd.f32), 1],
               x_field: qd.types.NDArray[qd.types.vector(2, qd.f32), 1]) -> None:
    for i in range(A_field.shape[0]):
        x_field[i] = qd.solve(A_field[i], b_field[i])
```

Each thread does an independent Gauss elimination on its own 2×2 system. For larger systems a CG / PCG iteration over the whole array is the standard Quadrants pattern; `qd.solve` is for the per-element case.

### 2×2 polar decomposition for shape matching

```python
@qd.func
def shape_match(A: qd.types.matrix(2, 2, qd.f32)) -> qd.types.matrix(2, 2, qd.f32):
    R, _ = qd.polar_decompose(A)
    return R
```

The rotation factor `R` from `A = R @ S` is the rigid alignment that minimises `‖R - A‖_F` — the building block of position-based dynamics shape-matching.

## Shapes, performance, portability

- **Compile time.**
  - **Closed-form ops** (`qd.svd`, `qd.sym_eig` 2×2/3×3, `qd.polar_decompose`, `qd.eig`, `qd.solve`) — each call is unrolled per thread into a moderately large block of straight-line code; compile time is generally fine at these shapes.
  - **Cyclic Jacobi** (`qd.sym_eig` ≥4×4, `qd.make_spd`) — the per-pair Givens step is unrolled but the outer sweep loop is a runtime `range`, so compile time is roughly proportional to `N²` (number of `(p, q)` pairs per sweep) rather than `N² · MAX_SWEEPS`. Concrete numbers on CUDA + LLVM 22.1: ~3 s at N=4, ~30 s at N=6, ~3 min at N=9, ~2 min at N=12 (yes, faster than N=9 — the per-pair body is dominated by `if static(p < q):` filtering).
- **Runtime cost.** Cyclic Jacobi at N=12 with `MAX_SWEEPS=12` does roughly `12 · 66 · 12 ≈ 9500` per-thread arithmetic ops — fast on any modern GPU, but if you're calling it inside a hot kernel for a million elements that's still ~10 GFLOP-equivalent. For larger matrices use a different algorithm (or call quadrants `linalg.*` for a sparse-aware path).
- **Backend portability.** All ops compile cleanly on CUDA, AMDGPU, Vulkan, and Metal — they are pure register arithmetic with no SIMT primitives, so there is no codegen split. The `qd.sym_eig` ≥4×4 / `qd.make_spd` paths have been verified at N ∈ {4,5,6,9,12} × {f32, f64} × five symmetric-matrix factories on CUDA + Vulkan + AMDGPU; Metal coverage is via the same parametrized tests.

## Related

- [matrix_vector](matrix_vector.md) — `qd.Matrix` / `qd.Vector` data types, fields, ndarrays, type annotations.
- [matrix_vector_per_thread](matrix_vector_per_thread.md) — element-wise / arithmetic matrix operations (`@`, `inverse`, `determinant`, `transpose`, `dot`, `cross`, `norm`, `outer_product`, `frobenius_inner`). Per-thread, but closed-form rather than iterative.
- `qd.math.*` — scalar math helpers (`qd.math.dot`, `qd.math.cross`, `qd.math.length`, etc.) that operate on vectors / matrices but are not decompositions.
- `qd.linalg.*` — cross-thread / sparse-matrix linear algebra (CG, sparse solvers); a different namespace and a different problem class.
