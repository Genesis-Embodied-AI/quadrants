# Matrix decompositions and solvers

Small matrix decompositions and linear solvers - the kinds of operations a thread does on a 2×2 or 3×3 matrix held in registers. They are a different category from the element-wise / arithmetic matrix operations covered in [matrix_vector](matrix_vector.md): each entry point here implements a *numerical algorithm* (Jacobi sweeps, Gauss elimination, Givens rotations) rather than a single closed-form formula.

All ops live at the top level (`qd.svd`, `qd.sym_eig`, `qd.polar_decompose`, `qd.eig`, `qd.solve`) and are intended to be called from inside a `@qd.kernel` or `@qd.func`. They run per thread - each thread independently decomposes its own matrix.

## What's available

| Op                              | Operates on            | Shapes        | Returns                                                |
|---------------------------------|------------------------|---------------|--------------------------------------------------------|
| `qd.svd(A)`                     | square real matrix     | 2×2, 3×3      | `(U, S, V)` such that `A = U @ S @ V.transpose()`      |
| `qd.sym_eig(A)`                 | symmetric real matrix  | 2×2, 3×3      | `(eigenvalues, eigenvectors)` (real)                   |
| `qd.polar_decompose(A)`         | square real matrix     | 2×2, 3×3      | `(R, S)` such that `A = R @ S`, `R` orthogonal, `S` SPD |
| `qd.eig(A)`                     | square real matrix     | 2×2           | `(eigenvalues, eigenvectors)` (complex, packed)        |
| `qd.solve(A, b)`                | square `A` + vector `b`| 2×2, 3×3      | `x` such that `A @ x = b`                              |

A few patterns to note:

- **Shapes are fixed.** Calling any of these on a matrix outside the supported shapes raises an exception at compile time (`"SVD only supports 2D and 3D matrices."`, etc.). Larger matrices need a different path - typically a Jacobi-style sweep applied iteratively, which Quadrants does not currently provide out of the box.
  - **FIXME (message wording):** these exception strings are misleading - "2D matrix" / "3D matrix" conventionally means "rank-2 / rank-3 tensor" (any matrix is rank-2), but here the intent is "matrix of shape 2×2 / 3×3". They should be updated to e.g. `"SVD only supports 2×2 and 3×3 matrices."`. This page reproduces the messages as they are emitted today.
- **All ops accept an optional `dt` argument.** When unspecified, it defaults to `impl.get_runtime().default_fp` - usually `qd.f32` unless overridden in `qd.init()`. Pass `dt=qd.f64` for the high-precision variant.
- **Output shape matches the input shape.** A 3×3 input yields 3×3 outputs (and a length-3 vector for `solve` / eigenvalues); a 2×2 input yields 2×2 outputs.
- **Real matrices only.** `qd.eig` returns complex results in a packed real layout (see below); the others all assume real-valued input and return real-valued output.

## Semantics

### `qd.svd(A, dt=None)`

Singular value decomposition - produces `(U, S, V)` such that `A = U @ S @ V.transpose()`, with:

- `U` orthogonal (`U @ U.transpose() = I`).
- `V` orthogonal.
- `S` diagonal, with non-negative singular values.

Shapes 2×2 and 3×3 use closed-form / Jacobi-style implementations specialized per shape. Sign convention for `U` and `V` is the implementation's natural one and is **not** guaranteed to enforce `det(U) = det(V) = +1`; if you depend on a particular handedness (e.g. for an ARAP rotation `R = U @ V.transpose()`), check it explicitly and flip a column if needed.

The 2×2 path returns singular values sorted descending (`S[0,0] >= S[1,1]`). The 3×3 path does **not** sort - singular values come out in whatever order the Sifakis algorithm produces them. If you depend on `S[0,0] >= S[1,1] >= S[2,2]` for 3×3, sort explicitly.

**FIXME (sort consistency):** the 2×2 / 3×3 split is an inconsistency in the API - both shapes should either sort descending or both should leave the order to the algorithm (or parametrize it with a boolean template parameter, e.g. `qd.svd(A, sorted=True)`). The 2×2 swap is essentially free; making 3×3 sort would cost a few comparisons and column swaps. Pick one and apply it across both shapes (and update this paragraph accordingly).

### `qd.sym_eig(A, dt=None)`

Symmetric eigendecomposition - for a real symmetric `A`, returns `(eigenvalues, eigenvectors)` with:

- `eigenvalues`: a `Vector(n)` of real eigenvalues.
- `eigenvectors`: a `Matrix(n, n)` whose columns are the corresponding orthonormal eigenvectors.

For 3×3, eigenvalues come out sorted in ascending order (the implementation explicitly sorts at the end). The 2×2 path does not perform an explicit sort - verify with a test if you need a particular order in 2D.

**FIXME (sort consistency):** same kind of inconsistency as `qd.svd`, with the additional twist that the sort directions disagree across ops - `qd.sym_eig` 3×3 sorts *ascending*, while `qd.svd` 2×2 sorts *descending*. Both shapes of `qd.sym_eig` should sort the same way, and ideally `qd.sym_eig` and `qd.svd` should agree on a direction (or parametrize via a boolean template parameter, e.g. `qd.sym_eig(A, sorted=True)`). Pick one and apply it across both shapes and across both ops (and update this paragraph accordingly).

`A` is *assumed* symmetric; the implementation does not symmetrize first. If your matrix is only approximately symmetric (e.g. accumulated floating-point error), explicitly compute `(A + A.transpose()) * 0.5` before calling.

### `qd.polar_decompose(A, dt=None)`

Polar decomposition - produces `(R, S)` such that `A = R @ S`, with:

- `R` orthogonal (the closest-rotation factor, modulo handedness).
- `S` symmetric positive semi-definite (the stretch factor).

Built on top of SVD: `A = U @ Σ @ Vᵀ` ⇒ `R = U @ Vᵀ`, `S = V @ Σ @ Vᵀ`. The same caveats about sign convention as `qd.svd` apply - `R` is the closest orthogonal factor to `A` but is not guaranteed to be a proper rotation (positive determinant) without a manual fix-up.

### `qd.eig(A, dt=None)`

General eigendecomposition for **2×2 only**. Returns:

- `eigenvalues`: a `Matrix(n, 2)` where row `i` is `(real_part, imaginary_part)` of the i-th eigenvalue.
- `eigenvectors`: a `Matrix(n*2, n)` where each column is an eigenvector with each entry expanded into two scalars (real, imaginary). For a 2×2 input, the eigenvector matrix is 4×2.

Eigenvalues of a real 2×2 matrix come in complex-conjugate pairs when the discriminant is negative; the packed layout keeps the function single-return-shape. Calling with `n=3` raises - for 3×3 general (non-symmetric) eigendecomposition, there is no Quadrants entry point today; the canonical path is `qd.sym_eig` if your matrix happens to be symmetric.

### `qd.solve(A, b, dt=None)`

Direct solve of `A @ x = b` via Gauss elimination with partial pivoting. Returns the solution vector `x`.

- Shapes 2×2 and 3×3.
- The implementation asserts `A.n == A.m` and `A.m == b.n`.
- Singular `A` is checked by a kernel `assert` (`"Matrix is singular in linear solve."`) inside the Gauss-elimination path. Kernel asserts only fire when the runtime is initialised with `qd.init(debug=True)` (see [debug](debug.md)) - under the default `debug=False` a singular input silently produces a divide-by-zero / NaN result with no diagnostic. If you need a signal in production, check singularity explicitly before calling `qd.solve` (e.g. `abs(A.determinant())` against a tolerance), or run development workloads with `debug=True` to catch the case.
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

### Project to symmetric positive semi-definite (`make_spd`)

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

Used to project an indefinite Hessian to its closest SPD approximation per element. Note the shape cap - only 3×3 today, since `qd.sym_eig` itself caps at 3×3.

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

The rotation factor `R` from `A = R @ S` is the rigid alignment that minimises `‖R - A‖_F` - the building block of position-based dynamics shape-matching.

## Shapes, performance, portability

- **Compile time.** Each call is unrolled per thread, so a kernel that calls `qd.svd` on a 3×3 matrix per element compiles a moderately large block of straight-line code per thread. Compile time is generally fine at these shapes; matrices larger than the cap may not be - register pressure plus unrolling explode quickly.
- **Backend portability.** All ops compile cleanly on CUDA, AMDGPU, Vulkan, and Metal - they are pure register arithmetic with no SIMT primitives, so there is no codegen split.

## Related

- [matrix_vector](matrix_vector.md) - element-wise / arithmetic matrix operations (`@`, `inverse`, `determinant`, `transpose`, dot, cross, norm, `outer_product`). Covers the operations whose implementation is a single closed-form formula.
- `qd.math.*` - scalar math helpers (`qd.math.dot`, `qd.math.cross`, `qd.math.length`, etc.) that operate on vectors / matrices but are not decompositions.
- `qd.linalg.*` - sparse-matrix linear algebra (CG, sparse solvers); a different namespace and a different problem class.
