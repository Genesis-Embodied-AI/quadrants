# Matrix and Vector

Quadrants provides `qd.Matrix` and `qd.Vector` types for small, fixed-size linear algebra inside kernels. These are stored in GPU registers and are unrolled at compile time, so they are fast — but should be kept small for best performance (more than 144 elements total — i.e. larger than 12×12 — will trigger a warning).

`qd.Vector` is a special case of `qd.Matrix` with one column.

## Creating vectors and matrices in kernels

```python
@qd.kernel
def compute() -> None:
    v = qd.Vector([1.0, 2.0, 3.0])
    m = qd.Matrix([[1.0, 0.0], [0.0, 1.0]])

    # Access elements
    x = v[0]          # 1.0
    val = m[0, 1]     # 0.0
```

## Vector and matrix fields

To store many vectors or matrices, use vector/matrix fields:

```python
import quadrants as qd

qd.init(arch=qd.gpu)

# first define the type
vec3f = qd.types.vector(3, qd.f32)

# A field of 100 3D vectors
positions = qd.field(vec3f, shape=(100,))

# or for ndarray
positions = qd.ndarray(vec3f, shape=(100,))

# first define the type
mat3f = qd.types.matrix(3, 3, qd.f32)

# A field of 100 3x3 matrices
rotations = qd.field(mat3f, shape=(100,))

# or for ndarray
rotations = qd.ndarray(mat3f, shape=(100,))

@qd.kernel
def initialize(pos: qd.Template, rot: qd.Template) -> None:
    for i in range(100):
        pos[i] = qd.Vector([0.0, 0.0, 0.0])
        rot[i] = qd.Matrix.diag(3, 1.0)  # identity matrix

initialize(positions, rotations)
```

## Vector and matrix ndarrays

```python
# An ndarray of 50 3D vectors
vel = qd.Vector.ndarray(3, qd.f32, shape=(50,))

# An ndarray of 50 4x4 matrices
transforms = qd.Matrix.ndarray(4, 4, qd.f32, shape=(50,))
```

## Type annotations

For kernel and `@qd.func` parameters, use `qd.types.vector` and `qd.types.matrix`:

```python
vec3f = qd.types.vector(3, qd.f32)
mat3f = qd.types.matrix(3, 3, qd.f32)

@qd.kernel
def transform(
    positions: qd.types.NDArray[vec3f, 1],
    matrices: qd.types.NDArray[mat3f, 1],
    out: qd.types.NDArray[vec3f, 1],
) -> None:
    for i in range(100):
        out[i] = matrices[i] @ positions[i]
```

## Operations

Element-wise arithmetic, dot / cross / outer products, norms, transpose / determinant / trace / inverse, Frobenius inner product, and matrix-vector / matrix-matrix multiply all live on a separate page since they all run **per thread, in registers, with no cross-thread cooperation**. See [matrix_vector_per_thread](matrix_vector_per_thread.md).

For per-thread *numerical* algorithms (SVD, symmetric eigendecomposition, polar decomposition, linear solve) see [linalg_per_thread](linalg_per_thread.md).

For cross-thread / sparse linear algebra (CG, sparse direct solvers) see `qd.linalg.*` (separate, not yet covered in user guide).

## Size limit

Matrices and vectors are unrolled into scalar registers at compile time. Using more than 144 elements (i.e. larger than 12×12) will trigger a warning and may cause very long compile times. For larger matrices, use a field instead:

```python
large = qd.field(qd.f32, shape=(64, 64))
```

The 144-element threshold matches the largest size officially supported by the per-thread linalg APIs (`qd.sym_eig` and `qd.make_spd` up to 12×12, `Matrix.inverse` up to 12×12) — those internal constructions stay below the warning threshold.
