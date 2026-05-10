# Matrix and Vector

Quadrants provides `qd.Matrix` and `qd.Vector` types for small, fixed-size linear algebra inside kernels. These are stored in GPU registers and are unrolled at compile time, so they are fast — but should be kept small for best performance (more than 32 elements total will trigger a warning).

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

### Arithmetic

Standard arithmetic works element-wise:

```python
@qd.func
def example() -> None:
    a = qd.Vector([1.0, 2.0, 3.0])
    b = qd.Vector([4.0, 5.0, 6.0])

    c = a + b       # [5.0, 7.0, 9.0]
    d = a * 2.0     # [2.0, 4.0, 6.0]
    e = a * b       # element-wise: [4.0, 10.0, 18.0]
```

### Dot and cross product

```python
@qd.func
def products() -> None:
    a = qd.Vector([1.0, 0.0, 0.0])
    b = qd.Vector([0.0, 1.0, 0.0])

    d = a.dot(b)      # 0.0
    c = a.cross(b)    # [0.0, 0.0, 1.0]
```

`cross` works for 2D vectors (returns a scalar) and 3D vectors (returns a vector).

### Norm and normalize

```python
@qd.func
def norms() -> None:
    v = qd.Vector([3.0, 4.0])

    length = v.norm()              # 5.0
    length_sq = v.norm_sqr()       # 25.0
    unit = v.normalized()          # [0.6, 0.8]
    inv_len = v.norm_inv()         # 0.2
```

Pass an `eps` argument for numerical safety: `v.normalized(eps=1e-8)`.

### Matrix operations

```python
@qd.func
def mat_ops() -> None:
    m = qd.Matrix([[1.0, 2.0], [3.0, 4.0]])

    t = m.transpose()       # [[1, 3], [2, 4]]
    d = m.determinant()     # -2.0
    tr = m.trace()          # 5.0
    inv = m.inverse()       # [[-2, 1], [1.5, -0.5]]
```

- `determinant()` supports matrices up to 4×4 (closed-form expansion).
- `inverse()` supports matrices up to 12×12. Sizes 1–4 use the closed-form cofactor expansion; sizes 5–12 use Gauss elimination with partial pivoting (fully unrolled). For the larger sizes, achievable precision scales as `cond(A) · eps` — a well-conditioned 12×12 in `f64` typically reconstructs to ~1e-12.

### Frobenius inner product and norm

```python
@qd.func
def inner_products() -> None:
    a = qd.Matrix([[1.0, 2.0], [3.0, 4.0]])
    b = qd.Matrix([[0.0, 1.0], [1.0, 0.0]])

    s = a.frobenius_inner(b)   # 2 + 3 = 5.0   (sum_ij a[i,j] * b[i,j])
    n = a.norm()               # sqrt(1 + 4 + 9 + 16) = sqrt(30)
    n_sq = a.norm_sqr()        # 30.0  (== a.frobenius_inner(a))
```

`frobenius_inner(other)` requires both matrices to have the same shape and supports any size. It's the natural inner product on matrices viewed as vectors of length `n × m` and is the correct bilinear form behind `norm` / `norm_sqr` (`A.frobenius_inner(A) == A.norm_sqr()`).

### Matrix-vector multiply

Use the `@` operator:

```python
@qd.func
def mat_vec() -> None:
    m = qd.Matrix([[1.0, 0.0], [0.0, 2.0]])
    v = qd.Vector([3.0, 4.0])

    result = m @ v    # [3.0, 8.0]
```

### Other operations

- `qd.Matrix.diag(dim, val)` — create a diagonal matrix
- `a.outer_product(b)` — outer product of two vectors

## Size limit

Matrices and vectors are unrolled into scalar registers at compile time. Using more than 32 elements (e.g. a 6x6 matrix) will trigger a warning and may cause very long compile times. For larger matrices, use a field instead:

```python
large = qd.field(qd.f32, shape=(64, 64))
```
