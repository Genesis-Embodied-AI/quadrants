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

## Storage layout: packed vs unpacked vectors

`qd.types.vector(N, dtype)` accepts an optional `unpacked: bool = False` kwarg that controls how the vector is laid out in per-thread memory when used as a `@qd.dataclass` field. The kwarg does not affect the *value type* — the field still holds a vector — it only affects the storage layout.

- `unpacked=False` (default): the field is stored as a single group of `N` scalars. The compiler keeps the whole vector in registers, or, if there is insufficient register storage, spills the whole vector to "local" memory (which despite the name lives in off-chip DRAM and is slow). It is all-or-nothing per vector.
- `unpacked=True`: the field is stored as `N` independent scalar slots. The compiler can decide per slot whether to keep it in a register or spill it. Under register pressure this can be the difference between "most of the vector still lives in registers" and "the entire vector is in DRAM".

```python
@qd.dataclass
class Tile:
    r: qd.types.vector(32, qd.f32, unpacked=True)
```

The flag is consumed at class-definition time. `@qd.dataclass` expands the field into `N` synthetic scalar members `_r0.._r{N-1}`, and each `obj.r[i]` (for python-int / `qd.static`-resolved `i`) is lowered to a direct reference to the i-th synthetic member. The generated LLVM IR / PTX is byte-identical to declaring `N` named scalar fields by hand (`r0: qd.f32`, `r1: qd.f32`, ...), but the source keeps the indexed read/write syntax.

### When to choose `unpacked=True` vs `unpacked=False`

| If you...                                                                                                | Use                              |
|----------------------------------------------------------------------------------------------------------|----------------------------------|
| want vector arithmetic (`a + b`, `a.dot(b)`, `a.norm()`, swizzle, broadcast) at the use site             | `unpacked=False` (the default)   |
| need to index with a *runtime* integer (`v[k]` where `k` isn't a python-int)                             | `unpacked=False`                 |
| need to pass the vector across a `@qd.func` boundary, copy it into shared memory, or instantiate it as a kernel-local value | `unpacked=False`                 |
| are storing the field in a smallish struct, register pressure is low, and the kernel works as-is        | `unpacked=False`                 |
| only ever access the field as `obj.r[i]` with python-int / `qd.static`-resolved `i` (unrolled inner loops), the group is small relative to the per-thread register budget, *and* the packed version is spilling under register pressure | `unpacked=True`                  |

With the default packed layout, all `N` elements share a single piece of stack memory, and the optimiser's "promote this to registers vs leave it in memory" decision is made once for the whole group: either every element gets a register, or none do and the whole vector lives in slow off-chip memory. Splitting the field into `N` independent pieces of memory up front gives the optimiser `N` independent promote-or-leave-in-memory decisions, so it can keep as many elements in registers as actually fit and leave the rest in memory.

As a rough rule on current NVIDIA GPUs, the practical per-thread register budget is around 32 to 64 32-bit registers. The architectural cap is 255, but using anywhere near that drops occupancy (the number of warps the SM can keep resident in parallel) sharply, so any production kernel ends up well below the cap. What counts as "small relative to the per-thread register budget" therefore depends on the occupancy target and what else is live. The pattern that motivates `unpacked=True` is high register pressure — e.g. several concurrent 32×32 tiles in a Cholesky + triangular solve — where the compiler gives up on the packed group as a whole and the whole vector ends up in local memory.

See [How to check for spills](#how-to-check-for-spills) below for how to confirm a spill is actually happening before reaching for `unpacked=True`.

### Constraints

- **Only valid as a `@qd.dataclass` field annotation.** The `unpacked=True` flag is consumed by `@qd.dataclass`. Attempting to instantiate the type as a value (`T(...)`, `T.field(...)`, `T.ndarray(...)`), use it as a kernel-arg / `@qd.func` parameter type, or apply it on a `@dataclasses.dataclass` raises `QuadrantsSyntaxError`. If you need a value type, drop `unpacked=True`.
- **Indexed access only.** The supported pattern is `obj.r[i]` for python-int / `qd.static`-resolved `i`. Operations that treat `obj.r` itself as a vector value — vector arithmetic, reductions, swizzle, runtime indexing, passing the field across a `@qd.func` boundary, copying into shared memory as a unit — are not supported on an unpacked field today, even though the value type is a vector. If you need any of those, use plain `qd.types.vector(N, dtype)`.
- **Synthetic field-name collisions are rejected.** The expansion uses the convention `_{group_name}{i}` (e.g. `_r0`, `_r1`, ..., `_r31`). Declaring your own field with one of those names raises `QuadrantsSyntaxError` at class-decoration time.

### Mixing with other fields

`unpacked=True` fields compose with anything else allowed on `@qd.dataclass`: scalars, plain vectors, matrices, other unpacked vectors, nested dataclasses.

```python
@qd.dataclass
class TwoTiles:
    a: qd.types.vector(32, qd.f32, unpacked=True)
    b: qd.types.vector(32, qd.f32, unpacked=True)
    scale: qd.f32
```

The generated struct in this example has 65 scalar members (`_a0..._a31`, `_b0..._b31`, `scale`).

# Advanced

## How the packed vs unpacked layout differs at the LLVM level

A plain `qd.types.vector(N, dtype)` field on a `@qd.dataclass` lowers to a single stack-allocated group of `N` packed scalars. LLVM's optimiser attempts to decompose that group into `N` per-slot register-resident values, but the decomposition is conservative: under high register pressure (e.g. two concurrent 32×32 tiles in a Cholesky + triangular solve) the optimiser bails out and the whole vector spills to local memory as a unit.

`qd.types.vector(N, dtype, unpacked=True)` expands to `N` independent scalar stack slots, one per element, so the optimiser can promote each slot to a register independently and the register allocator can spill only the slots it has to. That is exactly what the hand-rolled `r0..r{N-1}` form produces; the generated LLVM IR / PTX matches it byte-for-byte.

## How to check for spills

Quadrants compiles LLVM -> PTX directly via the LLVM NVPTX target — it does not invoke `nvcc` / `ptxas` at compile time, so the familiar `ptxas --verbose` "X bytes spill stores" output is not produced inline. To get that diagnostic you dump the PTX from Quadrants and run `ptxas -v` on it yourself. Three workflows, in order of usefulness:

### 1. Dump PTX, run `ptxas -v` offline

```python
qd.init(
    arch=qd.cuda,
    print_kernel_asm=True,    # dump PTX to CWD as quadrants_kernel_nvptx_NNNN.ptx
    offline_cache=False,      # bypass the kernel cache so the dump fires every time
)
```

Run your kernel once. For each `quadrants_kernel_nvptx_NNNN.ptx` file in the working directory:

```bash
ptxas --verbose -arch=sm_86 quadrants_kernel_nvptx_0007.ptx -o /dev/null 2>&1 | grep -E "spill|stack"
```

Sample output:

```
ptxas info    : Used 64 registers, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, 384 bytes stack frame, 256 bytes spill stores, 128 bytes spill loads
```

`spill stores` + `spill loads` > 0 means the register allocator gave up on something. `stack frame` > 0 indicates local memory in use (which is usually, but not always, spills — `qd.simt.shared_array` and explicitly addressed locals also count).

Adjust `-arch=sm_XX` to match your GPU (`sm_86` = Ampere consumer / RTX 30, `sm_89` = Ada / RTX 40, `sm_90` = Hopper).

### 2. Inspect the PTX directly for local-memory loads / stores

NVIDIA's PTX assembly uses the mnemonics `ld.local` and `st.local` for loads and stores against per-thread local memory (off-chip DRAM). Any kernel that spills must contain these mnemonics. Grep the dumped PTX for them:

```bash
grep -nE "ld\.local|st\.local" quadrants_kernel_nvptx_0007.ptx | head -20
```

Matches that aren't from deliberate `qd.simt.shared_array` accesses (those use the shared-memory mnemonics `ld.shared` / `st.shared`, not the `.local` ones) are spill traffic.

### 3. Runtime spill counters via Nsight Compute (most authoritative)

```bash
ncu --set full --section MemoryWorkloadAnalysis ./your_program
```

Look at the "Memory Workload Analysis -> Local Memory" section. This reports *actually executed* local-memory loads / stores, which catches issues `ptxas` doesn't (e.g. driver-stage JIT spills on a different GPU, hot-path-only spills that static analysis misses).

### Also useful: post-optimisation LLVM IR

```python
qd.init(arch=qd.cuda, print_kernel_llvm_ir_optimized=True)
```

Dumps `quadrants_kernel_cuda_llvm_ir_optimized_NNNN.ll`. In LLVM IR, every per-function stack allocation appears as an `alloca` instruction. The optimiser tries to promote each `alloca` into a register-resident value; any `alloca` that survives the optimiser into the post-optimisation dump is a stack slot it couldn't promote, and it will become PTX local memory. Grep for them:

```bash
grep -nE "alloca" quadrants_kernel_cuda_llvm_ir_optimized_0007.ll | head
```

### Gotcha: the offline cache

`print_kernel_asm` and `print_kernel_llvm_ir*` only fire on the first compile per kernel-key. If your kernel is already in the offline cache, the dump won't be regenerated. Either:

- pass `offline_cache=False` to `qd.init` (cleanest), or
- delete the cache (path is printed at `qd.init` time; typically under `~/.cache/quadrants/`).
