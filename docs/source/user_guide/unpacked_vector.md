# Unpacked vector

`qd.types.UnpackedVector[dtype, N]` is a `@qd.dataclass` field annotation that declares a group of `N` scalars exposed under a single name via indexed syntax (`t.r[i]`), laid out as `N` independent slots so the compiler can keep each one in a register independently.

## What problem does it solve?

The natural way to group `N` scalars on a per-thread struct is a vector member:

```python
@qd.dataclass
class Tile:
    r: qd.types.vector(32, qd.f32)
```

A `qd.types.vector(N, dtype)` is stored as a single group. The compiler either keeps the whole group in registers, or, if registers are tight, spills the whole group to "local" memory (which despite the name lives in off-chip DRAM and is slow). It is all-or-nothing per group.

An `UnpackedVector` is the same `N` scalars, but laid out so each slot is independent. The compiler can decide per slot whether to keep it in a register or spill it. Under register pressure this can be the difference between "most of the group still lives in registers" and "the entire group is in DRAM".

The alternative way to get the same per-slot layout is to declare `N` named scalar fields by hand:

```python
@qd.dataclass
class Tile:
    r0: qd.f32
    r1: qd.f32
    # ... 30 more lines ...
    r31: qd.f32
```

That works, but every read or write site has to spell out a cascade like `if k == 0: ... elif k == 1: ...` across all `N` slots -- ugly, error-prone, and slow to type-check and compile.

`qd.types.UnpackedVector` is the named-field layout with the indexed syntax restored:

```python
@qd.dataclass
class Tile:
    r: qd.types.UnpackedVector[qd.f32, 32]
```

The annotation expands at class-definition time into the `N` named-field equivalent, and the AST transformer rewrites each `obj.r[i]` (for python-int / `qd.static`-resolved `i`) into a direct reference to the i-th slot. So you get the best of both worlds: a clean indexed write/read at the source level, and the per-slot register residency of the hand-rolled named-field version.

The name "unpacked" contrasts with the packed vector layout above: a packed group is one slot, an unpacked group is `N` independent slots. Whether each slot ends up in a register is the optimiser's call; `UnpackedVector` removes the layout obstacle that was preventing it.

## How to use it

Declare the group as an `UnpackedVector[dtype, count]` annotation on a `@qd.dataclass`:

```python
import quadrants as qd

qd.init(arch=qd.gpu)


@qd.dataclass
class Tile:
    r: qd.types.UnpackedVector[qd.f32, 32]


@qd.kernel
def k(out: qd.types.NDArray[qd.f32, 1]) -> None:
    t = Tile()
    # python-int index: lowers to a direct write of t._r5
    t.r[5] = 1.0
    # qd.static loop variable: each iter is one AST node, fully unrolled,
    # no per-iter cascade.
    for i in qd.static(range(32)):
        t.r[i] = qd.f32(i)
    out[0] = t.r[3]
```

Read access works the same way:

```python
v = t.r[5]              # python-int index
v = t.r[i]              # i bound by `for i in qd.static(range(N)):`
```

You can mix `UnpackedVector` groups with regular scalar / vector fields on the same dataclass; they are independent. You can also have several `UnpackedVector` groups in one struct:

```python
@qd.dataclass
class TwoTiles:
    a: qd.types.UnpackedVector[qd.f32, 32]
    b: qd.types.UnpackedVector[qd.f32, 32]
    scale: qd.f32
```

The generated struct has 65 scalar members (`_a0..._a31`, `_b0..._b31`, `scale`).

## When to reach for it

Use `UnpackedVector` when:

- the group is *small relative to the per-thread register budget* (roughly speaking, a few dozen slots on current NVIDIA GPUs; an SM gives each thread up to 255 32-bit registers, shared with all live state in the kernel, so what counts as "small" depends on what else the kernel is holding), and
- the kernel body accesses it with python-int / `qd.static`-resolved indices (typically unrolled inner loops), and
- an equivalent `qd.types.vector(N, dtype)` version of the same kernel is, or might be, spilling the group to local memory under register pressure. See [How to check for spills](#how-to-check-for-spills) below for the concrete workflow to confirm a spill is actually happening.

Prefer `qd.types.vector(N, dtype)` for small groups where register pressure is low and runtime indexing is needed -- vectors keep all the usual arithmetic conveniences (element-wise ops, dot products, etc.) that `UnpackedVector` does not.

## Constraints and pitfalls

- **Use `@qd.dataclass`, not `@dataclasses.dataclass`.** The `UnpackedVector[dtype, count]` annotation is only expanded by `@qd.dataclass`; on a stdlib dataclass the annotation is inert metadata and the indexed-access syntax will not work. Subscripting or calling the marker outside that context raises a `QuadrantsSyntaxError`.
- **Subscript syntax, not a function call.** Write `r: qd.types.UnpackedVector[qd.f32, 32]`. There is no `qd.types.UnpackedVector(qd.f32, 32)` call form. The subscript spelling makes the marker visually read as a type annotation, not a runtime value.
- **Static indices only.** `t.r[k]` must resolve at compile time, i.e. `k` is a python-int literal or a `for k in qd.static(range(N)):` loop variable. A runtime-int index raises a `QuadrantsSyntaxError` at compile time. If you need a runtime index over the group, spell out the cascade explicitly. This is by design: the whole point of the layout is that each slot can be promoted to its own register, which requires every access site to name a specific slot at compile time.
- **No vector arithmetic.** An `UnpackedVector` group is storage only. There is no `t.r + other`, no `t.r.dot(...)`, no broadcast operations. If you want those, use `qd.types.vector(N, dtype)` instead.
- **Synthetic field-name collisions are rejected.** The expansion uses the convention `_{group_name}{i}` (e.g. `_r0`, `_r1`, ..., `_r31`). If you declare your own field with one of those names, `@qd.dataclass` raises a `QuadrantsSyntaxError` pointing at the collision.

## Relationship to other annotations

| annotation                          | storage layout                          | runtime indexing | best for                              |
|-------------------------------------|-----------------------------------------|:----------------:|---------------------------------------|
| `qd.f32` (per-field)                | one independent slot per field          | n/a              | individually-named scalars            |
| `qd.types.vector(N, dtype)`         | one slot holding `N` packed scalars     | yes              | small groups with vector arithmetic   |
| `qd.types.UnpackedVector[dtype, N]` | `N` independent slots                   | no               | groups that need to stay register-resident under pressure |

Under low register pressure the three options generate similar code. Under high register pressure `UnpackedVector` is the one most likely to stay in registers because the optimiser can promote each slot independently.

# Advanced

## How the layout works at the LLVM level

A `qd.types.vector(N, dtype)` field lowers to a single `alloca` of `N` packed scalars. LLVM's SROA + `mem2reg` passes attempt to decompose that `alloca` into `N` per-slot SSA values so each can live in a register, but the decomposition is conservative: under high register pressure (e.g. two concurrent `32x32` tiles in a Cholesky + triangular solve), SROA bails out on the packed `alloca` and the whole group spills to local memory as a unit. Each access then becomes a `ld.local` / `st.local`.

An `UnpackedVector[dtype, N]` field expands to `N` independent `alloca` instructions, one per slot, so `mem2reg` can promote each slot independently and the register allocator can spill only the slots it has to. That is exactly what the hand-rolled `r0..r{N-1}` form produces; the generated LLVM IR / PTX matches it byte-for-byte.

## How to check for spills

Quadrants compiles LLVM -> PTX directly via the LLVM NVPTX target -- it does not invoke `nvcc` / `ptxas` at compile time, so the familiar `ptxas --verbose` "X bytes spill stores" output is not produced inline. To get that diagnostic you dump the PTX from Quadrants and run `ptxas -v` on it yourself. Three workflows, in order of usefulness:

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

`spill stores` + `spill loads` > 0 means the register allocator gave up on something. `stack frame` > 0 indicates local memory in use (which is usually, but not always, spills -- `qd.simt.shared_array` and explicitly addressed locals also count).

Adjust `-arch=sm_XX` to match your GPU (`sm_86` = Ampere consumer / RTX 30, `sm_89` = Ada / RTX 40, `sm_90` = Hopper).

### 2. Inspect the PTX directly for `ld.local` / `st.local`

```bash
grep -nE "ld\.local|st\.local" quadrants_kernel_nvptx_0007.ptx | head -20
```

Any matches that aren't from deliberate `qd.simt.shared_array` accesses (which show up as `ld.shared` / `st.shared`, not `.local`) are spill traffic.

### 3. Runtime spill counters via Nsight Compute (most authoritative)

```bash
ncu --set full --section MemoryWorkloadAnalysis ./your_program
```

Look at the "Memory Workload Analysis -> Local Memory" section. This reports *actually executed* local-memory loads / stores, which catches issues `ptxas` doesn't (e.g. driver-stage JIT spills on a different GPU, hot-path-only spills that static analysis misses).

### Also useful: post-optimisation LLVM IR

```python
qd.init(arch=qd.cuda, print_kernel_llvm_ir_optimized=True)
```

Dumps `quadrants_kernel_cuda_llvm_ir_optimized_NNNN.ll`. Look for `alloca` instructions that survived `mem2reg` -- those will become PTX local memory.

### Gotcha: the offline cache

`print_kernel_asm` and `print_kernel_llvm_ir*` only fire on the first compile per kernel-key. If your kernel is already in the offline cache, the dump won't be regenerated. Either:

- pass `offline_cache=False` to `qd.init` (cleanest), or
- delete the cache (path is printed at `qd.init` time; typically under `~/.cache/quadrants/`).

## See also

- {doc}`compound_types` -- `@qd.dataclass` overview
- {doc}`matrix_vector_per_thread` -- `qd.types.vector` and per-thread matrices
- {doc}`linalg_per_thread` -- examples of tile-resident linear algebra where register residency matters
