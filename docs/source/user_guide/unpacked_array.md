# Unpacked array

`qd.unpacked_array(N, dtype)` is a `@qd.dataclass` field annotation that declares a group of `N` independently-allocated scalar fields exposed under a single name via indexed syntax.

It is a *layout hint*, not a new container. At source level you write `t.r[i]`; the compiler lowers that to a direct reference to a synthetic scalar field `t._r{i}`. The generated LLVM IR / PTX is byte-identical to a struct that was declared with `N` individually-named scalar fields.

## What problem does it solve?

The intuitive way to group `N` scalars on a per-thread struct is to declare a vector member:

```python
@qd.dataclass
class Tile:
    r: qd.types.vector(32, qd.f32)
```

`qd.types.vector(N, dtype)` lays the group out as a single packed `alloca`. LLVM's scalar-replacement-of-aggregates pass (SROA + `mem2reg`) tries to decompose that `alloca` into per-slot SSA values so each one can live in a register — but once a kernel's register pressure crosses a threshold (e.g. two concurrent `32x32` tiles in a Cholesky + triangular solve), SROA bails out on the packed `alloca` and the whole group spills to local memory as a unit. Each access then turns into a `ld.local` / `st.local` and the kernel slows down dramatically.

The alternative is to declare `N` named scalar fields by hand:

```python
@qd.dataclass
class Tile:
    r0: qd.f32
    r1: qd.f32
    # ... 30 more lines ...
    r31: qd.f32
```

Now each slot has its own `alloca`, and SROA + `mem2reg` can promote each one independently. The optimiser is also free to spill only the slots it has to, instead of the whole group as a unit. The cost is that every index becomes a cascade in source:

```python
def get_r(t, k):
    if k == 0:
        return t.r0
    elif k == 1:
        return t.r1
    # ... 30 more branches ...
```

…which is duplicated at every call site that wants to read or write the group.

`qd.unpacked_array` is the named-field layout with the ergonomic indexed syntax restored:

```python
@qd.dataclass
class Tile:
    r: qd.unpacked_array(32, qd.f32)
```

The annotation expands at struct-definition time into the `N` synthetic scalar fields. The AST transformer rewrites `obj.r[i]` (for any python-int / `qd.static`-resolved `i`) into a direct reference to the synthetic field `obj._r{i}`. The IR / PTX matches the hand-rolled named-field version exactly.

The name "unpacked" is a contrast with the packed-vector default: a packed group is one `alloca`, an unpacked group is `N` `alloca`s, one per slot. Whether the slots end up in registers is the optimiser's call; `unpacked_array` removes the layout obstacle that was preventing it.

## How to use it

Declare the group as an `unpacked_array(count, dtype)` annotation on a `@qd.dataclass`:

```python
import quadrants as qd

qd.init(arch=qd.gpu)


@qd.dataclass
class Tile:
    r: qd.unpacked_array(32, qd.f32)


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

You can mix `unpacked_array` groups with regular scalar / vector fields on the same dataclass; they are independent. You can also have several `unpacked_array` groups in one struct:

```python
@qd.dataclass
class TwoTiles:
    a: qd.unpacked_array(32, qd.f32)
    b: qd.unpacked_array(32, qd.f32)
    scale: qd.f32
```

The generated struct has 65 scalar members (`_a0..._a31`, `_b0..._b31`, `scale`).

## When to reach for it

Use `unpacked_array` when:

- the group is *small and statically-sized*, and
- the kernel body accesses it with python-int / `qd.static`-resolved indices (typically unrolled inner loops), and
- you have measured (or strongly suspect) that an equivalent `qd.types.vector(N, dtype)` is leaving slots in local memory under register pressure.

A good signal is `ptxas` reporting non-zero "bytes spill stores / loads" for the kernel, or `ld.local` / `st.local` instructions in the generated PTX that don't correspond to a deliberate shared-memory access.

Prefer `qd.types.vector(N, dtype)` for small groups where register pressure is low and runtime indexing is needed — vectors keep all the usual arithmetic conveniences (element-wise ops, dot products, etc.) that `unpacked_array` does not.

## Constraints and limitations

- **Static indices only.** `t.r[k]` must resolve at compile time, i.e. `k` is a python-int literal or a `for k in qd.static(range(N)):` loop variable. A runtime-int index raises a `QuadrantsSyntaxError` at compile time with a message pointing at the `qd.static` requirement. If you need a runtime index over the group, spell out the cascade explicitly (`if k == 0: ...`).
- **Static out-of-bounds is rejected at compile time.** `t.r[7]` on an `unpacked_array(4, ...)` group raises `QuadrantsSyntaxError: unpacked_array index out of bounds: r[7] (count=4)`.
- **Storage only.** An `unpacked_array` group has no vector arithmetic. There is no `t.r + other`, no `t.r.dot(...)`, no broadcast operations. If you want those, use `qd.types.vector(N, dtype)` instead.
- **`count` is fixed at struct-definition time.** It must be a positive python-int literal.
- **Naming.** The synthetic fields use the convention `_{group_name}{i}` (e.g. `_r0`, `_r1`, ..., `_r31`). Avoid declaring your own field with a name that collides with one of those, or `StructType` will report a duplicate member.

## Relationship to other annotations

| annotation                          | storage layout                  | runtime indexing | best for                              |
|-------------------------------------|---------------------------------|:----------------:|---------------------------------------|
| `qd.f32` (per-field)                | one `alloca` per field          | n/a              | individually-named scalars            |
| `qd.types.vector(N, dtype)`         | one packed `alloca`             | yes              | small groups with vector arithmetic   |
| `qd.unpacked_array(N, dtype)`       | `N` independent `alloca`s       | no               | groups that need to stay register-resident under pressure |

Under low register pressure the three options generate similar code. Under high register pressure `unpacked_array` is the one most likely to stay in registers because the optimiser can promote each slot independently.

## See also

- {doc}`compound_types` — `@qd.dataclass` overview
- {doc}`matrix_vector_per_thread` — `qd.types.vector` and per-thread matrices
- {doc}`linalg_per_thread` — examples of tile-resident linear algebra where register residency matters
