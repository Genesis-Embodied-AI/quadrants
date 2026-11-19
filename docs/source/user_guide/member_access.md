# Querying kernel member access

Call `kernel.get_member_access(*args, **kwargs)` to find which arrays a kernel uses, reads, and writes for the supplied arguments. Use this information to allocate only the members a configuration needs, or to identify buffers that a launch recorder must preserve before another kernel overwrites them.

The query accepts the same positional arguments, keyword arguments, and defaults as a normal kernel call. It evaluates compile-time code and analyzes the selected specialization without compiling device code or launching the kernel. Call `qd.init()` first. Array contents are neither inspected nor changed by the analysis; ordinary Python code executed during compile-time evaluation retains its usual side effects.

## Result and access semantics

The result is a `qd.MemberAccess` named tuple with three immutable sets:

```python
from typing import NamedTuple

Path = tuple[str | int, ...]


class MemberAccess(NamedTuple):
    used: frozenset[Path]
    read: frozenset[Path]
    written: frozenset[Path]
```

Each path identifies a whole array, not individual elements or slices. Only array leaves appear in these sets; containers, functions, scalar parameters, and scalar members do not. `read` and `written` are subsets of `used`. An empty set means the analysis found no accesses of that kind, not that analysis was unavailable.

| Set | Meaning |
| --- | --- |
| `used` | Arrays required by the selected frontend specialization, including arrays used only for metadata such as `.shape`. |
| `read` | Arrays whose contents may be loaded. Reading `.shape`, dtype, or other array metadata alone does not read contents. |
| `written` | Arrays whose contents may be stored to. Assignments, augmented assignments, atomic updates, and writes through supported in-kernel operations are included. |

An augmented assignment or atomic update counts as both a read and a write. A plain element assignment counts as a write; its right-hand side contributes its own reads. Filling a local vector and then assigning it into an array counts as a write to that array. Calling an array's host-side `fill()` outside the kernel is a separate operation and is not part of the query.

These are conservative frontend access sets. Both sides of a runtime conditional contribute, and a runtime loop contributes even if its trip count happens to be zero at launch. A `qd.static(...)` conditional contributes only its selected branch. Accesses are reported before backend optimization, so an access optimized away later can still appear. The query does not inspect runtime values to decide which branch executes or which elements are touched.

Calls to `@qd.func` are included in the caller's result, with parameter paths translated back to the kernel arguments. Separate static instantiations of a helper are analyzed separately: calling the same helper with different static flags does not transfer one invocation's member accesses to the other invocation's arguments.

## Query before the first launch

This example queries an update without changing any array contents:

```python
from dataclasses import dataclass

import quadrants as qd

qd.init(arch=qd.cpu)


@dataclass
class State:
    position: object
    velocity: object
    force: object


@qd.func
def update(state: qd.RuntimeTyped, use_force: qd.Template):
    for i in range(state.position.shape[0]):
        state.position[i] += state.velocity[i]
        if qd.static(use_force):
            state.velocity[i] += state.force[i]


@qd.kernel
def advance(state: qd.RuntimeTyped, use_force: qd.Template):
    update(state, use_force)


state = State(*(qd.ndarray(qd.f32, shape=1) for _ in range(3)))
access = advance.get_member_access(state, use_force=False)

assert access.used == frozenset({("state", "position"), ("state", "velocity")})
assert access.read == access.used
assert access.written == frozenset({("state", "position")})

with_force = advance.get_member_access(state, use_force=True)
assert with_force.read == frozenset(
    {("state", "position"), ("state", "velocity"), ("state", "force")}
)
assert with_force.written == frozenset({("state", "position"), ("state", "velocity")})
```

For a plain array parameter, the path has one component. For example, `output[i] = input[i]` reads `("input",)` and writes `("output",)`.

## Paths through containers and handlers

The first component is the kernel parameter's name. Subsequent strings select attributes, and integers select tuple elements. Names are preserved verbatim; paths are not strings split on dots or internal compiler separators. A `qd.Tensor` is an array leaf: its backing ndarray or Field does not add an extra path component.

For the `apply(values, handlers)` example in [Tuples of handlers](handler_tuples.md), querying two `Bound` handlers returns:

```python
access = apply.get_member_access(values, (Bound(add, first_data), Bound(add, second_data)))

assert access.read == frozenset(
    {
        ("values",),
        ("handlers", 0, "data", 0, "increments"),
        ("handlers", 1, "data", 0, "increments"),
    }
)
assert access.written == frozenset({("values",)})
assert access.used == access.read
```

Nested tuples add another integer for each level. An empty handler tuple contributes no captured-array accesses. Handler functions and their captured static configuration select the analyzed specialization just as they do for a launch.

For a bound kernel, call `solver.step.get_member_access(...)` without supplying `self`. Arrays reached through the receiver retain paths such as `("self", "state", "velocity")`. If a tuple contains a bound `@qd.func` method directly, the path to its receiver uses the bound callable's `instance` attribute, for example `("handlers", 0, "instance", "weights")`.

### Aliases

If multiple supported argument paths reach the same array, every such path appears in each applicable set. For example, if `source` and `destination` reference the same ndarray, a write through `destination` puts both `("source",)` and `("destination",)` in `written` and `used`. If there is also a read through `source`, both paths appear in `read`. This rule also applies across direct arguments and handler captures, and to distinct `qd.Tensor` wrappers around the same backing array.

Paths do not revisit the same container within a single path. This keeps the result finite for supported object graphs with back-references, while retaining separate paths through shared containers. Handler tuple captures retain their existing prohibition on cycles.

The query recognizes shared backing objects, not arbitrary overlapping memory regions. Distinct NumPy or PyTorch views can overlap without being identified as aliases. A recorder must account for that storage overlap itself. Deduplicate snapshots by backing storage rather than by the number of returned paths.

## Planning allocation with placeholders

For ndarray arguments, query with small placeholder arrays before allocating full simulation buffers. Preserve the argument container types and structure, array element dtype and vector/matrix element shape, rank, layout, gradient configuration, and all compile-time values. Preserve sharing as well: two paths that alias in the real arguments must alias in the placeholders. Use separate placeholders for arrays that will have separate storage.

Array extents may differ when they are used only at runtime. If an extent participates in compile-time evaluation, preserve that extent too. Fields are tied to their actual field declarations; replacing a Field with an ndarray does not stand in for the same specialization.

For each kernel and static configuration that the scene can launch, query its placeholder arguments and take the union of `used` paths mapped into the scene's data model. Allocate full-sized storage for those arrays. Keep correctly typed placeholders for members absent from that union. In the `advance(..., use_force=False)` example, `force` can remain a one-element placeholder while `position` and `velocity` receive full-sized storage.

A metadata-only array still appears in `used`: retaining only `read | written` is insufficient for allocation planning. Requery when changing static configuration, selected handlers, argument types, container structure, or sharing. A later launch that enables the force branch needs a fully allocated `force` array.

## Using the result in a launch recorder

Resolve `written` paths against the actual launch arguments before launching. They identify buffers that may overwrite values needed by earlier launches. A recorder can snapshot those buffers and restore the appropriate versions while reversing the recorded sequence.

The result describes the forward kernel's accesses. It is not the backward kernel's precise saved-value requirement, does not distinguish reads before and after a write within one kernel, and does not make otherwise invalid in-place autodiff operations valid. A recorder that lacks a more specific backward dependency analysis should conservatively preserve earlier used buffers that a later launch may overwrite. The query itself neither records snapshots nor changes `qd.ad.Tape` behavior. See [Automatic differentiation](autodiff.md) for the differentiation rules.

## Supported inputs and errors

The query supports ndarray and Field arguments, `qd.Tensor` wrappers, Python dataclasses, data-oriented receivers and objects, and the supported [handler tuple capture types](handler_tuples.md). External NumPy and PyTorch arrays are supported wherever a normal kernel argument accepts them; handler tuple captures retain their existing restrictions.

All accessed arrays must be reachable through supported paths rooted at kernel arguments, including an implicitly bound `self`. A kernel that accesses a global or closure-captured array with no argument-rooted path raises `qd.QuadrantsSyntaxError`. Pass that array explicitly or store it on a supplied object to make it queryable. An operation whose memory effects cannot be analyzed, including an opaque external call, also raises `qd.QuadrantsSyntaxError`; the query never returns a partial result as though it were complete.

Argument binding and unsupported argument types produce the same errors as a normal kernel call. This API queries ordinary forward kernels; querying generated gradient kernels or graph kernels is unsupported and raises `qd.QuadrantsSyntaxError`.

## Caching

Results are cached in memory per specialization. Replacing an ndarray with another having the same specialization properties and preserving the alias structure reuses the analysis. Changing array contents or an ordinary runtime scalar value does not require another analysis. Changing a static flag or selected handler does.

For kernels eligible for [fastcache](fastcache.md), access metadata is also persisted per specialization. It can be read without loading or compiling a device-code artifact, including when the only previous operation was a query. Results for different static configurations are kept separate. A normal compilation of a queryable kernel populates the same access cache, so querying afterward reuses its analysis. Kernels outside the query's supported scope remain launchable under the normal kernel rules.

Handler tuple arguments use the in-memory access cache only, matching their existing offline-cache restriction. Cached results contain paths and access classifications, not references to the supplied arrays.
