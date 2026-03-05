# CUDA Graphs

When a Quadrants kernel has multiple top-level `for` loops, each loop is launched as a separate GPU kernel. The per-kernel launch overhead can become significant when kernels are small and numerous. CUDA graphs let you capture these launches once and replay them as a single unit, eliminating the repeated launch overhead.

## Per-kernel opt-in with `cuda_graph=True`

Annotate a kernel with `cuda_graph=True` to enable graph capture:

```python
@qd.kernel(cuda_graph=True)
def step(x: qd.types.ndarray(qd.f32, ndim=1),
         y: qd.types.ndarray(qd.f32, ndim=1)):
    for i in range(x.shape[0]):
        x[i] = x[i] + 1.0
    for i in range(y.shape[0]):
        y[i] = y[i] + 2.0

step(x, y)  # first call: captures the graph
step(x, y)  # subsequent calls: replays the cached graph
```

On the first call, the kernel's offloaded tasks are captured into a CUDA graph using the explicit node API. Subsequent calls replay the cached graph. The arg buffer is re-uploaded on each replay, so calling the kernel with different ndarrays works correctly.

**When it applies**: graph capture only activates when there are 2 or more top-level `for` loops (offloaded tasks). A single-loop kernel with `cuda_graph=True` falls back silently to the normal launch path.

**Cross-platform**: `cuda_graph=True` is a harmless no-op on non-CUDA backends (CPU, Metal, etc.). You can annotate kernels unconditionally without breaking portability.

## GPU-side iteration with `graph_while`

For iterative algorithms (physics solvers, convergence loops), you often want to repeat the kernel body until a condition is met, without returning to the host each iteration. The `graph_while` parameter enables this:

```python
@qd.kernel(graph_while="counter")
def solve(x: qd.types.ndarray(qd.f32, ndim=1),
          counter: qd.types.ndarray(qd.i32, ndim=0)):
    for i in range(x.shape[0]):
        x[i] = x[i] + 1.0
    for i in range(1):
        counter[None] = counter[None] - 1

x = qd.ndarray(qd.f32, shape=(N,))
counter = qd.ndarray(qd.i32, shape=())
counter.from_numpy(np.array(10, dtype=np.int32))
solve(x, counter)
# x is now incremented 10 times; counter is 0
```

The `graph_while` value is the name of a scalar `qd.i32` ndarray parameter. The kernel body repeats while this value is non-zero.

- On SM 9.0+ (Hopper), this uses CUDA conditional while nodes — the entire iteration runs on the GPU with no host involvement.
- On older CUDA GPUs and non-CUDA backends, it falls back to a host-side do-while loop.
- `graph_while` implicitly enables `cuda_graph=True`.

### Patterns

**Counter-based**: set the counter to N, decrement each iteration. The body runs exactly N times.

```python
@qd.kernel(graph_while="counter")
def iterate(x: qd.types.ndarray(qd.f32, ndim=1),
            counter: qd.types.ndarray(qd.i32, ndim=0)):
    for i in range(x.shape[0]):
        x[i] = x[i] + 1.0
    for i in range(1):
        counter[None] = counter[None] - 1
```

**Boolean flag**: set a `keep_going` flag to 1, have the kernel set it to 0 when a convergence criterion is met.

```python
@qd.kernel(graph_while="keep_going")
def converge(x: qd.types.ndarray(qd.f32, ndim=1),
             keep_going: qd.types.ndarray(qd.i32, ndim=0)):
    for i in range(x.shape[0]):
        # ... do work ...
        pass
    for i in range(1):
        if some_condition(x):
            keep_going[None] = 0
```

### Do-while semantics

`graph_while` has **do-while** semantics: the kernel body always executes at least once before the condition is checked. This matches the behavior of CUDA conditional while nodes. The flag value must be >= 1 at launch time. Passing 0 with a kernel that decrements the counter will cause an infinite loop.

## When to use CUDA graphs

CUDA graphs are most beneficial when:

- A kernel has many small top-level `for` loops where launch overhead dominates runtime.
- An iterative algorithm needs to repeat the kernel body many times without host round-trips (`graph_while`).

They are less useful when:

- Kernels have only a single top-level loop (no graph is created).
- Individual kernel runtimes are large enough to fully hide launch latency.
