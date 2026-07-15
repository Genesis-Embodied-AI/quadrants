# Streams

Streams allow concurrent execution of GPU operations. By default, all Quadrants kernels launch on the default stream, which serializes everything. With streams, you can run multiple top-level for loops in parallel.

## Supported platforms

| Backend | Supported |
|---------|-----------|
| CUDA    | Yes       |
| AMDGPU  | Yes       |
| CPU     | No-op     |
| Metal   | No-op     |
| Vulkan  | No-op     |

On backends without native stream support, stream operations are no-ops and for loops run serially. Code using streams is portable across all backends — it will run without modifications, but serially.

## Stream parallelism

Inside a `@qd.kernel`, each `with qd.stream_parallel():` block runs on its own GPU stream.

```python
import quadrants as qd

qd.init(arch=qd.cuda)

N = 1024
a = qd.field(qd.f32, shape=(N,))
b = qd.field(qd.f32, shape=(N,))
c = qd.field(qd.f32, shape=(N,))

@qd.kernel
def compute_ab():
    with qd.stream_parallel():
        for i in range(N):
            a[i] = compute_a(i)
    with qd.stream_parallel():
        for j in range(N):
            b[j] = compute_b(j)

@qd.kernel
def combine():
    for i in range(N):
        c[i] = a[i] + b[i]

compute_ab()  # the two stream_parallel blocks run concurrently
combine()     # runs after compute_ab() returns — a[] and b[] are ready
```

Consecutive `with qd.stream_parallel():` blocks run concurrently. Multiple for loops within a single block share a stream and run serially on it. All streams are synchronized before the kernel returns.

> **For `graph=True` kernels**, use [`qd.graph_parallel_context` / `qd.graph_parallel`](graph.md#qdgraph_parallel-sections-with-qdgraph_parallel_context-experimental) instead - `stream_parallel` is not compatible with graphs (see [Limitations](#limitations)). `qd.graph_parallel_context` expresses the same "run these independent sequences concurrently" idea but is honored by the graph builder.

### Restrictions

- All top-level statements in a kernel must be either all `stream_parallel` blocks or all regular statements. Mixing the two at the top level is a compile-time error.
- Nesting `stream_parallel` blocks is not supported.

## Explicit streams

For cases that require manual control — such as launching separate kernels on different streams or interoperating with PyTorch — you can create and manage streams directly.

### Creating and using streams

Any `@qd.kernel` function accepts a special `qd_stream` keyword argument — you do not need to declare it in the kernel signature. The `@qd.kernel` decorator handles it automatically.

```python
@qd.kernel
def my_kernel():
    for i in range(N):
        a[i] = i

s1 = qd.create_stream()
s2 = qd.create_stream()

my_kernel(qd_stream=s1)
my_kernel(qd_stream=s2)

s1.synchronize()
s2.synchronize()

s1.destroy()
s2.destroy()
```

Kernels on different streams may execute concurrently. Call `synchronize()` to block until all work on a stream completes.

### Events

Events let you express dependencies between streams without full synchronization.

```python
s1 = qd.create_stream()
s2 = qd.create_stream()

@qd.kernel
def produce():
    for i in range(N):
        a[i] = 10.0

@qd.kernel
def consume():
    for i in range(N):
        b[i] = a[i]

produce(qd_stream=s1)

e = qd.create_event()
e.record(s1)       # record when s1 finishes produce()
e.wait(qd_stream=s2)  # s2 waits for that event before proceeding

consume(qd_stream=s2)  # safe to read a[] — produce() is guaranteed complete
s2.synchronize()

e.destroy()
s1.destroy()
s2.destroy()
```

`e.record(stream)` captures the point in `stream`'s execution. `e.wait(qd_stream=stream)` makes `stream` wait until the recorded point is reached. If `qd_stream` is omitted, the default stream waits.

### Context managers

Streams and events support `with` blocks for automatic cleanup:

```python
with qd.create_stream() as s:
    some_func1(qd_stream=s)
# s.destroy() called automatically — waits for in-flight work
```

## Synchronization notes

- **`qd.sync()` only waits on the default stream.** It does not drain explicit streams. Call `stream.synchronize()` on each stream you need to wait for.
- **No automatic synchronization with explicit streams.** When using explicit streams, you are responsible for inserting events or `synchronize()` calls when one stream's output is another stream's input. `stream_parallel` handles this automatically.

## Limitations

- **Not compatible with [graphs](graph.md).** Streams cannot be combined with [`graph=True`](graph.md) kernels: passing `qd_stream` to a graph kernel raises a `RuntimeError`, and using `qd.stream_parallel()` inside a graph kernel raises a `QuadrantsSyntaxError`.
- **Not compatible with [autodiff](autodiff.md).** Do not pass `qd_stream` to a kernel that uses reverse-mode or forward-mode differentiation, or inside a [`qd.ad.Tape`](autodiff.md) context (if you do, a `RuntimeError` will be raised).
