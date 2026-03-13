# CUDA Graph

CUDA graphs reduce kernel launch overhead by capturing a sequence of GPU operations into a graph, then replaying it in a single launch. This is most beneficial for kernels that compile into multiple GPU tasks (e.g. kernels with multiple top-level `for` loops), where the per-task launch overhead would otherwise dominate.

## Usage

Add `cuda_graph=True` to the `@qd.kernel` decorator:

```python
@qd.kernel(cuda_graph=True)
def my_kernel(
    x: qd.types.ndarray(qd.f32, ndim=1),
    y: qd.types.ndarray(qd.f32, ndim=1),
):
    for i in range(x.shape[0]):
        x[i] = x[i] + 1.0
    for i in range(y.shape[0]):
        y[i] = y[i] + 2.0
```

The kernel is used normally — no other API changes are needed:

```python
x = qd.ndarray(qd.f32, shape=(1024,))
y = qd.ndarray(qd.f32, shape=(1024,))

my_kernel(x, y)  # first call: builds and caches the graph
my_kernel(x, y)  # subsequent calls: replays the cached graph
```

### When to use

Use `cuda_graph=True` on kernels that:

- Run on CUDA (`arch=qd.cuda`)
- Contain **two or more** top-level `for` loops (i.e. compile into multiple offloaded tasks)
- Are called repeatedly with arguments of the same shape

Kernels with a single `for` loop compile into a single GPU task, so there is no multi-launch overhead to eliminate. The graph path will gracefully fall back to the normal launch path in this case.

### Restrictions

- **No struct return values.** Kernels that return values (e.g. `-> qd.i32`) cannot use CUDA graphs. An error is raised if `cuda_graph=True` is set on such a kernel.
- **Primal kernels only.** The `cuda_graph=True` flag is applied to the primal (forward) kernel only, not its adjoint. Autodiff kernels use the normal launch path.
- **Non-CUDA backends.** On non-CUDA backends (CPU, Vulkan, Metal), `cuda_graph=True` is silently ignored. This means you can annotate a kernel unconditionally and it will work on all platforms.

### Passing different arguments

You can pass different ndarrays to the same kernel on subsequent calls. The cached graph is replayed with the updated arguments — no graph rebuild occurs:

```python
x1 = qd.ndarray(qd.f32, shape=(1024,))
y1 = qd.ndarray(qd.f32, shape=(1024,))
my_kernel(x1, y1)  # builds graph

x2 = qd.ndarray(qd.f32, shape=(1024,))
y2 = qd.ndarray(qd.f32, shape=(1024,))
my_kernel(x2, y2)  # replays graph with new array pointers
```

### Fields as arguments

Fields (SNode-backed data created with `qd.field`) are accessed through the global runtime pointer, not through the kernel argument buffer. The graph captures this pointer at build time, so fields work transparently with CUDA graphs.

When different fields are passed as template arguments, each unique combination of fields produces a separately compiled kernel with its own graph cache entry. There is no interference between them.

---

## Advanced: Implementation Details

### Graph build and replay

On the first call to a `cuda_graph=True` kernel, the runtime:

1. **Allocates persistent device buffers** for the kernel's argument buffer and result buffer. These live for the lifetime of the runtime (until `qd.reset()`).
2. **Copies the host argument buffer** (containing scalar values and resolved device pointers for ndarrays) into the persistent device argument buffer.
3. **Builds a `RuntimeContext`** whose `arg_buffer` and `result_buffer` point at the persistent device buffers, and whose `runtime` pointer points at the `LLVMRuntime`. This `RuntimeContext` is stored inside the cache entry at a stable address.
4. **Constructs a CUDA graph** by iterating over the kernel's offloaded tasks and adding each as a kernel node. Each node receives a pointer to the persistent `RuntimeContext` as its sole kernel parameter. Nodes are chained with sequential dependencies.
5. **Instantiates** the graph into an executable (`cuGraphInstantiate`) and launches it.
6. **Caches** the graph executable, persistent buffers, and `RuntimeContext` in a map keyed by `launch_id`.

On subsequent calls (cache hit), the runtime:

1. **Copies the updated host argument buffer** into the persistent device argument buffer via `cuMemcpyHtoD`. This is the only operation needed — the graph's kernel nodes already point at the persistent `RuntimeContext`, which already points at the persistent argument buffer.
2. **Replays** the cached graph via `cuGraphLaunch`.

### How arguments reach the GPU kernels

Each compiled GPU kernel takes a single parameter: a pointer to `RuntimeContext`. The `RuntimeContext` contains:

- `arg_buffer`: a device-side buffer holding serialized scalar arguments and resolved ndarray device pointers
- `result_buffer`: a device-side buffer for return values
- `runtime`: a pointer to `LLVMRuntime`, which holds field/SNode tree data

For CUDA graphs, these pointers are baked into the graph at capture time. On replay, the *contents* of the argument buffer are updated (via a host-to-device memcpy), but the *pointers* themselves remain stable. This is what allows the graph to be replayed without rebuilding.

Before the argument buffer is copied to the device, `resolve_ctx_ndarray_ptrs` walks all array parameters and resolves `DeviceAllocation` handles into raw device pointers, writing them into the argument buffer. This ensures that even when different ndarrays are passed on subsequent calls, the argument buffer contains the correct device addresses.

### Cache keying and template specialization

The graph cache is keyed by `launch_id`, an integer assigned by `register_llvm_kernel` when a `CompiledKernelData` is first seen. Each unique combination of template arguments (including field arguments) produces a different compiled kernel with a different `launch_id`. This means:

- Calling the same kernel with field A and field B results in two independent compiled kernels, two independent `launch_id` values, and two independent graph cache entries.
- Each cached graph contains kernel nodes compiled specifically for that field combination's SNode layout.
- There is no risk of one template specialization's graph being replayed for a different specialization.

### Lifetime and cleanup

The `CachedCudaGraph` struct owns the graph executable and persistent device buffers via RAII. When the `KernelLauncher` is destroyed (which happens on `qd.reset()`), all cached graphs and their device allocations are freed. After a reset, the next kernel call triggers a fresh graph build against the new runtime.
