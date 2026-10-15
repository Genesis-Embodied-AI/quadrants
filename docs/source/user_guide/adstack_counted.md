# Counting adstack requirements during forward execution

This experimental prototype counts required stack depth during the real forward call, then uses that measurement to allocate storage during backward. Backward still replays forward calculations to record intermediate values before computing gradients. Counting does not copy the input tensors or save their historical contents.

```python
qd.init(arch=qd.cuda)

with qd.adstack:
    compute(x, y, iterations)

y.grad.fill(1)
compute.grad(x, y, iterations)
```

Preserve or restore all input values that backward needs to replay the original computation. If replay needs more stack space than measured, backward raises `RuntimeError`. Some gradients may already have changed: discard or reset the failed backward's results. The runtime does not roll back or retry automatically. Changes that fit the measured capacity can still produce wrong gradients without an overflow.

The prototype supports CUDA, x64, Vulkan, and Metal, range loops over ndarrays or dense fields, and dense field iteration. Kernel arguments use normal binding, including keyword arguments, supported external arrays (such as NumPy and PyTorch), and template fields. Sparse/quantized fields, random operations, cooperative/internal calls, while loops, continue statements, explicit returns, graph kernels, custom streams, and nested counting contexts are unsupported.

Each counted forward creates an invocation handle. Explicit `.grad(...)` calls consume pending invocations for that kernel in reverse order with matching arguments. `qd.ad.Tape` supplies that order automatically; put the counting context inside the Tape context.

For repeated backwards or a different invocation order, retain the scope and use its handles:

```python
with qd.adstack as scope:
    compute(x, y, iterations)
record = scope.records[0]
y.grad.fill(1)
record.backward()
x.grad.fill(0)
y.grad.fill(2)
record.backward()  # Reuses the capacity and bound arguments; seeds a different gradient.
record.release()  # Removes it from pending calls and drops its argument references.
```

Handle use removes that invocation from the automatic pending list, but the handle remains reusable until released. `.grad(...)` also leaves a retained handle reusable. Each context entry returns a distinct scope, with `records` and measured `capacities` in forward-call order. Release unused records as well as used ones when finished. A released handle or one from before `qd.reset()`/`qd.init()` cannot run backward.

Records retain arguments, not snapshots of their contents. Preserve or restore inputs before every replay, keep field SNode trees alive, and seed/reset gradients as needed. Gradients accumulate according to normal autodiff rules. After overflow, reset invalid gradients and restore replay inputs before retrying; the handle clears its error flag but does not increase capacity.

Persistent offline caching and `@qd.kernel(fastcache=True)` distinguish ordinary, counting-forward, and measured-backward variants. Each invocation retains its own measured capacity. Invocations of the same kernel share a small scratch buffer for counting and overflow reporting; synchronized readbacks finish before this buffer is reused. The buffer is cleared before each backward, including retries after overflow. CPU and CUDA read the two measurement words directly; Vulkan and Metal use a portable array readback. These runtime measurements are not part of the cached executable.

The initial layout uses the largest measured depth as the capacity for every stack. This can use more memory than separate per-stack capacities. Both forward measurement and backward error reporting synchronize with the device. On overflow, stack counts saturate and stack accesses remain within their allocation. Threads continue executing, including compiler-generated reductions and synchronization; backward then raises and its results must be discarded. Compiler thread-local reduction and block-local optimizations remain enabled. These are prototype tradeoffs to measure, not claims of negligible overhead.

On the counted-only branch, stack-based reverse mode requires a counted forward. The symbolic sizer, sparse stack allocation, mutation-tracking caches, and old overflow diagnostic registry are removed. The old `ad_stack_experimental_enabled`, `ad_stack_size`, and `ad_stack_sparse_threshold_bytes` initialization options are removed. Ordinary non-stack autodiff remains available outside the context. After using a kernel with counting, its explicit backwards use pending counted records; an empty record list is an error.

The SPIR-V path uses descriptor-bound measurement storage and measured heap metadata; it does not run the symbolic sizer. Its loop-carried stack values are limited to f32, i32, and booleans, as in the existing SPIR-V adstack implementation. Vulkan is validated on the Linux GPU cluster; Metal requires separate validation on Apple hardware.

Vector-valued stacks require `qd.init(force_scalarize_matrix=True)` so the counting transform receives scalar stacks.
