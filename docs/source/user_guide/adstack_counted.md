# Counting adstack requirements during forward execution

This experimental prototype counts required stack depth during the real forward call, then uses that measurement to allocate storage during backward. Backward still replays forward calculations to record intermediate values before computing gradients. Counting does not copy the input tensors or save their historical contents.

```python
qd.init(arch=qd.cuda, offline_cache=False)

with qd.adstack:
    compute(x, y, iterations)

y.grad.fill(1)
compute.grad(x, y, iterations)
```

Preserve or restore all input values that backward needs to replay the original computation. If replay needs more stack space than measured, backward raises `RuntimeError`. Some gradients may already have changed: discard or reset the failed backward's results. The runtime does not roll back or retry automatically. Changes that fit the measured capacity can still produce wrong gradients without an overflow.

The prototype supports CUDA and x64, ordinary ndarray range loops, and positional scalar or `qd.ndarray` arguments. It requires `offline_cache=False`. Field access, random operations, cooperative/internal calls, while loops, continue statements, explicit returns, graph kernels, streams, and nested counting contexts are unsupported.

Each counted forward produces a single-use record. Explicit gradient calls must consume the same kernel's records in reverse invocation order with matching arguments. `qd.ad.Tape` can supply that order automatically; put the counting context inside the Tape context. A consumed or failed record cannot be used for a second backward. Records do not detect changes to array contents.

The initial layout uses the largest measured depth as the capacity for every stack. This can use more memory than separate per-stack capacities. Both forward measurement and backward error reporting synchronize with the device. On overflow, stack counts saturate and stack accesses remain within their allocation. Threads continue executing, including compiler-generated reductions and synchronization; backward then raises and its results must be discarded. Compiler thread-local reduction and block-local optimizations remain enabled. These are prototype tradeoffs to measure, not claims of negligible overhead.

Existing kernels outside the counting context retain the existing sizing path. After using a kernel with counting, its explicit backwards use the pending counted records; an empty record list is an error rather than a silent fallback.
