# Debug mode

Quadrants provides a debug mode that enables additional runtime checks inside kernels. These checks are disabled by default for performance.

## Enabling debug mode

```python
qd.init(arch=qd.gpu, debug=True)
```

## What debug mode enables

### Bounds checking

With `debug=True`, out-of-bounds accesses on fields raise a `RuntimeError` at runtime instead of silently reading/writing garbage memory:

```python
qd.init(arch=qd.gpu, debug=True)

x = qd.field(qd.i32, shape=(8, 16))

@qd.kernel
def func() -> None:
    x[3, 16] = 1  # RuntimeError: index 16 out of bounds for axis 1 with size 16

func()
```

Without debug mode, this would silently corrupt memory or produce incorrect results.

Bounds checking also works for Python-scope field access:

```python
x = qd.field(qd.f32, shape=3)
x[3] = 10.0   # AssertionError in debug mode, silent corruption otherwise
a = x[-1]     # AssertionError in debug mode
```

#### Adstack overflow

The adstack overflow check on reverse-mode autodiff runs always, on every backend, regardless of `debug`. A push past the per-stack capacity raises `QuadrantsAssertionError("[Aa]dstack overflow")` at the next Quadrants Python entry that polls the overflow flag after the offending kernel has executed - kernel launch, host-side field / ndarray read, or `qd.sync()`. On CPU this is the entry of the offending launch itself; on GPU it can be one or more entries later, since the GPU may not have run the offending push by the time the poll at end-of-launch fires. The error message describes the cause (untracked tensor mutation between launches, or sizer under-estimate caused by a bug in Quadrants) and the recovery flow; see [Autodiff -> What can go wrong](autodiff.md) for the full description.

### Assertions in kernels

The `assert` statement works inside kernels when debug mode is enabled:

```python
qd.init(arch=qd.gpu, debug=True)

@qd.kernel
def check(a: qd.Template) -> None:
    for i in range(10):
        assert a[i] >= 0, f"negative value at index {i}"
```

Assertions support constant strings and f-strings for the error message.

Note: `assert` is compiled into the kernel but only checked at runtime when `debug=True`. On Linux ARM64, assertions are not currently supported.

## Performance impact

Debug mode adds runtime checks to every field access and assertion, which can significantly slow down kernel execution. It is intended for development and debugging, not production use.

A typical workflow:
1. Develop with `debug=True` to catch bounds errors and logic bugs
2. Switch to `debug=False` (the default) for benchmarking and production runs

`debug=True` always implies `check_out_of_bound=True`. For bounds safety in a release build without the rest of debug mode, set [`check_out_of_bound=True`](./init_options.md#check_out_of_bound) instead.

## Other debugging tools

### Disabling the cache

If you encounter crashes or unexpected behavior, try disabling the offline cache to rule out stale compiled kernels:

```python
qd.init(arch=qd.gpu, offline_cache=False)
```

Or clear the cache entirely:

```bash
rm -Rf ~/.cache/quadrants
```

See also [Troubleshooting](./troubleshooting.md).

### Printing from kernels

`print()` works inside kernels for scalar values, which can be useful for debugging:

```python
@qd.kernel
def debug_kernel(a: qd.Template) -> None:
    for i in range(10):
        print("i =", i, "val =", a[i])
```

Per-backend support:

| Backend | Kernel `print()` |
|---------|------------------|
| CPU | yes |
| CUDA | yes |
| AMDGPU | no (silently dropped) |
| Metal | no (silently dropped) |
| Vulkan | yes (via debug-printf SPIR-V extension) |

**Note.** Output from GPU kernels appears in order despite parallel execution because all kernels are queued in the same compute stream.

**Important.** Avoid kernel `print()` calls in production code where you can. Quadrants synchronizes the compute queue after every dispatch of a kernel that contains a `print()` so the output appears as close as possible to the call site. The synchronization happens unconditionally on every launch of that kernel, even when the surrounding control flow leaves the `print()` unreached at runtime; the cost is the full per-launch sync overhead, not just the cost of the `print()` itself.

### Dumping compiled IR

To inspect the compiled intermediate representation, use the `QD_DUMP_IR` environment variable:

```bash
QD_DUMP_IR=1 QD_OFFLINE_CACHE=0 python my_script.py
```

Compiled kernels will be written to `/tmp/ir` by default. Use `QD_DEBUG_DUMP_PATH=` to redirect to a custom directory.

### Tracing adstack heap allocations

```bash
QD_DEBUG_ADSTACK=1 python my_script.py
```

Prints one line per task per kernel launch describing each adstack heap binding: task name, heap kind (float or int), sizing source (per-task reducer count or dispatched-threads worst case), per-thread stride, and resulting allocation in bytes. Useful for pinning which task drives the peak when an adstack-bearing kernel hits an OOM and the remedies in [Avoiding OOM on GPU](./autodiff.md#avoiding-oom-on-gpu) do not point at an obvious culprit.
