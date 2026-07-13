# Performance Dispatch

`@qd.perf_dispatch` is a decorator that automatically selects the fastest implementation of a function at runtime. When you have multiple implementations of the same operation (e.g. different algorithmic strategies, or platform-specific kernels), `perf_dispatch` benchmarks them and picks the winner.

## Basic usage

Define a meta-function with `@qd.perf_dispatch`. The function body should be empty — it serves only as a prototype declaring the signature and geometry hash:

```python
@qd.perf_dispatch(get_geometry_hash=lambda a, b: hash(a.shape + b.shape))
def my_op(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]): ...
```

Then register concrete implementations using `@my_op.register`. Each implementation must have the same parameter names as the prototype.

```python
@my_op.register
@qd.kernel
def my_op_v1(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(a.shape[0]):
        b[i] = a[i] * 2

@my_op.register
@qd.kernel
def my_op_v2(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> None:
    i = 0
    # will run the same speed as the other version, but just an example
    while i < a.shape[0]:
        b[i] = a[i] << 1
```

Call the meta-function as though calling a standard Quadrants kernel:

```python
my_op(a, b)
```

On the first several calls, `perf_dispatch` will cycle through implementations (warming up, then timing). Once all implementations have been timed, the fastest is cached and used until re-evaluation is triggered (see [Tuning parameters](#tuning-parameters)).

## Decorator order

When registering a `@qd.kernel`, the `@my_op.register` decorator must be the **topmost** decorator:

```python
# Correct
@my_op.register
@qd.kernel
def impl(...) -> None: ...

# Wrong — will raise QuadrantsSyntaxError
@qd.kernel
@my_op.register
def impl(...) -> None: ...
```

## Registering plain Python functions

Implementations do not have to be `@qd.kernel`. Plain Python functions work too. You can mix kernel and Python implementations under the same meta-function:

```python
@my_op.register
def my_op_python(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(a.shape[0]):
        b[i] = a[i] * 2
```

## Return values

Implementations are not limited to returning `None`. The return value of whichever implementation runs is passed through to the caller:

```python
@qd.perf_dispatch(get_geometry_hash=lambda a, b: hash(a.shape + b.shape))
def my_op(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> int: ...

@my_op.register
def my_op_python(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> int:
    for i in range(a.shape[0]):
        b[i] = a[i] * 2
    return a.shape[0]

count = my_op(a, b)  # receives the return value from whichever impl runs
```

Note that return types need to match across implementations, and the meta function.

## Compatibility filtering

Some implementations may only work under certain conditions (specific platforms, input shapes, etc.). Use the `is_compatible` parameter to declare when an implementation is eligible:

```python
@my_op.register(is_compatible=lambda a, b: a.shape[0] >= 1024)
@qd.kernel
def my_op_large(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]) -> None:
    # optimized for large inputs
    ...
```

`is_compatible` receives the same arguments as the meta-function and must return `True` or `False`. If an implementation cannot handle certain inputs, `is_compatible` **must** be provided and must return `False` for those inputs. Implementations without `is_compatible` are assumed to always be compatible.

If only one implementation is compatible for a given call, it is used immediately without benchmarking.

## Geometry hash

The `get_geometry_hash` function maps call arguments to a hash representing the "geometry" of the input. Different geometries are benchmarked independently, so `perf_dispatch` can select different winners for different input shapes or configurations.

```python
# Different shapes benchmark independently
@qd.perf_dispatch(get_geometry_hash=lambda a, b: hash(a.shape + b.shape))
def my_op(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]): ...
```

Guidelines for `get_geometry_hash`:

- Return a constant (e.g. `0`) if all inputs have the same performance characteristics — a single winner will be chosen.
- Hash input shapes when different shapes may favor different implementations.
- **Avoid reading GPU data** in the hash function, as this creates a GPU sync point and will severely degrade performance. Prefer metadata like `.shape` which is available on the CPU.

## Tuning parameters

`@qd.perf_dispatch` accepts several optional parameters to control the benchmarking process:

| Parameter | Default | Description |
|---|---|---|
| `warmup` | 3 | Number of untimed warmup calls per implementation before measuring. |
| `active` | 1 | Number of timed calls per implementation. |
| `repeat_after_count` | 0 | Re-run benchmarking after this many additional calls. 0 disables. |
| `repeat_after_seconds` | 1.0 | Re-run benchmarking after this many seconds have elapsed. 0 disables. |

Example with custom tuning:

```python
@qd.perf_dispatch(
    get_geometry_hash=lambda a, b: hash(a.shape),
    warmup=5,
    active=2,
    repeat_after_seconds=60.0,
)
def my_op(a: qd.types.NDArray[qd.f32, 1], b: qd.types.NDArray[qd.f32, 1]): ...
```

## How benchmarking works

1. **Warmup phase**: Each compatible implementation is called `warmup` times in round-robin order. These calls are not timed.
2. **Active phase**: Each compatible implementation is called `active` times in round-robin order. The GPU is synchronized before and after each call to get accurate wall-clock measurements.
3. **Selection**: The implementation with the lowest active-phase time is cached as the winner for that geometry hash.
4. **Steady state**: Subsequent calls with the same geometry go directly to the cached winner with no overhead.
5. **Re-evaluation** (optional): After `repeat_after_count` calls or `repeat_after_seconds` seconds, the entire warmup + active cycle restarts from scratch, allowing the dispatcher to adapt if conditions change.

## Forcing a specific implementation

For debugging or profiling, you can bypass the auto-tuning and force a specific implementation using the `QD_PERFDISPATCH_FORCE` environment variable:

```bash
QD_PERFDISPATCH_FORCE=my_op:my_op_v2 python my_script.py
```

The format is `dispatcher_name:implementation_name`, where `dispatcher_name` is the name of the meta-function and `implementation_name` is the name of the registered function.

To force implementations for multiple dispatchers, separate entries with commas:

```bash
QD_PERFDISPATCH_FORCE=my_op:my_op_v2,transform:transform_v1 python my_script.py
```

Dispatchers not listed in the env var will benchmark normally.

### Discovering available names

When `QD_PERFDISPATCH_FORCE` is set, all dispatchers automatically print their name and registered implementations at startup. To discover valid values, set the env var to any value and run the program:

```bash
QD_PERFDISPATCH_FORCE=? python my_script.py
```

This will produce output like:

```
perf_dispatch 'my_op': registered 'my_op_v1'
perf_dispatch 'my_op': registered 'my_op_v2'
perf_dispatch 'my_op': available implementations: ['my_op_v1', 'my_op_v2']
perf_dispatch 'transform': registered 'transform_v1'
perf_dispatch 'transform': registered 'transform_v2'
perf_dispatch 'transform': available implementations: ['transform_v1', 'transform_v2']
```

If the requested implementation name doesn't match any registered function, a warning is printed and the dispatcher falls back to normal benchmarking.

## Important notes

- All registered implementations **must produce identical results**, including side effects. `perf_dispatch` does not verify this — incorrect results will be silently returned if implementations disagree.
- Only one implementation runs per call. Implementations do not need to be idempotent.
- `perf_dispatch` is **not thread-safe**. Do not call the same meta-function concurrently from multiple threads.
- Set `QD_PERFDISPATCH_PRINT_DEBUG=1` to print debug messages showing which implementation was registered and which was selected.

## Complete example

```python
import quadrants as qd

@qd.perf_dispatch(
    get_geometry_hash=lambda data, out: hash(data.shape),
    repeat_after_seconds=0,
)
def transform(data: qd.types.NDArray[qd.f32, 1], out: qd.types.NDArray[qd.f32, 1]): ...

@transform.register
@qd.kernel
def transform_v1(data: qd.types.NDArray[qd.f32, 1], out: qd.types.NDArray[qd.f32, 1]) -> None:
    for i in range(data.shape[0]):
        out[i] = data[i] * 2

@transform.register
@qd.kernel
def transform_v2(data: qd.types.NDArray[qd.f32, 1], out: qd.types.NDArray[qd.f32, 1]) -> None:
    i = 0
    # will run the same speed as the other version, but just an example
    while i < data.shape[0]:
        out[i] = data[i] << 1

data = qd.ndarray(qd.f32, (1024,))
out = qd.ndarray(qd.f32, (1024,))

for _ in range(100):
    transform(data, out)
```
