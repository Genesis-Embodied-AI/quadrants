# Streams

Streams allow concurrent execution of GPU operations. By default, all Quadrants kernels launch on the default stream, which serializes everything. By creating explicit streams, you can run independent kernels concurrently and control synchronization with events.

## Supported platforms

| Backend | Streams | Events | Notes |
|---------|---------|--------|-------|
| CUDA    | Yes     | Yes    | Full concurrent execution |
| CPU     | No-op   | No-op  | `qd_stream` is silently ignored, kernels run serially |
| Metal   | No-op   | No-op  | `qd_stream` is silently ignored, kernels run serially |
| Vulkan  | No-op   | No-op  | `qd_stream` is silently ignored, kernels run serially |

On backends without native stream support, `create_stream()` and `create_event()` return objects with handle `0`. All stream/event operations become no-ops and kernels run serially. Code written with streams is portable across all backends in the sense that it will run without modifications, but serially.

## Creating and using streams

```python
import quadrants as qd

qd.init(arch=qd.cuda)

N = 1024
a = qd.field(qd.f32, shape=(N,))
b = qd.field(qd.f32, shape=(N,))

@qd.kernel
def fill_a():
    for i in range(N):
        a[i] = 1.0

@qd.kernel
def fill_b():
    for i in range(N):
        b[i] = 2.0

s1 = qd.create_stream()
s2 = qd.create_stream()

fill_a(qd_stream=s1)
fill_b(qd_stream=s2)

s1.synchronize()
s2.synchronize()

s1.destroy()
s2.destroy()
```

Pass `qd_stream=` to any kernel call to launch it on that stream. Kernels on different streams may execute concurrently. Call `synchronize()` to block until all work on a stream completes.

## Events

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

## Context managers

Streams and events support `with` blocks for automatic cleanup:

```python
with qd.create_stream() as s:
    fill_a(qd_stream=s)
    s.synchronize()
# s.destroy() called automatically
```

## PyTorch interop (CUDA)

When mixing Quadrants kernels with PyTorch operations on CUDA, both frameworks must use the same stream to avoid race conditions. Without explicit stream management, Quadrants and PyTorch may launch work on different streams with no ordering guarantees, leading to intermittent data corruption.

### Running Quadrants kernels on PyTorch's stream

```python
import torch
from quadrants.lang.stream import Stream

torch_stream_ptr = torch.cuda.current_stream().cuda_stream
stream = Stream(torch_stream_ptr)

physics_kernel(qd_stream=stream)
observations = compute_obs_tensor()  # PyTorch op on the same stream
apply_actions_kernel(qd_stream=stream)
```

Wrap PyTorch's raw `CUstream` pointer in a Quadrants `Stream` object. Do **not** call `destroy()` on this wrapper — PyTorch owns the underlying stream.

### Running PyTorch operations on a Quadrants stream

```python
qd_stream = qd.create_stream()
torch_stream = torch.cuda.ExternalStream(qd_stream.handle)

with torch.cuda.stream(torch_stream):
    physics_kernel(qd_stream=qd_stream)
    observations = compute_obs_tensor()
    apply_actions_kernel(qd_stream=qd_stream)

qd_stream.destroy()
```

`Stream.handle` is the raw `CUstream` pointer, which `torch.cuda.ExternalStream` accepts directly.

## Limitations

- **Not compatible with graphs.** Do not pass `qd_stream` to a kernel decorated with `graph=True`.
- **Not compatible with autodiff.** Do not pass `qd_stream` to a kernel that uses reverse-mode or forward-mode differentiation, or inside a `qd.ad.Tape` context.
- **`qd.sync()` only waits on the default stream.** It does not drain explicit streams. Call `stream.synchronize()` on each stream you need to wait for.
- **No automatic synchronization.** You are responsible for inserting events or `synchronize()` calls when one stream's output is another stream's input.
