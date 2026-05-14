# Shared Metal command queue (PyTorch MPS)

On Apple Silicon, Quadrants and PyTorch MPS both dispatch GPU work via Metal. By default each framework creates its own `MTLCommandQueue`, which means there is no GPU-level ordering between them. Every zero-copy interop point therefore requires explicit CPU-side synchronisation (`qd.sync()` and `torch.mps.synchronize()`) to guarantee data visibility.

The `external_metal_command_queue` option lets you pass PyTorch's command queue to Quadrants so that both frameworks share a single queue. Metal processes command buffers in commit order within a queue, so GPU-side ordering is automatic and the per-interop sync overhead is eliminated.

## Quick start

```python
import quadrants as qd
from quadrants.interop import get_mps_command_queue

queue_ptr = get_mps_command_queue()
qd.init(
    arch=qd.metal,
    external_metal_command_queue=queue_ptr,
    external_metal_command_queue_is_torch_queue=True,
)
```

Two flags work together:

- `external_metal_command_queue` — the raw `MTLCommandQueue*` pointer. Quadrants dispatches all GPU work on this queue instead of creating its own.
- `external_metal_command_queue_is_torch_queue` — set to `True` when the queue comes from PyTorch MPS. This tells Quadrants that PyTorch shares the same queue, so the explicit interop syncs can be safely skipped. Defaults to `False`, which preserves the sync calls even when an external queue is provided (useful when the external queue belongs to a non-PyTorch framework).

Once initialised with both flags:

- `to_torch(copy=False)` no longer calls `qd.sync()` internally.
- `to_torch(copy=True)` no longer calls `torch.mps.synchronize()` after the copy.
- GPU work submitted by Quadrants and by PyTorch executes in the order it was committed — no manual sync needed between the two.

You can still call `qd.sync()` when you need to read results back to the CPU (e.g. `to_numpy()`); what changes is that you no longer need *both* `qd.sync()` and `torch.mps.synchronize()` at every framework boundary.

## Extracting PyTorch's MTLCommandQueue

PyTorch does not expose its MPS command queue through a public Python API. Quadrants provides a built-in helper that extracts it at runtime using `ctypes` and the Objective-C runtime:

```python
from quadrants.interop import get_mps_command_queue

queue_ptr = get_mps_command_queue()  # returns int (raw pointer), or 0 on failure
```

The function initialises PyTorch MPS if needed, then returns the `MTLCommandQueue*` as a Python `int`. It returns `0` if extraction fails (e.g. non-macOS platform, PyTorch not installed, MPS not available, or unsupported PyTorch build). The underlying C++ symbol (`_ZN2at3mps19getDefaultMPSStreamEv`) has been stable since PyTorch 1.13.

## Init ordering

`get_mps_command_queue()` handles PyTorch MPS initialisation internally, so you can call it before `qd.init()` without any manual setup:

```python
import quadrants as qd
from quadrants.interop import get_mps_command_queue

queue_ptr = get_mps_command_queue()  # initialises MPS if needed
qd.init(
    arch=qd.metal,
    external_metal_command_queue=queue_ptr,
    external_metal_command_queue_is_torch_queue=True,
)
```

## What changes with a shared queue

| Scenario | Separate queues (default) | Shared queue |
|----------|--------------------------|--------------|
| `f.to_torch(copy=False)` | `qd.sync()` called internally | no sync needed |
| `f.to_torch(copy=True)` | `qd.sync()` + `torch.mps.synchronize()` | no sync needed |
| Quadrants kernel after torch write | manual `torch.mps.synchronize()` required | automatic (same queue) |
| `f.to_numpy()` | `qd.sync()` (always needed for CPU readback) | `qd.sync()` (still needed — the kernel-copy path is Quadrants-internal, not routed through MPS) |

## Lifetime and ownership

The caller (your application) owns the command queue. Quadrants borrows the pointer without retaining it, so the caller must keep the queue alive for the lifetime of the Quadrants runtime. In practice this means keeping PyTorch (and its MPS backend) alive for as long as `qd.init()` is active.

## Fallback

`get_mps_command_queue()` returns `0` on failure (non-macOS, missing PyTorch, unsupported build) rather than raising. You can use this to fall back to the default separate-queue path:

```python
from quadrants.interop import get_mps_command_queue

queue_ptr = get_mps_command_queue()
qd.init(
    arch=qd.metal,
    external_metal_command_queue=queue_ptr or None,
    external_metal_command_queue_is_torch_queue=queue_ptr != 0,
)
```

When `external_metal_command_queue` is `0` (or omitted), Quadrants creates its own queue and the explicit sync path is used as before.
