# Shared Metal command queue (PyTorch MPS)

On Apple Silicon, Quadrants and PyTorch MPS both dispatch GPU work via Metal. By default each framework creates its own `MTLCommandQueue`, which means there is no GPU-level ordering between them. Every zero-copy interop point therefore requires explicit CPU-side synchronisation (`qd.sync()` and `torch.mps.synchronize()`) to guarantee data visibility.

The `external_metal_command_queue` option lets you pass PyTorch's command queue to Quadrants so that both frameworks share a single queue. Metal processes command buffers in commit order within a queue, so GPU-side ordering is automatic and the per-interop sync overhead is eliminated.

## Quick start

```python
import quadrants as qd

queue_ptr = get_mps_command_queue()   # see below
qd.init(arch=qd.metal, external_metal_command_queue=queue_ptr)
```

Once initialised this way:

- `to_torch(copy=False)` no longer calls `qd.sync()` internally.
- `to_torch(copy=True)` no longer calls `torch.mps.synchronize()` after the copy.
- GPU work submitted by Quadrants and by PyTorch executes in the order it was committed — no manual sync needed between the two.

You can still call `qd.sync()` when you need to read results back to the CPU (e.g. `to_numpy()`); what changes is that you no longer need *both* `qd.sync()` and `torch.mps.synchronize()` at every framework boundary.

## Extracting PyTorch's MTLCommandQueue

PyTorch does not expose its MPS command queue through a public Python API. The following helper extracts it at runtime using `ctypes` and the Objective-C runtime, with no build-time PyTorch dependency:

```python
import ctypes
import os
import torch


def get_mps_command_queue() -> int:
    """Return PyTorch MPS's MTLCommandQueue* as a Python int."""
    # Ensure MPS is initialised
    torch.zeros(1, device="mps")

    torch_lib = os.path.join(
        os.path.dirname(torch.__file__), "lib", "libtorch_cpu.dylib"
    )
    handle = ctypes.CDLL(torch_lib)._handle

    libdl = ctypes.CDLL(None)
    dlsym = libdl.dlsym
    dlsym.restype = ctypes.c_void_p
    dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    # at::mps::getDefaultMPSStream() -> MPSStream*
    func_addr = dlsym(handle, b"_ZN2at3mps19getDefaultMPSStreamEv")
    assert func_addr, "Cannot find getDefaultMPSStream — check PyTorch version"
    stream_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p)(func_addr)()

    # MPSStream::commandBuffer() -> id<MTLCommandBuffer>
    cb_addr = dlsym(handle, b"_ZN2at3mps9MPSStream13commandBufferEv")
    assert cb_addr, "Cannot find MPSStream::commandBuffer"
    cb_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(cb_addr)(
        stream_ptr
    )

    # [commandBuffer commandQueue] via ObjC runtime
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    sel_reg = objc.sel_registerName
    sel_reg.restype = ctypes.c_void_p
    sel_reg.argtypes = [ctypes.c_char_p]
    msg_send = objc.objc_msgSend
    msg_send.restype = ctypes.c_void_p
    msg_send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    queue_ptr = msg_send(cb_ptr, sel_reg(b"commandQueue"))
    assert queue_ptr, "Failed to extract MTLCommandQueue"
    return queue_ptr
```

The C++ symbol `_ZN2at3mps19getDefaultMPSStreamEv` has been stable since PyTorch 1.13.

## Init ordering

PyTorch MPS must be initialised **before** `qd.init()` so that the command queue exists when Quadrants starts:

```python
import torch
torch.zeros(1, device="mps")        # trigger MPS init

import quadrants as qd
queue_ptr = get_mps_command_queue()
qd.init(arch=qd.metal, external_metal_command_queue=queue_ptr)
```

## What changes with a shared queue

| Scenario | Separate queues (default) | Shared queue |
|----------|--------------------------|--------------|
| `f.to_torch(copy=False)` | `qd.sync()` called internally | no sync needed |
| `f.to_torch(copy=True)` | `qd.sync()` + `torch.mps.synchronize()` | no sync needed |
| Quadrants kernel after torch write | manual `torch.mps.synchronize()` required | automatic (same queue) |
| `f.to_numpy()` | `qd.sync()` (always needed for CPU readback) | `qd.sync()` (still needed) |

## Lifetime and ownership

The caller (your application) owns the command queue. Quadrants retains it for the duration of the runtime and does **not** release it on `qd.reset()`. You must keep PyTorch (and its MPS backend) alive for as long as the Quadrants runtime is active.

## Fallback

If extracting the queue fails (e.g. on an older PyTorch version or a non-Apple system), fall back to the default separate-queue path:

```python
try:
    queue_ptr = get_mps_command_queue()
except (AssertionError, OSError):
    queue_ptr = 0   # 0 means "create a new queue" (the default)

qd.init(arch=qd.metal, external_metal_command_queue=queue_ptr)
```
