# CUDA and AMDGPU device operations

The C++ device interface supports the same upload, readback, mapping, device-to-device copy, import, and allocation
operations for CUDA and AMDGPU. AMDGPU now implements `upload_data` and `readback_data`, including allocation offsets,
using the same implementation as CUDA. Python APIs are unchanged.

Runtime-pool allocation clears its result on the active stream, launches the allocation kernel, waits for that stream,
and then checks the returned pointer. A null result reports exhaustion with the appropriate backend name.

The shared implementation preserves these existing backend policies:

| Policy | CUDA | AMDGPU |
| --- | --- | --- |
| Merge adjacent cached blocks on release | Yes | No |
| Free driver-pool allocations | Asynchronous free on the default stream | Synchronous free |
| Release direct allocations | Free the raw allocation | Return it to the cache |
| Host-access flags in `allocate_memory_runtime` | Existing behavior ignores these flags | Reject these flags |

Use `allocate_memory` with host-access flags when managed memory is needed. This path already requests managed memory
on both backends. The policies above are retained compatibility choices, not claims that the hardware requires them.
