# Building the CUDA graph conditional fatbin

The `graph_do_while` feature uses a tiny CUDA kernel that calls
`cudaGraphSetConditional` (a device-side function from NVIDIA's
`libcudadevrt.a`) to control CUDA graph conditional while nodes. These
conditional nodes require SM 9.0+ (Hopper or later); on older GPUs,
`graph_do_while` falls back to a host-side loop automatically.

There are three distinct phases:

1. **Fatbin generation** (rare, manual) - A developer runs
   `scripts/build_condition_kernel_fatbin.py`, which compiles the kernel and
   device-links it with `libcudadevrt.a` to resolve `cudaGraphSetConditional`.
   The output is a self-contained fatbin, committed to git as a C header.
   Requires `nvcc` and the CUDA toolkit.
2. **Quadrants build** (CI / developers) - The C header is `#include`d as a
   plain byte array. No CUDA toolkit needed.
3. **Runtime** (end users) - The fatbin is loaded via `cuModuleLoadData`.
   No CUDA toolkit needed.

This page documents phase 1: regenerating the pre-built fatbin.

## When to regenerate

You only need to regenerate the fatbin if:

- The condition kernel source (`quadrants/runtime/cuda/graph_do_while_cond.cu`)
  changes.
- You need to add support for a new SM architecture.

## Prerequisites

- CUDA toolkit with `nvcc` (13.0 or later, required for SM 110 support;
  earlier toolkits will fail with `Unsupported gpu architecture 'compute_110'`).
- The `nvcc` binary must be on your `PATH`, or set `CUDA_HOME`.

## Regenerating

Run the script from the repo root:

```bash
python scripts/build_condition_kernel_fatbin.py
```

This will:

1. Compile `quadrants/runtime/cuda/graph_do_while_cond.cu` with relocatable device
   code for each target SM architecture.
2. Device-link the result with `libcudadevrt.a` to resolve the
   `cudaGraphSetConditional` extern.
3. Write the fatbin as a C byte array to
   `quadrants/runtime/cuda/graph_do_while_cond_fatbin.h`.

After regenerating, commit the updated header. Quadrants must be rebuilt to
pick up the new fatbin.

## Adding a new SM architecture

Edit the `SM_VERSIONS` list in `scripts/build_condition_kernel_fatbin.py` to
add the new SM version number (e.g., `130`), then regenerate.

## Files

| File | Purpose |
|------|---------|
| `quadrants/runtime/cuda/graph_do_while_cond.cu` | CUDA C source for the condition kernel |
| `scripts/build_condition_kernel_fatbin.py` | Regeneration script |
| `quadrants/runtime/cuda/graph_do_while_cond_fatbin.h` | Generated C header (checked into git) |

## How it's used at runtime

`GraphManager::ensure_condition_kernel_loaded()` in
`quadrants/runtime/cuda/graph_manager.cpp` loads the fatbin via
`cuModuleLoadData`. If the fatbin does not contain SASS for the current GPU's
SM architecture, loading fails with a clear error pointing to this script.
