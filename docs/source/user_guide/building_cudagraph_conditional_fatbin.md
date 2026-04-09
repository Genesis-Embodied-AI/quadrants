# Building the CUDA graph conditional fatbin

The `graph_do_while` feature uses a tiny CUDA kernel to control conditional
while nodes on SM 9.0+ GPUs. This kernel calls `cudaGraphSetConditional`, a
device-side function from NVIDIA's `libcudadevrt.a`. The regeneration script
device-links with `libcudadevrt.a` to resolve this call, producing a
self-contained fatbin that is checked into the repo as a C header. At runtime,
the fatbin is loaded directly — no CUDA toolkit is needed on the user's system
or during the Quadrants build.

This page documents how to regenerate the pre-built fatbin.

## When to regenerate

You only need to regenerate the fatbin if:

- The condition kernel source (`quadrants/runtime/cuda/graph_do_while_cond.cu`)
  changes.
- You need to add support for a new SM architecture.

## Prerequisites

- CUDA toolkit with `nvcc` (12.4 or later, for conditional node support).
- The `nvcc` binary must be on your `PATH`, or set `CUDA_HOME`.

## Regenerating

Run the script from the repo root:

```bash
./scripts/build_condition_kernel_fatbin.sh
```

This will:

1. Compile `quadrants/runtime/cuda/graph_do_while_cond.cu` with relocatable device
   code for each target SM architecture.
2. Device-link the result with `libcudadevrt.a` to resolve the
   `cudaGraphSetConditional` extern.
3. Write the fatbin as a C byte array to
   `quadrants/runtime/cuda/graph_do_while_cond_fatbin.h`.

After regenerating, rebuild Quadrants and commit the updated header.

## Adding a new SM architecture

Edit the `SM_TARGETS` array in `scripts/build_condition_kernel_fatbin.sh` to
add the new `-gencode` flag, then regenerate.

## Files

| File | Purpose |
|------|---------|
| `quadrants/runtime/cuda/graph_do_while_cond.cu` | CUDA C source for the condition kernel |
| `scripts/build_condition_kernel_fatbin.sh` | Regeneration script |
| `quadrants/runtime/cuda/graph_do_while_cond_fatbin.h` | Generated C header (checked into git) |

## How it's used at runtime

`GraphManager::ensure_condition_kernel_loaded()` in
`quadrants/runtime/cuda/graph_manager.cpp` loads the fatbin via
`cuModuleLoadData`. If the fatbin does not contain SASS for the current GPU's
SM architecture, loading fails with a clear error pointing to this script.
