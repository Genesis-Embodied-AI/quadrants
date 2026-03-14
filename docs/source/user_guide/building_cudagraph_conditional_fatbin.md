# Building the CUDA graph conditional fatbin

The `graph_do_while` feature uses a tiny CUDA kernel to control conditional
while nodes on SM 9.0+ GPUs. This kernel calls `cudaGraphSetConditional`, a
device-side function from NVIDIA's `libcudadevrt.a`.

At runtime, Quadrants can load this kernel in two ways:

1. **Pre-built fatbin** (preferred) — a self-contained binary checked into the
   repo, with no runtime dependencies beyond the CUDA driver.
2. **JIT fallback** — links the kernel's PTX with `libcudadevrt.a` at runtime.
   Requires the CUDA toolkit to be installed on the user's system.

This page documents how to regenerate the pre-built fatbin.

## When to regenerate

You only need to regenerate the fatbin if:

- The condition kernel source (`quadrants/runtime/cuda/condition_kernel.cu`)
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

1. Compile `quadrants/runtime/cuda/condition_kernel.cu` with relocatable device
   code for each target SM architecture.
2. Device-link the result with `libcudadevrt.a` to resolve the
   `cudaGraphSetConditional` extern.
3. Write the fatbin as a C byte array to
   `quadrants/runtime/cuda/condition_kernel_fatbin.h`.

After regenerating, rebuild Quadrants and commit the updated header.

## Adding a new SM architecture

Edit the `SM_TARGETS` array in `scripts/build_condition_kernel_fatbin.sh` to
add the new `-gencode` flag, then regenerate.

## Files

| File | Purpose |
|------|---------|
| `quadrants/runtime/cuda/condition_kernel.cu` | CUDA C source for the condition kernel |
| `scripts/build_condition_kernel_fatbin.sh` | Regeneration script |
| `quadrants/runtime/cuda/condition_kernel_fatbin.h` | Generated C header (checked into git) |

## How it's used at runtime

`KernelLauncher::ensure_condition_kernel_loaded()` in
`quadrants/runtime/cuda/kernel_launcher.cpp` tries to load the pre-built fatbin
first. If that fails (e.g. running on an SM architecture not included in the
fatbin), it falls back to JIT linking the embedded PTX with `libcudadevrt.a`
from the system's CUDA toolkit.
