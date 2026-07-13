# Building the CUDA graph conditional fatbin

The `graph_do_while` feature uses a tiny CUDA kernel that calls
`cudaGraphSetConditional` (a device-side function from NVIDIA's
`libcudadevrt.a`) to control CUDA graph conditional while nodes. These
conditional nodes require SM 9.0+ (Hopper or later); on older GPUs,
`graph_do_while` falls back to a host-side loop automatically.

There are three distinct phases:

1. **Fatbin generation** (rare, manual) — A developer runs
   `scripts/build_condition_kernel_fatbin.py` (and the sibling
   `build_checkpoint_gate_fatbin.py` / `build_checkpoint_yield_check_fatbin.py`),
   which compiles the kernel and device-links it with `libcudadevrt.a` to resolve
   `cudaGraphSetConditional`. Each script emits **one fatbin per build toolkit**
   (see Prerequisites), all committed to git in a single C header. Requires the
   CUDA toolkits listed below.
2. **Quadrants build** (CI / developers) — The C header is `#include`d as
   plain byte arrays. No CUDA toolkit needed.
3. **Runtime** (end users) — The fatbins are loaded via `cuModuleLoadData`.
   No CUDA toolkit needed.

This page documents phase 1: regenerating the pre-built fatbins.

## When to regenerate

You only need to regenerate the fatbin if:

- The condition kernel source (`quadrants/runtime/cuda/graph_do_while_cond.cu`)
  changes.
- You need to add support for a new SM architecture.

## Prerequisites

Each SM architecture is compiled with the oldest suitable CUDA toolkit, and the
per-toolkit results are bundled into one header. This is what keeps each cubin
loadable on the widest range of drivers: a SASS cubin can only be loaded by a
driver whose CUDA version is `>=` the toolkit that produced it, so building e.g.
sm_120 (RTX 5090) with a newer toolkit than necessary would make it reject older
drivers with `CUDA_ERROR_INVALID_IMAGE`. The per-arch toolkit requirement lives
in the `SM_TOOLKIT` mapping in `scripts/_fatbin_common.py`, where each SM maps to
a version spec — `==X.Y` (exact) or `>=X.Y` (minimum):

- **`==12.8` (matched exactly)** — every architecture except sm_110 (pre-Hopper
  sm_60 / sm_70 / sm_80, Hopper sm_90, Blackwell sm_100 / sm_120). 12.8 is the
  oldest toolkit that covers all shipping Blackwell parts, and is the
  wide-compatibility anchor, so it is matched exactly rather than "or newer".
- **`>=13.0`** — sm_110 (Thor) only; earlier toolkits fail with
  `Unsupported gpu architecture 'compute_110'`. This is a minimum: the oldest
  available toolkit `>=` 13.0 is used (e.g. 13.1 is fine). There is no
  old-driver constraint pulling it lower, since Thor is new hardware.

**All** required toolkits must be installed; the scripts raise
`MissingToolkitError` (listing what is missing and what was discovered) before
compiling anything if any is absent. Toolkits are auto-discovered from
`/usr/local/cuda-*/bin/nvcc`, `nvcc` on `PATH`, and `CUDA_HOME` / `CUDA_PATH`. To
point at toolkits in non-standard locations, set `QUADRANTS_NVCC_CANDIDATES` to an
`os.pathsep`-separated list of nvcc binaries and/or CUDA root directories, e.g.:

```bash
export QUADRANTS_NVCC_CANDIDATES=/opt/cuda-12.8:/opt/cuda-13.1/bin/nvcc
```

## Regenerating

Run the script from the repo root:

```bash
python scripts/build_condition_kernel_fatbin.py
```

This will, for each build toolkit:

1. Compile `quadrants/runtime/cuda/graph_do_while_cond.cu` with relocatable device
   code for that toolkit's target SM architectures.
2. Device-link the result with `libcudadevrt.a` to resolve the
   `cudaGraphSetConditional` extern.

It then writes all the per-toolkit fatbins as C byte arrays (plus a small lookup
table) to `quadrants/runtime/cuda/graph_do_while_cond_fatbin.h`.

Regenerate the checkpoint kernels the same way with
`scripts/build_checkpoint_gate_fatbin.py` and
`scripts/build_checkpoint_yield_check_fatbin.py`. The generator emits a compact
byte-array layout; run `pre-commit run -a` (clang-format) on the regenerated
headers so they match the committed formatting before committing them. Quadrants
must be rebuilt to pick up the new fatbins.

## Verifying the committed fatbins

`tests/python/test_fatbin_arch_coverage.py` holds an explicit, hand-maintained
`EXPECTED_LAYOUT` for each committed `*_fatbin.h` — the architectures and the CUDA
toolkit expected in every blob (e.g. `sm_90 / sm_100 / sm_120` built with CUDA 12.8
in blob 0 and `sm_110` built with CUDA 13.0+ in blob 1 for the condition kernel).
It deliberately does *not* re-derive this from `scripts/_fatbin_common.py`, so a
regression in the `SM_TOOLKIT` mapping is caught rather than silently baked into a
new header. For every embedded cubin it checks both the SASS arch and the build
toolkit that `cuobjdump` reports, so it directly asserts the load-bearing fact
behind #2942: the sm_120 (RTX 5090) cubin is built with the wide-compatibility CUDA
12.8, not a newer toolkit that would reintroduce `CUDA_ERROR_INVALID_IMAGE` on
570-series drivers. It also drift-guards each `SM_VERSIONS` list against the arch
set actually baked into the checked-in header.

The test shells out to `cuobjdump`, so it is skipped unless `cuobjdump` is on
`PATH`; the toolkit providing it must also be at least as new as the newest one
used to build the fatbins (CUDA 13.0+, for sm_110), otherwise the affected header
is skipped rather than failed. Run it after regenerating:

```bash
# Make a recent CUDA toolkit's cuobjdump visible on PATH, e.g.:
export PATH=/usr/local/cuda/bin:$PATH
python -m pytest tests/python/test_fatbin_arch_coverage.py -v
```

## Adding a new SM architecture

Add the new SM version number to the `SM_VERSIONS` list in the relevant
`scripts/build_*_fatbin.py`, and add an entry for it to `SM_TOOLKIT` in
`scripts/_fatbin_common.py`: use `"==X.Y"` to pin it to an exact toolkit (the
oldest one that covers it, as for the `==12.8` archs) or `">=X.Y"` for a minimum
(as done for `110: ">=13.0"`). Then make sure a matching toolkit is installed and
regenerate. (Every SM in `SM_VERSIONS` must have an `SM_TOOLKIT` entry, or the
scripts raise `KeyError`.)

## Files

| File | Purpose |
|------|---------|
| `scripts/_fatbin_common.py` | Shared multi-toolkit build + header generation; arch → toolkit mapping |
| `quadrants/runtime/cuda/graph_do_while_cond.cu` | CUDA C source for the condition kernel |
| `scripts/build_condition_kernel_fatbin.py` | Regeneration script (condition kernel) |
| `quadrants/runtime/cuda/graph_do_while_cond_fatbin.h` | Generated C header (checked into git) |
| `scripts/build_checkpoint_gate_fatbin.py`, `scripts/build_checkpoint_yield_check_fatbin.py` | Regeneration scripts (checkpoint kernels) |
| `quadrants/runtime/cuda/checkpoint_gate_fatbin.h`, `quadrants/runtime/cuda/checkpoint_yield_check_fatbin.h` | Generated C headers (checked into git) |
| `tests/python/test_fatbin_arch_coverage.py` | Regression test: committed headers match an explicit per-blob arch + toolkit layout (needs `cuobjdump`) |

## How it's used at runtime

`GraphManager::ensure_condition_kernel_loaded()` in
`quadrants/runtime/cuda/graph_manager.cpp` loads the fatbins via
`GraphManager::load_first_matching_fatbin()`, which tries each bundled per-toolkit
fatbin with `cuModuleLoadData` and keeps the first that loads (a device only
loads the fatbin containing SASS for its SM). If none contain SASS for the
current GPU's SM architecture, loading fails with a clear error pointing to the
relevant regeneration script. The checkpoint kernels load the same way from
`graph_manager_checkpoint.cpp`.
