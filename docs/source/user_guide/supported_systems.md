# Supported systems

## CI Tested systems

We test the following systems in our CI servers:
- Python 3.10-Python 3.13
- Mac OS X 14 and 15
- Ubuntu 22.04 and Ubuntu 24.04
- Windows Server 2025

## Supported systems:

### Operating systems
- Mac Silicon 14 or 15
- Ubuntu 22.04 and 24.04 (x64 and ARM64)
- Windows 10 or later (x86 only)

### GPUs

- CUDA GPUs, `sm_60` (Pascal) through `sm_120` (Blackwell / Thor) — i.e. `>=sm_60` and `<=sm_120`
- Metal GPUs
- AMD GPUs
- Vulkan-compatible GPUs (e.g. Intel Arc)

If you have a newer NVIDIA GPU (above `sm_120`), please [open an issue on the Quadrants repo](https://github.com/Genesis-Embodied-AI/quadrants/issues).

### Backend / OS matrix

Which backends are available on each supported platform. `qd.cpu` and `qd.vulkan` run on every OS; the other GPU backends are platform-specific because they wrap vendor drivers (CUDA on NVIDIA, ROCm on AMD, Metal on Apple).

| OS \ backend | `qd.cpu` | `qd.cuda` | `qd.amdgpu` | `qd.metal` | `qd.vulkan` |
| --- | --- | --- | --- | --- | --- |
| macOS (Apple Silicon) | yes | n/a | n/a | yes | yes |
| Linux x64 | yes | yes | yes | n/a | yes |
| Linux ARM64 | yes | no | no | n/a | yes |
| Windows x86 | yes | yes | no | n/a | yes |
| Windows ARM64 | yes | no | no | n/a | yes |

Notes:
- `qd.cuda` requires an NVIDIA driver + CUDA runtime on the host; quadrants links against the CUDA runtime discovered at import time. NVIDIA ships CUDA for Linux ARM64 and Windows ARM64, but quadrants does not support them yet.
- `qd.amdgpu` currently wires up the Linux x64 ROCm path only. AMD's HIP SDK also ships on Windows and on some Linux ARM64 targets, but quadrants does not support them yet. AMDGPU codegen always runs in wave64 mode, so [`qd.simt.subgroup`](./subgroup.md) primitives operate over 64 lanes regardless of whether the GPU is CDNA (Instinct, wave64-default) or RDNA (Radeon, wave32-default — quadrants overrides it to wave64).
- `qd.metal` is only available on Apple hardware and is the recommended GPU backend there.
- `qd.vulkan` on macOS ships a bundled MoltenVK dylib inside the wheel, so no separate MoltenVK install is required.

### Python backend

A pure-Python backend (`qd.python`) is available on any system where PyTorch is installed. See [Python backend](./python_backend.md).
