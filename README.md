# What is Quadrants?

Quadrants is a high-performance multi-platform compiler for physics simulation. The project began as a fork of the original Taichi by [Genesis AI](https://genesis-ai.company/) in June 2025. As the codebase evolved into a fully independent compiler with its own direction and long-term roadmap, we decided to give it a name that reflects both its roots and its new identity.

Quadrants is a high-performance compiler designed for large-scale physics simulation and robotics workloads. It compiles Python code into highly optimized parallel kernels that run on:

* NVIDIA GPUs (CUDA)
* Vulkan-compatible GPUs (SPIR-V)
* Apple Metal GPUs
* x86 and ARM64 CPUs

## The origin
Quadrants is an open-source project being continuously developped by [Genesis AI](https://genesis-ai.company/). The project was originally forked from the original [Taichi](https://github.com/taichi-dev/taichi) in June 2025. As the original Taichi is no longer being maintained and the codebase evolved into a fully independent compiler with its own direction and long-term roadmap, we decided to give it a name that reflects both its roots and its new identity. The name _Quadrants_ is inspired by the Chinese saying:

> 太极生两仪，两仪生四象
>
> The Supreme Polarity (Taichi) begets the Two Modes (Ying & Yang), which in turn begets the Four Forms (_Quandrants_).

_Quadrants_ captures the idea of progression originated from taichi — built on the same foundation, evolving in its own direction while acknowledging its roots.
This project is now fully independent and does not aim to maintain backward compatibility with upstream Taichi.

## How Quadrants differs from upstream Taichi

While the repository still resembles upstream in structure, major changes include:

### Modernized infrastructure

* Revamped CI
* Support for Python 3.10–3.13
* Support for macOS up to 15
* Significantly improved reliability (≥90% CI success on correct code)

### Structural improvements

* Added `dataclasses.dataclass` structs:

  * Work with both ndarrays and fields
  * Can be passed into child `ti.func` functions
  * Can be nested
  * No kernel runtime overhead (kernels see only underlying arrays)

### Removed components

To focus the compiler and reduce maintenance burden, we removed:

* GUI / GGUI
* C-API
* AOT
* DX11 / DX12
* iOS / Android
* OpenGL / GLES
* argpack
* CLI

### Performance improvements

#### Reduced launch latency

* Release 4.0.0 improved non-batched ndarray CPU performance by **4.5×** in Genesis benchmarks.
* Release 3.2.0 improved ndarray performance from **11× slower than fields** to **1.8× slower** (on a 5090 GPU, Genesis benchmark).

#### Reduced warm-cache latency

On Genesis simulator (Linux + NVIDIA 5090):

* `single_franka_envs.py` cache load time reduced from **7.2s → 0.3s**

#### Zero-copy Torch interop

* Added `to_dlpack`
* Enables zero-copy memory sharing between PyTorch and Quadrants
* Avoids kernel-based accessors
* Significantly improves performance

### Compiler upgrades

* Upgraded to LLVM 20
* Enabled ARM support

---

# Installation
## Prerequisites
- Python 3.10-3.13
- Mac OS 14, 15, Windows, or Ubuntu 22.04-24.04 or compatible

## Procedure
```
pip install gstaichi
```

(For how to build from source, see our CI build scripts, e.g. [linux build scripts](.github/workflows/scripts_new/linux_x86/) )

# Documentation

- [docs](https://genesis-embodied-ai.github.io/gstaichi/user_guide/index.html)
- [API reference](https://genesis-embodied-ai.github.io/gstaichi/autoapi/index.html)

# Something is broken!

- [Create an issue](https://github.com/Genesis-Embodied-AI/gstaichi/issues/new/choose)

# Acknowledgements

Quadrants stands on the shoulders of the original [Taichi](https://github.com/taichi-dev/taichi) project, built with care and vision by many contributors over the years.
For the full list of contributors and credits, see the [original Taichi repository](https://github.com/taichi-dev/taichi).

We are grateful for that foundation.
