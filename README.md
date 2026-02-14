# What is Quadrants?

Quadrants is a high-performance multi-platform compiler for physics simulation being continuously developed by [Genesis AI](https://genesis-ai.company/).

It is designed for large-scale physics simulation and robotics workloads. It compiles Python code into highly optimized parallel kernels that run on:

* NVIDIA GPUs (CUDA)
* Vulkan-compatible GPUs (SPIR-V)
* Apple Metal GPUs
* AMD GPUs (ROCm HIP)
* x86 and ARM64 CPUs

## The origin

The quadrants project was originally forked from [Taichi](https://github.com/taichi-dev/taichi) in June 2025. As the original Taichi is no longer being maintained and the codebase evolved into a fully independent compiler with its own direction and long-term roadmap, we decided to give it a name that reflects both its roots and its new identity. The name _Quadrants_ is inspired by the Chinese saying:

> 太极生两仪，两仪生四象
>
> The Supreme Polarity (Taichi) gives rise to the Two Modes (Ying & Yang), which in turn give rise to the Four Forms (_Quadrants_).

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
pip install quadrants
```

(For how to build from source, see our CI build scripts, e.g. [linux build scripts](.github/workflows/scripts_new/linux_x86/) )

# Documentation

- [docs](https://genesis-embodied-ai.github.io/quadrants/user_guide/index.html)
- [API reference](https://genesis-embodied-ai.github.io/quadrants/autoapi/index.html)

# Something is broken!

- [Create an issue](https://github.com/Genesis-Embodied-AI/quadrants/issues/new/choose)

# Acknowledgements

Quadrants stands on the shoulders of the original [Taichi](https://github.com/taichi-dev/taichi) project, built with care and vision by many contributors over the years.
For the full list of contributors and credits, see the [original Taichi repository](https://github.com/taichi-dev/taichi).

We are grateful for that foundation.
