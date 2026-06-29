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
> The Supreme Polarity (Taichi) gives rise to the Two Modes (Yin & Yang), which in turn give rise to the Four Forms (_Quadrants_).

_Quadrants_ captures the idea of progression originated from taichi — built on the same foundation, evolving in its own direction while acknowledging its roots.
This project is now fully independent and does not aim to maintain backward compatibility with upstream Taichi.

## How Quadrants differs from upstream Taichi

While the repository still resembles upstream in structure, major changes include:

### Platform support

* LLVM 22, ARM (aarch64) support

### CI

* Kernel-level [code coverage](https://genesis-embodied-ai.github.io/quadrants/user_guide/kernel_coverage.html) — device-side branch coverage in standard `coverage.py` format, integrated with pytest-cov
* AI-driven checks for line wrapping, deleted comments, test coverage, and feature factorization

### Structural improvements

* `dataclasses.dataclass` structs — work with ndarrays and fields, nestable, passable to `qd.func`, zero kernel-runtime overhead
* [`qd.Tensor`](https://genesis-embodied-ai.github.io/quadrants/user_guide/tensor.html) — unified API over fields and ndarrays with per-tensor layout control, pickle support, and a `backend=` switch
* [`BufferView`](https://genesis-embodied-ai.github.io/quadrants/user_guide/buffer_view.html) — safe sub-range ndarray access with bounds checking in debug mode

### Removed components

To focus the compiler and reduce maintenance burden, we removed: GUI/GGUI, C-API, AOT, DX11/DX12, iOS/Android, OpenGL/GLES, argpack, CLI.

### Performance

* **Reduced launch latency** — ndarray CPU performance improved **4.5×**; ndarray GPU performance went from 11× slower than fields to ~30% slower (5090 GPU, Genesis benchmark)
* **[Fastcache](https://genesis-embodied-ai.github.io/quadrants/user_guide/fastcache.html)** — opt-in source-level cache (`@qd.kernel(fastcache=True)`) that bypasses front-end AST parsing; reduces warm-cache kernel load from **7.2 s → 0.3 s** on Genesis benchmarks
* **[GPU Graphs](https://genesis-embodied-ai.github.io/quadrants/user_guide/graph.html)** — `@qd.kernel(graph=True)` captures kernel sequences into a graph; `qd.graph_do_while` runs GPU-side iteration loops (hardware conditional nodes on CUDA SM 9.0+)
* **[perf_dispatch](https://genesis-embodied-ai.github.io/quadrants/user_guide/perf_dispatch.html)** — auto-benchmarks multiple kernel implementations and selects the fastest at runtime
* **[Zero-copy interop](https://genesis-embodied-ai.github.io/quadrants/user_guide/interop.html)** — `to_torch(copy=False)` / `to_numpy(copy=False)` via DLPack on CUDA, CPU, AMDGPU, and Metal; direct torch tensor pass-through into kernels

### SIMT primitives

* **[Tile16x16 / Tile32x32](https://genesis-embodied-ai.github.io/quadrants/user_guide/tile.html)** — register-resident 16×16 and 32×32 matrix tiles with Cholesky, triangular solve, and rank-1 updates; 5× faster than shared-memory baselines on blocked linear algebra
* **[Subgroup ops](https://genesis-embodied-ai.github.io/quadrants/user_guide/subgroup.html)** — cross-platform `shuffle`, `shuffle_down`, `reduce_add`, `reduce_all_add` across CUDA, AMDGPU, Metal and Vulkan

### Autodiff

* [**Autodiff with dynamic loops**](https://genesis-embodied-ai.github.io/quadrants/user_guide/autodiff.html#autodiff-with-dynamic-loops) — computes the gradient of any kernel transparently using reverse-mode differentiation and runtime-based memory allocation
* Forward-mode AD, custom gradients (`@qd.ad.grad_replaced`), `qd.ad.Tape`


### Debugging & development

* **[Python backend](https://genesis-embodied-ai.github.io/quadrants/user_guide/python_backend.html)** — `qd.init(qd.python)` interprets kernels as plain Python so they can be stepped through in a standard Python debugger

---

# Installation
## Prerequisites
- Python 3.10-3.13
- Mac OS 14, 15, Windows, or Ubuntu 22.04-24.04 or compatible
- ROCm 5.2 or newer for AMD GPU support

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
