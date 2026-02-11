# Quadrants

[Quadrants](https://github.com/Genesis-Embodied-AI/quadrants) was forked in June 2025. This repository (or quadrants) is now a fully independent project with no intention of maintaining backward compatibility with the original taichi. Whilst the repo largely resembles upstream for now, we have made the following changes:
- revamped continuous integration, to run using recent python versions (up to 3.13), recent mac os x versions (up to 15), and to run reliably (at least 90% of runs with correct code succeed)
- added dataclasses.dataclass structs:
    - work with both ndarrays and fields (cf ti.struct (field only), ti.dataclass (field only), ti.data_oriented (field only), argpack (ndarray only))
    - can be passed into child `ti.func`tions (cf argpack)
    - can be nested
    - does not affect kernel runtime speed (kernels see only the underlying arrays, no indirection is added within the kernel layer)
- removed GUI/GGUI, C-API, AOT, DX11, DX12, IOS, Android, OpenGL, GLES, argpack, CLI
- reduced launch latency
    - for example, release 4.0.0 increased the speed of non-batched ndarray on CPU by 4.5x in Genesis benchmarks
    - release 3.2.0 added many optimizations so that ndarrays run much faster, changing from 11x slower than fields before this release, to 1.8x slower than fields with this release. (on a specific Genesis test, using a 5090 GPU)
- reduced warm cache launch latency
    - concretely, on Genesis simulator, running on linux, with an NVidia 5090 GPU, cache load time for single_franka_envs.py changed from 7.2s to 0.3s.
- added `to_dlpack`, which enables zero-copy memory sharing between torch and quadrants, avoiding going through kernels for data-accessors. This significantly improves performance.
- upgraded to LLVM 20
- enabled ARM

# What is quadrants?

Quadrants is a high performance multi-platform compiler, targeted at physics simulations. It compiles Python code into parallelizable kernels that can run on:
- NVidia GPUs, using CUDA
- Vulkan-compatible GPUs, using SPIR-V
- Mac Metal GPUs
- x86 and arm64 CPUs

Quadrants supports automatic differentiation. Quadrants lets you build fully fused GPU kernels, using Python.

[Genesis simulator](https://genesis-world.readthedocs.io/en/latest/)'s best-in-class performance can be largely attributed to Taichi, its underlying GPU acceleration framework for Python. Given how critical is this component, we decided to fork Taichi and build our own very framework from there, so that from now on, we are free to drive its development in the direction that best supports the continuous improvement of Genesis simulator.

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

- The original [Taichi](https://github.com/taichi-dev/taichi) was developed with love by many contributors over many years. For the full list of contributors and credits, see [Original taichi contributors](https://github.com/taichi-dev/taichi?tab=readme-ov-file#contributing)
