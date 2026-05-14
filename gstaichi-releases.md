# Releases — Genesis-Embodied-AI/gstaichi

This file contains the old releases of this repo when it was called gstaichi, before we renamed it to Quadrants.

## [v4.7.0b1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.7.0b1)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.7.0b1` |
| **Date** | 2026-01-16 |
| **Commit** | `hp/disable-o3-windows` |

# Pre-release v4.7.0b1

This pre-release is to test changing llvm optimization on windows from -O3 to -O1.

## What's Changed
### cuda,amdgpu,cpu,vulkan
* [cuda,amdgpu,cpu,vulkan] Add public API for kernel clock counter. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/314
### Perf
* [Perf] dlpack with amdgpu by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/313
### Misc
* [MISC] Rename excluded_parameters => template_slot_locations by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/338
* [MISC] Create ASTTransformerGlobalContext and remove Kernel from Runtime by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/340
* [MISC] Add xfailing py dataclass tests, and test instrumentation by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/342
* [MISC] Rename leaves => parameters by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/343
* [MISC] Improve information in exceptions during fuse_args by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/341
* [Misc] Add ti.clock_speed_hz() by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/346
### Type
* [Type] Py dataclass arguments can be renamed by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/353


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.6.0...v4.7.0b1

---

## [v4.6.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.6.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.6.0` |
| **Date** | 2025-12-30 |
| **Commit** | `main` |

# Release v4.6.0

This release introduces a number of performance improvements when running single-threaded on CPU.

## What's Changed
### Perf
* [Perf] Early return earlier when materializing kernels. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/324
* [Perf] Cache compiled kernel data systematically. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/325
* [Perf] Enable dlpack on metal for pytorch ! <= 2.9.1 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/336

### Misc
* [Misc] Refactor part1: ONLY moves by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/326
* [Misc] Refactor _func_base.py functions into new base class FuncBase by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/327
* [Misc] Factorize out ASTGenerator, and remove debug dump to checksums.csv by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/328
* [Misc] Fuse extract_args method by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/330
* [Misc] Factorize out _try_load_fastcache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/331
* [Misc] Factorize out launch context buffer cache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/329
* [Misc] Miscellaneous refactors around kernel.py and associated files by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/332
* [Misc] Rename ASTTransformerContext to ASTTransformerFuncContext by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/334
* [Misc] Renaming args => py_args; process_args => fuse_args by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/335


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.5.0...v4.6.0

---

## [v4.6.0b4](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.6.0b4)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.6.0b4` |
| **Date** | 2025-12-27 |
| **Commit** | `hp/renaming-dataclass-args-v2` |

# Pre-release v4.6.0b4

This pre-release is to test enabling renaming py dataclass parameters when passing to sub-functions.

## What's Changed
### Type
* [Type] Can rename py dataclass structs when calling sub functions by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/333
### Perf
* [Perf] Early return earlier when materializing kernels. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/324
* [Perf] Cache compiled kernel data systematically. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/325
### Misc
* [Misc] Refactor part1: ONLY moves by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/326
* [Misc] Refactor _func_base.py functions into new base class FuncBase by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/327
* [Misc] Factorize out ASTGenerator, and remove debug dump to checksums.csv by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/328
* [Misc] Fuse extract_args method by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/330
* [Misc] Factorize out _try_load_fastcache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/331
* [Misc] Factorize out launch context buffer cache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/329
* [Misc] Miscellaneous refactors around kernel.py and associated files by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/332


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.5.0...v4.6.0b4

---

## [v4.6.0b3](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.6.0b3)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.6.0b3` |
| **Date** | 2025-12-27 |
| **Commit** | `hp/renaming-dataclass-args-v2` |

# Pre-release v4.6.0b3

This pre-release is to test enabling renaming py dataclass parameters when passing to sub-functions.

## What's Changed
### Type
* [Type] Can rename py dataclass structs when calling sub functions by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/333
### Perf
* [Perf] Early return earlier when materializing kernels. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/324
* [Perf] Cache compiled kernel data systematically. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/325
### Misc
* [Misc] Refactor part1: ONLY moves by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/326
* [Misc] Refactor _func_base.py functions into new base class FuncBase by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/327
* [Misc] Factorize out ASTGenerator, and remove debug dump to checksums.csv by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/328
* [Misc] Fuse extract_args method by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/330
* [Misc] Factorize out _try_load_fastcache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/331
* [Misc] Factorize out launch context buffer cache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/329
* [Misc] Miscellaneous refactors around kernel.py and associated files by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/332


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.5.0...v4.6.0b3

---

## [v4.6.0rc3](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.6.0rc3)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.6.0rc3` |
| **Date** | 2025-12-27 |
| **Commit** | `main` |

# Pre-release v4.6.0rc3

This pre-release provides two performance improvements from @duburcqa .

## What's Changed
### Perf
* [Perf] Early return earlier when materializing kernels. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/324
* [Perf] Cache compiled kernel data systematically. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/325
### Misc
* [Misc] Refactor part1: ONLY moves by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/326
* [Misc] Refactor _func_base.py functions into new base class FuncBase by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/327
* [Misc] Factorize out ASTGenerator, and remove debug dump to checksums.csv by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/328
* [Misc] Fuse extract_args method by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/330
* [Misc] Factorize out _try_load_fastcache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/331
* [Misc] Factorize out launch context buffer cache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/329
* [Misc] Miscellaneous refactors around kernel.py and associated files by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/332
* [Misc] Rename ASTTransformerContext to ASTTransformerFuncContext by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/334


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.5.0...foo

---

## [v4.6.0b2](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.6.0b2)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.6.0b2` |
| **Date** | 2025-12-26 |
| **Commit** | `hp/renaming-dataclass-args-v2` |

# Pre-release v4.6.0b2

This pre-release is to test enabling renaming py dataclass parameters when passing to sub-functions.

## What's Changed
### Type
* [Type] Can rename py dataclass structs when calling sub functions by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/333
### Perf
* [Perf] Early return earlier when materializing kernels. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/324
* [Perf] Cache compiled kernel data systematically. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/325
### Misc
* [Misc] Refactor part1: ONLY moves by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/326
* [Misc] Refactor _func_base.py functions into new base class FuncBase by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/327
* [Misc] Factorize out ASTGenerator, and remove debug dump to checksums.csv by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/328
* [Misc] Fuse extract_args method by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/330
* [Misc] Factorize out _try_load_fastcache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/331
* [Misc] Factorize out launch context buffer cache by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/329
* [Misc] Miscellaneous refactors around kernel.py and associated files by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/332


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.5.0...v4.6.0b2

---

## [v4.6.0rc2](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.6.0rc2)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.6.0rc2` |
| **Date** | 2025-12-18 |
| **Commit** | `hp/func-base-refactorization` |

# Pre-release v4.6.0rc2

This pre-release is for a behind-the-scenes refactor of kernel_impl.py

## What's Changed
- [Misc] Refactorizing kernel_impl.py by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/319

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.6.0...v4.6.0rc2

---

## [v4.6.0rc1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.6.0rc1)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.6.0rc1` |
| **Date** | 2025-12-18 |
| **Commit** | `hp/func-base-refactorization` |

# Pre-release v4.6.0rc1

This pre-release is for a behind-the-scenes refactor of kernel_impl.py

## What's Changed
- [Misc] Refactorizing kernel_impl.py by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/319

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.6.0...v4.6.0rc1

---

## [v4.5.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.5.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.5.0` |
| **Date** | 2025-12-18 |
| **Commit** | `main` |

# Release v4.5.0

This release provides new performance improvements.

## What's Changed
### Perf
* [perf] Speed up python-side arg processing. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/323
* [Perf] Speed up computation of cache key. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/321
### Misc
* [Misc] Add optimization instrumentation to dump kernels to files by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/317
* [Misc] Add TI_DUMP_CFG to dump CFG graph during optimization passes by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/318
* [Misc] Remove unused inlining module by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/316
* [Misc] TI_DUMP_IR honors config.debug_dump_path by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/322


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.4.0...v4.5.0

---

## [v4.5.0b2](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.5.0b2)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.5.0b2` |
| **Date** | 2025-12-17 |
| **Commit** | `hp/func-base-refactorization` |

# Pre-release v4.5.0b2

This pre-release is to test that a refactorization of kernel_impl.py, under the hood, doesn't break anything.

## What's Changed
- [Misc] Refactorizing kernel_impl.py by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pulls

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.4.0...v4.5.0b2

---

## [v4.5.0b1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.5.0b1)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.5.0b1` |
| **Date** | 2025-12-17 |
| **Commit** | `hp/func-base-refactorization` |

# Pre-release v4.5.0b1

This pre-release is to test that a refactorization of kernel_impl.py, under the hood, doesn't break anything.

## What's Changed
- [Misc] Refactorizing kernel_impl.py by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pulls

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.4.0...v4.5.0b1

---

## [v4.4.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.4.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.4.0` |
| **Date** | 2025-12-17 |
| **Commit** | `main` |

# Release v4.4.0

This release enables AMDGPU in the WIndows and Linux wheels, improves performance for kernels having templated primitive parameters, and fixes a segfault when using to_dlpack on fields.

## What's Changed
### Perf
* [Perf] Fix edge-cases defeating caching mechanism by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/320
### Bug
* [Bug] Fix dlpack segfault for field by @erizmr in https://github.com/Genesis-Embodied-AI/gstaichi/pull/312
### AMDGPU
* [AMDGPU] Enable AMDGPU build and AMDGPU test runner by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/306
### Build
* [Build] Remove conda by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/311
* [Build] Remove pybind11 and libc++-*-dev from linux build by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/305
* [Build] Windows build uses clang 20 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/310


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.3.1...v4.4.0

---

## [v4.4.0rc2](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.4.0rc2)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.4.0rc2` |
| **Date** | 2025-12-17 |
| **Commit** | `duburcqa/fix_caching_edge_cases` |

# Pre-release v4.4.0rc2

## What's Changed
* [Perf] Fix edge-cases defeating caching mechanism by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/320
* [Build] Remove conda by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/311
* [Build] Remove pybind11 and libc++-*-dev from linux build by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/305
* [Bug] Fix dlpack segfault for field by @erizmr in https://github.com/Genesis-Embodied-AI/gstaichi/pull/312
* [Build] Windows build uses clang 20 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/310
* [AMDGPU] Enable AMDGPU build and AMDGPU test runner by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/306


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.3.1...v4.4.0rc2

---

## [v4.4.0rc1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.4.0rc1)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.4.0rc1` |
| **Date** | 2025-12-17 |
| **Commit** | `duburcqa/fix_caching_edge_cases` |

# Pre-release v4.4.0rc1

## What's Changed
* [Perf] Fix edge-cases defeating caching mechanism by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/320
* [Build] Remove conda by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/311
* [Build] Remove pybind11 and libc++-*-dev from linux build by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/305
* [Bug] Fix dlpack segfault for field by @erizmr in https://github.com/Genesis-Embodied-AI/gstaichi/pull/312
* [Build] Windows build uses clang 20 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/310
* [AMDGPU] Enable AMDGPU build and AMDGPU test runner by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/306


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.3.1...v4.4.0rc1

---

## [v4.4.0b3](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.4.0b3)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.4.0b3` |
| **Date** | 2025-12-17 |
| **Commit** | `hp/func-base-refactorization-factorize-cache` |

# v4.4.0b3

This pre-release is to test the kernel_impl.py refactor.

## What's Changed
* [Misc] Refactorizing kernel_impl.py
* [Build] Remove conda by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/311
* [Build] Remove pybind11 and libc++-*-dev from linux build by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/305
* [Bug] Fix dlpack segfault for field by @erizmr in https://github.com/Genesis-Embodied-AI/gstaichi/pull/312
* [Build] Windows build uses clang 20 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/310
* [AMDGPU] Enable AMDGPU build and AMDGPU test runner by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/306


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.3.1...v4.4.0b3

---

## [v4.4.0b2](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.4.0b2)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.4.0b2` |
| **Date** | 2025-12-16 |
| **Commit** | `hp/func-base-refactorization` |

# v4.4.0b2

This pre-release is to test the kernel_impl.py refactor.

## What's Changed
* [Misc] Refactorizing kernel_impl.py
* [Build] Remove conda by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/311
* [Build] Remove pybind11 and libc++-*-dev from linux build by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/305
* [Bug] Fix dlpack segfault for field by @erizmr in https://github.com/Genesis-Embodied-AI/gstaichi/pull/312
* [Build] Windows build uses clang 20 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/310
* [AMDGPU] Enable AMDGPU build and AMDGPU test runner by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/306


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.3.1...v4.4.0b2

---

## [v4.4.0b1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.4.0b1)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.4.0b1` |
| **Date** | 2025-12-16 |
| **Commit** | `hp/func-base-refactorization` |

# v4.4.0b1

This pre-release is to test the kernel_impl.py refactor.

## What's Changed
* [Misc] Refactorizing kernel_impl.py
* [Build] Remove conda by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/311
* [Build] Remove pybind11 and libc++-*-dev from linux build by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/305
* [Bug] Fix dlpack segfault for field by @erizmr in https://github.com/Genesis-Embodied-AI/gstaichi/pull/312
* [Build] Windows build uses clang 20 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/310
* [AMDGPU] Enable AMDGPU build and AMDGPU test runner by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/306


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.3.1...v4.4.0b1

---

## [v4.3.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.3.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.3.1` |
| **Date** | 2025-12-03 |
| **Commit** | `main` |

# Release v4.3.1

This release fixes a bug with offline cache.

## What's Changed
* [Bug] Fix parallel cache write by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/309


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.3.0...v4.3.1

---

## [v4.3.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.3.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.3.0` |
| **Date** | 2025-11-29 |
| **Commit** | `main` |

# Release v4.3.0

This release provides some additional kernel launchtime accelerations, fixes several fastcache bugs, and several to_dlpack bugs.

## What's Changed

### Perf
* [Pref] Avoid runtime overhead due to custom data_oriented attribute getter. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/279
* [Perf] Redo fastcache kernel key to reuse existing front end cache key by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/283
* [Perf] Speed up argument processing on Python-side. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/282
* [Perf] Implement to_dlpack for ndarrays on Metal by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/287
* [Perf] Add fastcache test for dupe kernels and fix failure by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/286
* [Perf] Fastcache key contains gstaichi version by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/289

### Bug
* [Bug] Fix dlpack zero-copy memory alignment by @erizmr in https://github.com/Genesis-Embodied-AI/gstaichi/pull/298
* [Bug] Fix ndarray memory leak by @BernardoCovas and @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/278

### Build
* [Build] Remove Dockerfile by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/276
* [Build] Pin manylinux arm to 2025.11.11-1 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/285
* [Build] Change Mac CI build from Mac 15 to Mac 26 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/288
* [Build] Only use clang 20 or un-versioned for linux build by @hughperkins in https://github.com/Genesis-Embodied
* [Build] Remove uploads of wheels to aws by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/292
AI/gstaichi/pull/290
* [Build] Remove some final clang 20 warnings about using VLAs by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/295
* [Build] Split linux CI gpu tests into three parallel jobs by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/300

### Misc
* [Misc] Add ti.dump_compile_config() by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/281
* [Misc] Fix homepage URL in pyproject.toml by @oliver-batchelor-work in https://github.com/Genesis-Embodied-AI/gstaichi/pull/291

### Test
* [Test] Add tests for consistency between kernel accessors, external accessors, to/from numpy by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/296

### Type
* [Type] Fix u1 consistency tests for vulkan and metal by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/302
* [Type] Enable dlpack for metal u1 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/303

### SPIRV
* [SPIRV] Add additional spir-v dump before optimization by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/304

## New Contributors
* @oliver-batchelor-work made their first contribution in https://github.com/Genesis-Embodied-AI/gstaichi/pull/291

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.2.0...v4.3.0

---

## [v4.2.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.2.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.2.0` |
| **Date** | 2025-11-17 |
| **Commit** | `main` |

# Release v4.2.0

This release upgrades gstaichi to use LLVM 20, enabling use of compute capability up to and including sm_120.

## What's Changed
### Build
* [Build] Remove libjpg by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/277
* [Build] Llvm 20 by @johnnynunez and @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/275


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.1.0...v4.2.0

---

## [v4.2.0b1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.2.0b1)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.2.0b1` |
| **Date** | 2025-11-17 |
| **Commit** | `hp/llvm-20` |

# Pre-release v4.2.0b1

This pre-release tests upgradsing to LLVM 20.

- [Build] Upgrade to LLVM-20 by @hughperkins  in https://github.com/Genesis-Embodied-AI/gstaichi/pull/275

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.1.0...v4.2.0b1

---

## [v4.1.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.1.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.1.0` |
| **Date** | 2025-11-16 |
| **Commit** | `main` |

# Release 4.1.0

This release adds `to_dlpack`, which provides zero-copy usage of gstaichi tensors in torch; and upgrades LLVM from 15.0.7 to 18.1.8.

## What's Changed
### Type
* [Type] Add to_dlpack to ndarray tensors by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/270
* [Type] Add to_dlpack for dense fields by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/272
### Build
* [Build] LLVM-18 by @johnnynunez  and @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/274


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.0.0...v4.1.0

---

## [v4.1.0b6](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.1.0b6)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.1.0b6` |
| **Date** | 2025-11-16 |
| **Commit** | `hp/llvm-18-v2` |

# Pre-release v4.1.0b6

This pre-release is in order to test migrating to LLVM-18.

## What's Changed
### Build
* [Build] Migrate to LLVM-18 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/274
### Type
* [Type] Add to_dlpack to ndarray tensors by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/270
* [Type] Add to_dlpack for dense fields by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/272


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.0.0...v4.1.0b6

---

## [v4.1.0b2](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.1.0b2)  *(pre-release)*

| Field  | Value |
|--------|-------|
| **Tag** | `v4.1.0b2` |
| **Date** | 2025-11-15 |
| **Commit** | `hp/llvm-18-v2` |

# Pre-release v4.1.0b2

This pre-release is in order to test migrating to LLVM-18.

## What's Changed
### Build
* [Build] Migrate to LLVM-18 by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/273
### Type
* [Type] Add to_dlpack to ndarray tensors by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/270
* [Type] Add to_dlpack for dense fields by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/272


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v4.0.0...v4.1.0b2

---

## [Release 4.0.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v4.0.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v4.0.0` |
| **Date** | 2025-11-12 |
| **Commit** | `main` |

This release increases the speed of non-batched ndarray on CPU by 4.5x in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) benchmarks. We are removing support for textures, hence the major version bump.

## What's Changed

### CPU

* [cpu] Move from 'nehalem' to 'x86-64-v3' as x86 micro-architecture baseline. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/265

### Misc

* [Misc] Remove textures by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/268

### Perf

* [Perf] Prune unused dataclass fields from being passed to kernels by @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/259
* [Perf] Add template mapper key caching. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/264
* [Perf] Avoid dynamic cast if possible. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/263
* [Perf] Replace dynamic-size std::vector by fixed-size std::array. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/266
* [Perf] Do not copy kernel parameters. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/267
* [Perf] Minor launch kernel python overhead optimisation. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/269

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v3.3.0b7...v4.0.0

---

## [v3.3.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v3.3.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v3.3.0` |
| **Date** | 2025-11-07 |
| **Commit** | `main` |

# Release 3.3.0

This pre-release adds support for ARM on Linux, and a further ndarray performance improvement.

## What's Changed
* [Perf] Minor performance improvement. by @duburcqa in https://github.com/Genesis-Embodied-AI/gstaichi/pull/258
* [Build] Add build for ARM by @johnnynunez and @hughperkins in https://github.com/Genesis-Embodied-AI/gstaichi/pull/261

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v3.2.1...v3.3.0

---

## [v3.2.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v3.2.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v3.2.1` |
| **Date** | 2025-11-02 |
| **Commit** | `main` |

# Release v3.2.1

This pre-release fixes a bug with ndarray optimizations.

## What's Changed

### Perf
* [Perf] Store template mapper key at instance level in https://github.com/Genesis-Embodied-AI/gstaichi/pull/260


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v3.2.0...v3.2.1

---

## [v3.2.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v3.2.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v3.2.0` |
| **Date** | 2025-11-02 |
| **Commit** | `main` |

# Release v3.2.0

This release adds many optimizations so that ndarrays run much faster, changing from 11x slower than fields before this release, to 1.8x slower than fields with this release. (on a specific Genesis test, using a 5090 GPU)

## What's Changed
### Perf
* Optimize kernel launch overhead. in https://github.com/Genesis-Embodied-AI/gstaichi/pull/250
* Cleanup launch kernel logics in https://github.com/Genesis-Embodied-AI/gstaichi/pull/251
* Further optimization of kernel launch overhead. in https://github.com/Genesis-Embodied-AI/gstaichi/pull/252
* Optimize of kernel launch overhead on C++ side. in https://github.com/Genesis-Embodied-AI/gstaichi/pull/253
* Diagnose launch context buffer cache-ability. in https://github.com/Genesis-Embodied-AI/gstaichi/pull/254
* Add python caching of launch context buffer. in https://github.com/Genesis-Embodied-AI/gstaichi/pull/255

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v3.1.1...v3.2.0

---

## [v3.1.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v3.1.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v3.1.1` |
| **Date** | 2025-10-24 |
| **Commit** | `main` |

# Release v3.1.1

This release fixes a critical bug in fastcache source code reader.

## What's Changed
* [Bug] Improve robustness of fastcache source code file reader in https://github.com/Genesis-Embodied-AI/gstaichi/pull/249


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v3.1.0...v3.1.1

---

## [v3.1.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v3.1.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v3.1.0` |
| **Date** | 2025-10-22 |
| **Commit** | `main` |

# Release v3.1.0

This release fixes various fast cache bugs, in particular it aims at fixing fast cache corruption that was occuring.

## What's Changed
### Build
* [Build] Free up CI runner disk space in https://github.com/Genesis-Embodied-AI/gstaichi/pull/244
### Perf
* [Perf] Add fastcache= to ti.kernel and mark other approaches as deprecated in https://github.com/Genesis-Embodied-AI/gstaichi/pull/243
* [Perf] Reduce fast cache spam for data oriented members in https://github.com/Genesis-Embodied-AI/gstaichi/pull/242
* [Perf] Avoid fast cache corruption and recover from errors in https://github.com/Genesis-Embodied-AI/gstaichi/pull/239
* [Perf] Enable NamedTuple data oriented classes for fastcache in https://github.com/Genesis-Embodied-AI/gstaichi/pull/248
* [Perf] Fast cache works with derived torch tensors now. in https://github.com/Genesis-Embodied-AI/gstaichi/pull/241
### Lang
* [Lang] Add option to raise exception if use templated floats in https://github.com/Genesis-Embodied-AI/gstaichi/pull/247

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v3.0.0...v3.1.0

---

## [v3.0.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v3.0.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v3.0.0` |
| **Date** | 2025-10-14 |
| **Commit** | `main` |

# Release 3.0.0

This adds additional validation for pure kernels, and for AD-compatible kernels. It automatically adds primitive values in data oriented objects to the fast cache key. The major version upgrade is because adstack is no longer automatically enabled, and needs to be activated explicitly using new option `ti.init(ad_stack_experimental_enable=True)`.

## What's Changed
### Test
* [Test] Completely skip test_matrix_ndarray_oob on windows in https://github.com/Genesis-Embodied-AI/gstaichi/pull/238
### Misc
* [Misc] Remove apparently unused fp16 includes in https://github.com/Genesis-Embodied-AI/gstaichi/pull/199
### Perf
* [Perf] Add primitive values in data oriented to fastcache key in https://github.com/Genesis-Embodied-AI/gstaichi/pull/237
* [Perf] Detect pure violation in https://github.com/Genesis-Embodied-AI/gstaichi/pull/230
### Autodiff
* [Autodiff] Fail on non-static range in backwards in https://github.com/Genesis-Embodied-AI/gstaichi/pull/229


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.6.1...v3.0.0

---

## [v2.6.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.6.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.6.1` |
| **Date** | 2025-10-10 |
| **Commit** | `main` |

# Release 2.6.1

This patch release fixes an OOM issue we encountered in Genesis CI for push_differentiable.py example.

## What's Changed
### Bug
* [Bug] Fix memory OOM in Genesis push_differentiable example in https://github.com/Genesis-Embodied-AI/gstaichi/pull/228
* [Bug] Fix dataclass expansion with kwargs, and add test for this in https://github.com/Genesis-Embodied-AI/gstaichi/pull/235
### Doc
* [Doc] 95pct CI build succes to 80pct in https://github.com/Genesis-Embodied-AI/gstaichi/pull/224
## Build
* [Build] Rename linux_x86 folder and file to linux in https://github.com/Genesis-Embodied-AI/gstaichi/pull/236
### Test
* [Test] Disable flaky OOB test on windows in https://github.com/Genesis-Embodied-AI/gstaichi/pull/234


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.6.0...v2.6.1

---

## [v2.6.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.6.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.6.0` |
| **Date** | 2025-10-07 |
| **Commit** | `main` |

# Release 2.6.0

This release doesn't change anything user-facing. Under the hood, we are migrating to our in-house built llvm, as the first step of upgrading LLVM to a newer version.

## What's Changed
### Type
* [Type] Add tests for () indexing, and ndarray ndim == 0 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/233
### Build
* [Build] Hopefully fix autoapi for docs build in https://github.com/Genesis-Embodied-AI/gstaichi/pull/232
* [Build] Make build no longer download pybind11 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/231
* [Build] Build using freshly built LLVM 15.0.7 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/216
### Test
* [Test] Improve error message for TI_LIB_DIR when running c++ unit tests in https://github.com/Genesis-Embodied-AI/gstaichi/pull/223


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.5.0...v2.6.0

---

## [v2.5.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.5.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.5.0` |
| **Date** | 2025-10-04 |
| **Commit** | `main` |

# Release 2.5.0

This release fixes two bugs on Mac Metal that were causing some [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) CI tests to fail.

## What's Changed
### Build
* [build] Try moving more things into pyproject.toml in https://github.com/Genesis-Embodied-AI/gstaichi/pull/189
* [Build] Fix error function in https://github.com/Genesis-Embodied-AI/gstaichi/pull/214
* [Build] Split mac build into build and test in https://github.com/Genesis-Embodied-AI/gstaichi/pull/217
* [Build] Remove git fetch tags, and make changelog, and some other files from misc in https://github.com/Genesis-Embodied-AI/gstaichi/pull/215
* [Build] Windows pypi publish waits for test to finish first in https://github.com/Genesis-Embodied-AI/gstaichi/pull/221
* [Build] Linux pypi publish depends also on test gpu in https://github.com/Genesis-Embodied-AI/gstaichi/pull/222
### Type
* [Type] Remove type: ignore from some files in gstaichi folder in https://github.com/Genesis-Embodied-AI/gstaichi/pull/183
### Mac
* [Mac] Fix spirv return values in https://github.com/Genesis-Embodied-AI/gstaichi/pull/212
* [Bug] Fix issue with missing tmp222 in spirv in https://github.com/Genesis-Embodied-AI/gstaichi/pull/211
### Misc
* [Misc] Add capabilities to facilitate fastcache debugging in https://github.com/Genesis-Embodied-AI/gstaichi/pull/206
* [Misc] Move taskgen class declaration to header in https://github.com/Genesis-Embodied-AI/gstaichi/pull/210


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.4.0...v2.5.0

---

## [v2.4.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.4.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.4.0` |
| **Date** | 2025-09-16 |
| **Commit** | `main` |

# Release v2.4.0

This release:
adds a new `pure: bool` parameter to `@ti.kernel`, which marks a kernel as only accessing data passed in as kernel parameters, and therefore eligible for fast src-ll cache. It is equivalent to adding `@ti.pure` in front of the `@ti.kernel` annotation, but easier to parametrize.
upgrades many external dependencies, and removes unused ones

## What's Changed
### Build
* [Build] Split windows build into build job and test job, so reruns are faster in https://github.com/Genesis-Embodied-AI/gstaichi/pull/191
### Lang
* [Lang] Add `pure` parameter to `@ti.kernel` in https://github.com/Genesis-Embodied-AI/gstaichi/pull/190
### Misc
* [Misc] Update googletest to 1.17.0 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/185
* [Misc] Remove GLFW library in https://github.com/Genesis-Embodied-AI/gstaichi/pull/201
* [Misc] Remove GLM library in https://github.com/Genesis-Embodied-AI/gstaichi/pull/198
* [Misc] Upgrade all Vulkan external libraries to 1.4.321 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/196
* [Misc] Upgrade vulkan sdk to 1.4.321.1 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/204
* [Misc] Upgrade Eigen to commit 70d8d9 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/192
### CUDA
* [Cuda] Remove misleading cuda_version() function, and add link to posts about slim libdevice.10.bc in https://github.com/Genesis-Embodied-AI/gstaichi/pull/202


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.3.1...v2.4.0

---

## [v2.3.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.3.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.3.1` |
| **Date** | 2025-09-12 |
| **Commit** | `main` |

# Release 2.3.1

This patch release fixes two bugs in fast cache:
- templated primitive kernel parameters are now handled correctly
- return values from a kernel no longer cause a crash

## What's Changed
### Perf
* [Perf] Cache template values for fast cache in https://github.com/Genesis-Embodied-AI/gstaichi/pull/177
* [Perf] Fix issue with return type with fastcache, and add unit test for this in https://github.com/Genesis-Embodied-AI/gstaichi/pull/187
### Build
* [Build] Ruff now checks for unused imports in https://github.com/Genesis-Embodied-AI/gstaichi/pull/178
* [Build] Migrate build metadata to pyproject.toml in https://github.com/Genesis-Embodied-AI/gstaichi/pull/179
### Misc
* [Misc] Delete non-working version check in https://github.com/Genesis-Embodied-AI/gstaichi/pull/182


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.3.0...v2.3.1

---

## [v2.3.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.3.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.3.0` |
| **Date** | 2025-09-09 |
| **Commit** | `main` |

# Release 2.3.0

This release adds the possibility of using a static inline `if` as the top level expression in a `for`-loop iterator.

## What's Changed
* [Build] Remove directx headers in https://github.com/Genesis-Embodied-AI/gstaichi/pull/175
* [Lang] Allow static if expression as top level in for loop iterator in https://github.com/Genesis-Embodied-AI/gstaichi/pull/176


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.2.1...v2.3.0

---

## [v2.2.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.2.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.2.1` |
| **Date** | 2025-09-08 |
| **Commit** | `main` |

# Release 2.2.1

This patch release fixes an issue with fast cache that meant one had to run the same script 3 times to be fully cached; and a crash bug after ti.reset for ndarrays.

## What's Changed
* [Type] Fix NotImplementedError in https://github.com/Genesis-Embodied-AI/gstaichi/pull/165
* [Perf] Ensure consistent module name when using src-ll cache in https://github.com/Genesis-Embodied-AI/gstaichi/pull/164
* [Bug] Fix ndarray crash after ti reset in https://github.com/Genesis-Embodied-AI/gstaichi/pull/172
* [Misc] Better error reporting in https://github.com/Genesis-Embodied-AI/gstaichi/pull/167


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.2.0...v2.2.1

---

## [v2.2.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.2.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.2.0` |
| **Date** | 2025-09-04 |
| **Commit** | `main` |

# 2.2.0

This release is focusing on enabling src-ll cache for [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).

## What's Changed
### Misc
* [Misc] Add logging for invalid params to pure kernels in https://github.com/Genesis-Embodied-AI/gstaichi/pull/144
* [Misc] Add ti init option print_non_pure in https://github.com/Genesis-Embodied-AI/gstaichi/pull/145
### Build
* [Build] Pin pytest-rerunfailures to < 16 in https://github.com/Genesis-Embodied-AI/gstaichi/pull/162
### Perf
* [Perf] Disable fast cache for fields in https://github.com/Genesis-Embodied-AI/gstaichi/pull/163
### Test
* [Test] Instrument fe-ll-cache so we can test when we get a cache hit in https://github.com/Genesis-Embodied-AI/gstaichi/pull/161


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.1.1...v2.2.0

---

## [v2.1.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.1.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.1.1` |
| **Date** | 2025-08-27 |
| **Commit** | `main` |

# Release 2.1.1

Patch release to fix a bug in SRC-LL-Cache that caused repeated calls to a cached function to fail.

## What's Changed
### Type
* [Type] Revise the ndarray annotations on test_ndarray.py in https://github.com/Genesis-Embodied-AI/gstaichi/pull/134
* [Type] Clean typing on misc.py in https://github.com/Genesis-Embodied-AI/gstaichi/pull/142
### Build
* [Build] Wheel built on macosx 15 runs on lower mac versions in https://github.com/Genesis-Embodied-AI/gstaichi/pull/154
* [Build] Docs are built with correct version number displayed now in https://github.com/Genesis-Embodied-AI/gstaichi/pull/157
### Misc
* [Misc] Add src_ll_cache flag to ti.init to disable src-ll-cache in https://github.com/Genesis-Embodied-AI/gstaichi/pull/143
* [Misc] Remove superfluous self.compiled-kernel_data = None in https://github.com/Genesis-Embodied-AI/gstaichi/pull/160
### Perf
* [Perf] Improve testing of pure functions to call function repeatedly after cache load; and fix failure in https://github.com/Genesis-Embodied-AI/gstaichi/pull/159

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.1.0...v2.1.1

---

## [v2.1.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.1.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.1.0` |
| **Date** | 2025-08-26 |
| **Commit** | `main` |

# Release 2.1.0

This release removes spam associated with the new PTX cache, and removes the incorrect warning about the wheel being 'restricted'. We also start to add some initial documentation.

## Highlights
### Build
* [Build] Remove restricted warning in https://github.com/Genesis-Embodied-AI/gstaichi/pull/149
### Misc
* [Misc] Reduce ptx cache spam in https://github.com/Genesis-Embodied-AI/gstaichi/pull/156
### Doc
* [Doc] Add new doc in https://github.com/Genesis-Embodied-AI/gstaichi/pull/89

## What's Changed
### Test
* [Test] Fix broken merge of test ndarray max num args skip in https://github.com/Genesis-Embodied-AI/gstaichi/pull/140
### Doc
* [Doc] Add new doc in https://github.com/Genesis-Embodied-AI/gstaichi/pull/89
* [Doc] Remove copyright from header in https://github.com/Genesis-Embodied-AI/gstaichi/pull/150
* [Doc] Fix readme links to docs. in https://github.com/Genesis-Embodied-AI/gstaichi/pull/148
### Build
* [Build] Remove restricted warning in https://github.com/Genesis-Embodied-AI/gstaichi/pull/149
### Misc
* [Misc] Reduce ptx cache spam in https://github.com/Genesis-Embodied-AI/gstaichi/pull/156


**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v2.0.0...v2.0.1

---

## [v2.0.0](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v2.0.0)

| Field  | Value |
|--------|-------|
| **Tag** | `v2.0.0` |
| **Date** | 2025-08-22 |
| **Commit** | `main` |

# Release 2.0.0

Py dataclasses can now be nested. We added faster cache load time [*1] for kernels annotated with `@ti.pure`, and running on CUDA. We removed paddle and argpack.

Note that we are using [semver.org](http://semver.org/), and since we are removing things, which is a backwards-incompatible change, hence the major version bump.

[*1] Concretely, on [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator, running on a Ubuntu 24.04 box, with an NVidia 5090 GPU, kernel cache load time for single_franka_envs.py has changed as follows:
- baseline: 7.2s
- with SRC-LL cache added: 2.9s
- with PTX cache added: 4.6s
- with both SRC-LL cache and PTX cache added: 0.3s

## HIghlights:
### Type
* [Type] Add nested py dataclasses and enable .shape for members in https://github.com/Genesis-Embodied-AI/gstaichi/pull/91
### Perf
* [Perf] Add SRC-LL caching to accelerate cache load time in https://github.com/Genesis-Embodied-AI/gstaichi/pull/131
* [Perf] Add ptx caching to accelerate cache load time in https://github.com/Genesis-Embodied-AI/gstaichi/pull/130
https://github.com/Genesis-Embodied-AI/gstaichi/pull/131
### Misc
* [Misc] Remove Paddle, argpack in #132 #127

## What's Changed
### Vulkan
* [vulkan] Fix test_print on Vulkan in https://github.com/Genesis-Embodied-AI/gstaichi/pull/118
### Type
* [Type] Add nested py dataclasses and enable .shape for members in https://github.com/Genesis-Embodied-AI/gstaichi/pull/91
* [type] Fix pyright warnings in https://github.com/Genesis-Embodied-AI/gstaichi/pull/136
### Perf
* [Perf] Add SRC-LL caching to accelerate warm cache load time in
* [Perf] Add ptx caching in https://github.com/Genesis-Embodied-AI/gstaichi/pull/130
https://github.com/Genesis-Embodied-AI/gstaichi/pull/131
### Misc
* [Misc] Remove paddle in https://github.com/Genesis-Embodied-AI/gstaichi/pull/132
* [Misc] Remove argpack in https://github.com/Genesis-Embodied-AI/gstaichi/pull/127
* [Misc] Fix bug with np.bool in args hasher and construct path for debugging in https://github.com/Genesis-Embodied-AI/gstaichi/pull/138
### Test
* [Test] Rename test_py_dataclass.py, and change ti.template() to ti.Template in https://github.com/Genesis-Embodied-AI/gstaichi/pull/135
* [Test] Use temporary_module to avoid leaving modules behind in namespace in https://github.com/Genesis-Embodied-AI/gstaichi/pull/146
### Build
* [Build] Enable building cpp tests in CI in https://github.com/Genesis-Embodied-AI/gstaichi/pull/120
* [Build] Add gpu runner in https://github.com/Genesis-Embodied-AI/gstaichi/pull/133
* [Build] Run c++ tests on gpu ci runner in https://github.com/Genesis-Embodied-AI/gstaichi/pull/139

**Full Changelog**: https://github.com/Genesis-Embodied-AI/gstaichi/compare/v1.0.1...v2.0.0

---

## [v1.0.1](https://github.com/Genesis-Embodied-AI/quadrants/releases/tag/v1.0.1)

| Field  | Value |
|--------|-------|
| **Tag** | `v1.0.1` |
| **Date** | 2025-08-07 |
| **Commit** | `main` |

# 1.0.1 release notes

This initial release is mostly to ensure that our own CI build system is working and can publish wheels correctly. We provide support for passing heterogeneous  python dataclasses into kernels and sub-functions., We made some initial typing improvements. We removed functionality we won't be using.

All contributions in this release are from Genesis team (@hughperkins).

## Highlights:
### Misc
   - Remove C API, AOT, DX11, DX12, Android, IOS, OpenGL, GLES, UI, CLI (#123, #115, #27, #124)
   - Increase max kernel args to 512 (#114)
   - Add TI_SHOW_COMPILING flag to show when a kernel is compiled (#92)
   - Fixed a broad family of debug mode crashes (#8705)  (technically part of upstream 1.7.4, since we contributed this to upstream)
### Type system
   - Add fields to python dataclasses support (#76)
   - Add python dataclasses support (#73)
   - Allow ti.types.NDArray with square brackets as type (#42)
   - Allow ti.template and ti.Template for typing (#41)
   - Allow none return typing, for kernels and funcs (#18)
### Build
   - Add pyi stubs to wheel build (#49)

## Full changelog:

### Misc
   - [Misc] Remove CLI (#124)
   - [Misc] Remove shaders (#126)
   - [Misc] Remove C api, AOT, DX11, DX12, Android, IOS (#123)
   - [Misc] Increase max kernel args to 512 (#114)
   - [Misc] Remove opengl and gles (#115)
   - [Misc] Add TI_SHOW_COMPILING (#92)
   - [Misc] Remove ui, and related doc and examples (#27)
   - [Misc] Migrate to TI_DUMP_IR and TI_LOAD_IR and compare with 1 (#53)
   - [Misc] Add TI_DUMP_AST (#52)
   - [Misc] Dump struct uses file sequencer (#40)
   - [Misc] Add TAICHI_DUMP_IR and TAICHI_LOAD_IR (#15)

### Build
   - [Build] Publish to pypi (#113)
   - [Build] Migrate from environment to repo secrets (#106)
   - [Build] Merge from upstream (#112)
   - [Build] reduce concurrency, and only test python 3.10 (#111)
   - [Build] Update git config --global --add safe.directory for gs-taichi (#109)
   - [Build] Update gitignore for whl, so, stubs, CHANGELOG.md (#102)
   - [Build] Try to fix manylinux build issue (#105)
   - [Build] Migrate pyright to run from manylinux (#80)
   - [Build] Add gc before each unit test, to prevent ndarray issues (#74)
   - [Build] Fix broken builds, hopefully (#77)
   - [Build] Reduce concurrency (#75)
   - [Build] Run on merge to main (#70)
   - [Build] Remove a remaining .github old script (#72)
   - [Build] Add stub postprocessing (#57)
   - [Build] Remove the old build scripts ,that we arent using currently (#36)
   - [Build] Win build on matrix, and upload to s3 (#51)
   - [Build] re-add scikit-build (#55)
   - [Build] Build and publish api doc (#46)
   - [Build] Add pyi stubs to wheel build (#49)
   - [Build] Grid of macos python versions and upload to s3 (#50)
   - [Build] Explicitly start sccache (#45)
   - [Build] Manylinux wheel (#37)
   - [Build] Remove conditional guard on -Wno-unused-but-set-variable (#48)
   - [Build] Make apple definitions conditional on apple (#47)
   - [Build] Enable pyright on all files (that dont have `# type: ignore` at the top) (#44)
   - [Build] Add codeowners (#38)
   - [Build] Add mac build for Mac OS 14 and 15 (#5)
   - [Build] Enable ruff check (#32)
   - [Build] Increase sccache timeout (#35)
   - [Build] Bulk mark # type: ignore (#31)
   - [Build] Add name to windows 2025 build, for status check registration (#28)
   - [Build] Fix build issue with filesystem (#29)
   - [Build] Run ruff check --select I --fix (#20)
   - [Build] Fix status check names (#21)
   - [Build] Blanket 3 retries (#26)
   - [Build] Add Windows github runner (#4)
   - [Build] Add linters (#6)
   - [Build] Fix tools test on wheel (#17)
   - [Build] Add clang-tidy linter, and fix lint errors (#11)
   - [Build] Improve link checker (#9)
   - [Build] Linux x86 runner (#3)

### Type
   - [Type] Factorization for nested structs (#99)
   - [Type] Update _graph.py to use GraphBuilderCxx (#119)
   - [Type] Typing 5: Rename imported cpp classes to end in Cxx (#69)
   - [Type] Typing batch 4, including handling kernel/func/real_func wrapper (#67)
   - [Type] Add fields to dataclasses.dataclass support (#76)
   - [Type] Fix ti.Template on ti.funcs (#94)
   - [Type] Typing additions batch 2 c (#61)
   - [Type] Add ndarray struct (#73)
   - [Type] Bunch of typing added for kernel_impl.py, impl.py, and related (#60)
   - [Type] Add lots of typing to ast_transformer.py (#54)
   - [Type] Allow ti.types.NDArray with square brackets as type (#42)
   - [Type] Allow ti.template and ti.Template for typing (#41)
   - [Type] Allow none return typing, for kernels and funcs (#18)

### Test
   - [Test] Add a test to cover calling class method (#96)
   - [Test] test_api sorts the names (needed for renaming taichi => gstaichi) (#108)
   - [Test] Unit tests print full traceback on exception (#101)
   - [Test] Make test_print no longer break if stdout output (#68)
   - [Test] Don't use print to test quant (#62)
   - [Test] Fold py38_only.py into appropriate other test scripts (#39)

### Doc
   - [Doc] Update issue template for gstaichi (#121)
   - [Doc] The readme is ready to be made publicly visible (#82)
   - [Doc] Migrate to sphinx (#90)
   - [Doc] Nuke doc and examples (#88)
   - [Doc] Migrate docs links to point into repo, rather than to docs server (#33)
   - [Doc] Remove rfcs (#34)
   - [Doc] Check markup links (#7)

### Cuda
   - [Cuda] Move implementation from jit_cuda.h into jit_cuda.cpp (#23)

### Mac
   - [Mac] Fix metal device build (#12)

### Rhi
   - [Rhi] [bug] Fix the Unified Allocator to no longer return first two allocations as dupes (#8705)  (technically part of 1.7.4, since we contributed this to upstream)

### Vulkan
   - [Vulkan] Fix exception for max ndarray args test on mac vulkan (#122)

---
