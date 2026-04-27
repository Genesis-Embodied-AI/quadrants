# qd.init options

`qd.init(...)` accepts every field of the underlying `CompileConfig` struct as a keyword argument; the same fields are also reachable as environment variables of the form `QD_<UPPERCASE_NAME>` (e.g. `QD_OFFLINE_CACHE=0`). This page collects the knobs Genesis or downstream users reach for in practice. The underlying source of truth is `quadrants/program/compile_config.h`.

## Caching

### `offline_cache`

Whether to read and write the on-disk kernel cache. Default `True`.

`offline_cache=False` (or `QD_OFFLINE_CACHE=0`) takes effect at two layers on CUDA so successive runs really do recompile from scratch:

1. The Quadrants kernel-IR cache and the per-arch PTX cache under `~/.cache/quadrants/.../ptx_cache_sm_*` are not consulted and not written.
2. The NVIDIA driver compute cache at `~/.nv/ComputeCache` is keyed by PTX content hash. Because `CUDA_CACHE_DISABLE` is captured by libcuda at process start (so toggling it from inside Python has no effect), Quadrants instead appends a per-process nonce comment to the PTX it submits to `cuModuleLoadDataEx`. The nonce is constant within one process - kernels with identical PTX still share a cubin in the same run - and changes between processes so cross-run hits cannot quietly serve stale SASS.

When to use it:
- Taking compile-time profiles where any cached SASS would mask the real cost.
- Investigating a stale-cache bug or suspected cache corruption.

For normal use, leave it at `True`; the cache is the dominant source of fast warm-up.
