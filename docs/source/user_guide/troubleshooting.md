# Troubleshooting

## In case of crash/seg fault

- run without cache - or clear cache - to see if this resolves the issue
- if running without cache solves the seg fault, then clear the cache

To run without cache:
```python
qd.init(offline_cache=False, ...)
```

On CUDA, `offline_cache=False` (or `QD_OFFLINE_CACHE=0`) takes effect at two layers so successive runs really do recompile from scratch:

1. The Quadrants kernel-IR cache and the per-arch PTX cache under `~/.cache/quadrants/.../ptx_cache_sm_*` are not consulted and not written.
2. The NVIDIA driver compute cache at `~/.nv/ComputeCache` is keyed by PTX content hash. Because `CUDA_CACHE_DISABLE` is captured by libcuda at process start (so toggling it from inside Python has no effect), Quadrants instead appends a per-process nonce comment to the PTX it submits to `cuModuleLoadDataEx`. The nonce is constant within one process - kernels with identical PTX still share a cubin in the same run - and changes between processes so cross-run hits cannot quietly serve stale SASS.

Use this when taking compile-time profiles to make sure the numbers reflect a real cold compile and not a warm cache.

To clear cache:
- the cache is located by default on linux and mac at `~/.cache/quadrants`
- simply remove this entire folder:
```bash
rm -Rf ~/.cache/quadrants
```

If this doesn't solve the problem, then you'll likely need to log a github issue, providing
as much information as possible, and crucially a minimum reproducible example, to reproduce
the seg fault.
