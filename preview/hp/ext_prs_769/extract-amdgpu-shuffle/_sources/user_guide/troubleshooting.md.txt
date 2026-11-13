# Troubleshooting

## In case of crash/seg fault

- run without cache - or clear cache - to see if this resolves the issue
- if running without cache solves the seg fault, then clear the cache

To run without cache:
```python
qd.init(offline_cache=False, ...)
```

See [qd.init options](init_options.md) for what `offline_cache=False` actually does on CUDA (it bypasses both the Quadrants PtxCache and the NVIDIA driver compute cache).

To clear cache:
- the cache is located by default on linux and mac at `~/.cache/quadrants`
- simply remove this entire folder:
```bash
rm -Rf ~/.cache/quadrants
```

If this doesn't solve the problem, then you'll likely need to log a github issue, providing
as much information as possible, and crucially a minimum reproducible example, to reproduce
the seg fault.

## AMDGPU hangs on first field allocation

Some APUs (notably Rembrandt `gfx90c`) advertise HIP memory-pool support while `hipMallocAsync`
never returns. Quadrants then hangs on the first dense field / SNode allocation after
`qd.init(arch=qd.amdgpu)`.

- On `gfx90c`, HIP memory pools are disabled automatically; allocations use sync `hipMalloc`.
- To force that path on any AMDGPU device (for example another target with a broken async
  allocator, or to A/B the pool path):

```bash
export QD_ENABLE_HIP_MEMPOOL=0
```

Tradeoff: with pools off, device allocations come from the preallocated arena sized by
`device_memory_fraction` / `device_memory_GB` in `qd.init(...)`. Large batched scenes may need a
higher fraction (e.g. `qd.init(..., device_memory_fraction=0.9)`). Kernel speed is unchanged;
only allocation growth behavior differs.
