# type: ignore

import warnings

from quadrants._lib import core as _qd_core
from quadrants.lang import impl


def arch_uses_spv(arch):
    return arch == _qd_core.vulkan or arch == _qd_core.metal


def mem_fence():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("grid_mem_fence", with_runtime_context=False)
    if arch_uses_spv(arch):
        # On Vulkan: `OpMemoryBarrier(ScopeDevice, ...)`. On Metal (via MoltenVK + SPIRV-Cross -> MSL) this is
        # translated to `atomic_thread_fence(metal::memory_scope_device)` on Apple Silicon / macOS 10.13+. Cross-
        # workgroup ordering guarantees on older Apple hardware and very old macOS Intel GPUs are weaker than on CUDA
        # -- see `block.md` for the support note.
        return impl.call_internal("gridMemoryBarrier", with_runtime_context=False)
    raise ValueError(f"qd.simt.grid.mem_fence is not supported for arch {arch}")


def memfence():
    warnings.warn(
        "qd.simt.grid.memfence() is deprecated; use qd.simt.grid.mem_fence() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mem_fence()
