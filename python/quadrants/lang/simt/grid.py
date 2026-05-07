# type: ignore

import warnings

from quadrants.lang import impl


def mem_fence():
    return impl.call_internal("grid_memfence", with_runtime_context=False)


def memfence():
    warnings.warn(
        "qd.simt.grid.memfence() is deprecated; use qd.simt.grid.mem_fence() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mem_fence()
