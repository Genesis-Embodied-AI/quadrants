"""Invalidate runtime dataclass fast paths when a Quadrants Tensor changes its backing object."""

from typing import TYPE_CHECKING
from weakref import WeakSet

if TYPE_CHECKING:
    from quadrants.lang.kernel import Kernel

# Weak references avoid keeping kernels alive solely to invalidate their cached bindings.
_runtime_kernels: WeakSet["Kernel"] = WeakSet()


def invalidate_tensor_bindings() -> None:
    """Drop schema, specialization, and launch caches before a changed Tensor can be reused."""
    # pylint: disable=import-outside-toplevel
    from quadrants.lang.runtime_typing._runtime_dataclass import _schema_by_context_by_instance

    # Rebinding is rare. Clear all tracked bindings instead of maintaining a dependency graph on every launch.
    for ref, _ in tuple(_schema_by_context_by_instance._entries.values()):
        dataclass_instance = ref()
        if dataclass_instance is not None:
            attributes = getattr(dataclass_instance, "__dict__", None)
            if attributes is not None:
                for name in ("_qd_spec_key", "_qd_dc_unwrapped", "_qd_all_field"):
                    attributes.pop(name, None)
    _schema_by_context_by_instance._entries.clear()
    for kernel in tuple(_runtime_kernels):
        kernel._runtime_metadata_args_hash = None
        kernel.mapper._mapping_cache.clear()
        kernel.mapper._mapping_cache_tracker.clear()
        kernel.launch_context_buffer_cache._launch_ctx_cache.clear()
        kernel.launch_context_buffer_cache._launch_ctx_cache_tracker.clear()
