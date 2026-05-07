"""Metal interop synchronisation helpers.

When Quadrants and PyTorch MPS use separate Metal command queues, explicit CPU-side syncs (``qd.sync()`` and
``torch.mps.synchronize()``) are needed at every interop point to guarantee data visibility.  When both frameworks share
a single command queue (via ``external_metal_command_queue`` + ``external_metal_command_queue_is_torch_queue``), Metal's
sequential command buffer ordering makes these syncs unnecessary.

This module caches the "do we need interop sync?" decision and exposes it to ``field.py`` and other consumers.  The
cache is lazily populated on first access and invalidated on ``qd.reset()`` via the ``impl.on_reset`` hook.
"""

from __future__ import annotations

from quadrants._lib.core import quadrants_python as _qd_core
from quadrants.lang import impl

_ARCH_METAL = _qd_core.Arch.metal

_metal_needs_interop_sync_cached: bool | None = None


def _recompute_metal_interop_sync() -> None:
    """Recompute and cache the Metal interop sync flag from the current config."""
    global _metal_needs_interop_sync_cached
    cfg = impl.current_cfg()
    _metal_needs_interop_sync_cached = cfg.arch == _ARCH_METAL and not (
        cfg.external_metal_command_queue and cfg.external_metal_command_queue_is_torch_queue
    )


def _clear_metal_interop_cache() -> None:
    """Invalidate the cached flag.  Registered as a reset hook."""
    global _metal_needs_interop_sync_cached
    _metal_needs_interop_sync_cached = None


_metal_interop_hook_registered = False


def metal_needs_interop_sync() -> bool:
    """Return True when explicit sync is needed between Quadrants and PyTorch MPS (separate Metal queues)."""
    global _metal_interop_hook_registered
    if not _metal_interop_hook_registered:
        impl.on_reset(_clear_metal_interop_cache)
        _metal_interop_hook_registered = True
    if _metal_needs_interop_sync_cached is None:
        _recompute_metal_interop_sync()
    return _metal_needs_interop_sync_cached  # type: ignore[return-value]


def mps_sync_if_metal() -> None:
    """Call ``torch.mps.synchronize()`` when running on the Metal backend with separate command queues.

    When Quadrants and PyTorch MPS use separate Metal command queues, ``qd.sync()`` only guarantees Quadrants writes are
    complete. A subsequent ``.clone()`` or kernel copy is queued on the MPS stream and may execute *after* the next
    Quadrants kernel overwrites the source buffer. We must also synchronize MPS after the copy.

    When a shared command queue is configured (``external_metal_command_queue != 0``), Metal's sequential command buffer
    semantics guarantee ordering automatically and no sync is needed.
    """
    if metal_needs_interop_sync():
        import torch  # pylint: disable=C0415

        torch.mps.synchronize()
