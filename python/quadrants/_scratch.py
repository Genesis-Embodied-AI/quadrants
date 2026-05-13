"""Quadrants-level scratch buffer for device-wide algorithms.

A single ``Field(u32)`` shared by every ``qd.algorithms.*`` device kernel.
Algorithms ``qd.bit_cast`` to/from this buffer to support other 4-byte dtypes
(``i32``, ``u32``, ``f32``). Sized to comfortably cover device-wide reduce,
exclusive scan, and select / compact on inputs up to roughly 64M elements per
the design doc at ``perso_hugh/doc/qipc/qipc_device_algos_design.md``.

Sizing rationale: every first-land algorithm needs at most one ``(B,)``-shaped
partials buffer where ``B = ceil(N / BLOCK_DIM)``. For ``BLOCK_DIM = 256`` and
``N = 64M`` that is ``B = 262144`` u32 slots = 1 MB. Two-level recursion adds
``ceil(B / BLOCK_DIM)`` more slots = a couple of KB, which is in the noise.

Allocation strategy: lazy on first use, invalidated on ``qd.reset()`` via the ``impl.on_reset`` hook. This avoids
paying the 1 MB allocation cost in programs that never touch ``qd.algorithms``, and avoids coupling ``qd.init()``'s
argument surface to the device-algos work for the first land. A future change can add ``qd.init(scratch_bytes=...)``
if a caller needs to override the default before any allocation has happened.
"""

from quadrants.lang.impl import field, on_reset
from quadrants.types.primitive_types import u32

DEFAULT_SCRATCH_BYTES: int = 1 << 20

_scratch_field = None
_scratch_bytes: int = DEFAULT_SCRATCH_BYTES


def set_scratch_bytes(scratch_bytes: int) -> None:
    """Set the scratch capacity in bytes for the next allocation.

    Must be called before the first ``get_scratch_u32()`` call in the current runtime cycle. Has no effect on an
    already-allocated scratch field; users wishing to enlarge an existing scratch must ``qd.reset()`` and ``qd.init()``
    again, then re-call ``set_scratch_bytes`` (capacity resets to ``DEFAULT_SCRATCH_BYTES`` on every ``qd.reset()``).
    """
    global _scratch_bytes
    if _scratch_field is not None:
        raise RuntimeError(
            "set_scratch_bytes called after scratch was already allocated; "
            "call before any qd.algorithms.* op runs, or qd.reset() first"
        )
    if scratch_bytes <= 0 or scratch_bytes % 4 != 0:
        raise ValueError(f"scratch_bytes must be a positive multiple of 4; got {scratch_bytes}")
    _scratch_bytes = scratch_bytes


def get_scratch_u32():
    """Return the shared scratch ``Field(u32)``, allocating on first use.

    The field is invalidated automatically by the ``impl.on_reset`` hook
    registered below, so a subsequent call after ``qd.reset()`` will reallocate
    against the fresh runtime.
    """
    global _scratch_field
    if _scratch_field is None:
        _scratch_field = field(u32, shape=_scratch_bytes // 4)
    return _scratch_field


def scratch_capacity_u32() -> int:
    """Return the scratch capacity in u32 slots for the *next* allocation."""
    return _scratch_bytes // 4


def _invalidate() -> None:
    """Drop the cached scratch handle *and* reset the capacity setting back to ``DEFAULT_SCRATCH_BYTES``. Registered as
    an ``impl.on_reset`` hook so every ``qd.reset()`` → ``qd.init()`` transaction is a clean slate: the next
    ``get_scratch_u32()`` call reallocates against the fresh runtime at the default capacity, and any prior
    ``set_scratch_bytes(...)`` bump has to be re-applied before the new runtime's first algorithm call.

    The persistence-vs-clean-slate trade-off was explicitly resolved in favour of clean slate: ``qd.init`` /
    ``qd.reset`` is meant to be "free to use whenever, no constraints", which only holds if all module state tied to a
    runtime cycle (resource handles *and* runtime-scoped config) goes away on reset. Apps that want a persistent bump
    should call ``set_scratch_bytes`` immediately after each ``qd.init``."""
    global _scratch_field, _scratch_bytes
    _scratch_field = None
    _scratch_bytes = DEFAULT_SCRATCH_BYTES


on_reset(_invalidate)
