"""Quadrants-level scratch buffer for device-wide algorithms.

Two scratch fields - one ``Field(u32)`` and one ``Field(u64)`` - shared by every ``qd.algorithms.*`` device kernel.
Algorithms ``qd.bit_cast`` to / from these buffers to support every supported scalar dtype: 4-byte ``i32`` / ``u32``
/ ``f32`` go through the u32 scratch; 8-byte ``i64`` / ``u64`` / ``f64`` go through the u64 scratch. Sized to
comfortably cover device-wide reduce, exclusive scan, select / compact, radix sort, and reduce-by-key on inputs up
to ``N = 1M`` out of the box (qipc's hot path), per the design doc at
``perso_hugh/doc/qipc/qipc_device_algos_design.md``.

Sizing rationale: ``device_select`` / ``device_radix_sort`` need ~``N`` u32 slots per call (one write index /
tile-histogram entry per input element). At ``N = 1M`` that is 4 MB of u32 slots; we round up to 5 MB to leave
headroom for the recursion overhead (``ceil(N / BLOCK_DIM)`` extra slots) and the second-level scan partials.
``device_reduce_*`` / ``device_exclusive_scan_*`` need only ~``N / BLOCK_DIM`` u32 slots, so the same 5 MB
covers them well past ``N = 64M``. The u64 scratch sees half as many slots at the same byte budget.

Allocation strategy: lazy on first use, invalidated on ``qd.reset()`` via the ``impl.on_reset`` hook. This avoids
paying the 5 MB-per-width allocation cost in programs that never touch ``qd.algorithms``, and avoids coupling
``qd.init()``'s argument surface to the device-algos work for the first land. Programs that only touch 4-byte
algorithms never pay for the u64 buffer. A future change can add ``qd.init(scratch_bytes=...)`` if a caller needs
to override the default before any allocation has happened.
"""

from quadrants.lang.impl import field, on_reset
from quadrants.types.primitive_types import u32, u64

DEFAULT_SCRATCH_BYTES: int = 5 * (1 << 20)

_scratch_field = None
_scratch_field_u64 = None
_scratch_bytes: int = DEFAULT_SCRATCH_BYTES


def set_scratch_bytes(scratch_bytes: int) -> None:
    """Set the scratch capacity in bytes for the next allocation.

    Must be called before the first ``get_scratch_u32()`` / ``get_scratch_u64()`` call in the current runtime cycle.
    Has no effect on an already-allocated scratch field; users wishing to enlarge an existing scratch must
    ``qd.reset()`` and ``qd.init()`` again, then re-call ``set_scratch_bytes`` (capacity resets to
    ``DEFAULT_SCRATCH_BYTES`` on every ``qd.reset()``).
    """
    global _scratch_bytes
    if _scratch_field is not None or _scratch_field_u64 is not None:
        raise RuntimeError(
            "set_scratch_bytes called after scratch was already allocated; "
            "call before any qd.algorithms.* op runs, or qd.reset() first"
        )
    if scratch_bytes <= 0 or scratch_bytes % 8 != 0:
        raise ValueError(f"scratch_bytes must be a positive multiple of 8; got {scratch_bytes}")
    _scratch_bytes = scratch_bytes


def get_scratch_u32():
    """Return the shared scratch ``Field(u32)``, allocating on first use.

    The field is invalidated automatically by the ``impl.on_reset`` hook registered below, so a subsequent call
    after ``qd.reset()`` will reallocate against the fresh runtime.
    """
    global _scratch_field
    if _scratch_field is None:
        _scratch_field = field(u32, shape=_scratch_bytes // 4)
    return _scratch_field


def get_scratch_u64():
    """Return the shared scratch ``Field(u64)``, allocating on first use.

    Used by 8-byte algorithms (``i64`` / ``u64`` / ``f64`` reduce + scan, ``u64`` radix-sort keys). Lives alongside
    the u32 scratch rather than overlaying it: a u64 backing aliasing into u32-sized half-cells would require
    dtype-punning fields, which Quadrants doesn't expose. Same byte budget, half as many slots.
    """
    global _scratch_field_u64
    if _scratch_field_u64 is None:
        _scratch_field_u64 = field(u64, shape=_scratch_bytes // 8)
    return _scratch_field_u64


def scratch_capacity_u32() -> int:
    """Return the scratch capacity in u32 slots for the *next* allocation."""
    return _scratch_bytes // 4


def scratch_capacity_u64() -> int:
    """Return the scratch capacity in u64 slots for the *next* allocation."""
    return _scratch_bytes // 8


def _invalidate() -> None:
    """Drop the cached scratch handles *and* reset the capacity setting back to ``DEFAULT_SCRATCH_BYTES``. Registered
    as an ``impl.on_reset`` hook so every ``qd.reset()`` -> ``qd.init()`` transaction is a clean slate: the next
    ``get_scratch_*()`` call reallocates against the fresh runtime at the default capacity, and any prior
    ``set_scratch_bytes(...)`` bump has to be re-applied before the new runtime's first algorithm call.

    The persistence-vs-clean-slate trade-off was explicitly resolved in favour of clean slate: ``qd.init`` /
    ``qd.reset`` is meant to be "free to use whenever, no constraints", which only holds if all module state tied to
    a runtime cycle (resource handles *and* runtime-scoped config) goes away on reset. Apps that want a persistent
    bump (or persistent shrink, for apps that know their N is small and don't want to pay 10 MB across the two
    scratch fields) should call ``set_scratch_bytes`` immediately after each ``qd.init``.
    """
    global _scratch_field, _scratch_field_u64, _scratch_bytes
    _scratch_field = None
    _scratch_field_u64 = None
    _scratch_bytes = DEFAULT_SCRATCH_BYTES


on_reset(_invalidate)
