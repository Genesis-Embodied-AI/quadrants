"""Zero-copy tensor interop between Quadrants and PyTorch/NumPy via DLPack.

Provides cached, zero-copy conversion of Fields and Ndarrays to PyTorch tensors using the DLPack protocol.
The returned tensors directly alias the Quadrants device memory -- modifications to either side are visible
in the other (after ``qd.sync()`` on Apple Metal).
"""

from __future__ import annotations

import functools

from quadrants._lib import core as _qd_core
from quadrants.lang import impl
from quadrants.types import primitive_types

_ARCH_METAL = _qd_core.Arch.metal
_ARCH_VULKAN = _qd_core.Arch.vulkan

_DLPACK_SUPPORTED_DTYPES = frozenset(
    {primitive_types.i32, primitive_types.i64, primitive_types.f32, primitive_types.f64, primitive_types.u1}
)


@functools.lru_cache(maxsize=1)
def _torch_mps_supports_dlpack_bytes_offset() -> bool:
    """``True`` when the installed PyTorch supports DLPack ``bytes_offset`` on MPS.

    Required for zero-copy export of Fields on Apple Metal.
    Available since torch > 2.9.1 (see pytorch/pytorch#168193).
    """
    try:
        import torch  # pylint: disable=C0415

        parts = torch.__version__.replace("+", ".").split(".")[:3]
        return tuple(map(int, parts)) > (2, 9, 1)
    except (ImportError, ValueError):
        return False


def can_zerocopy(is_field: bool, dtype=None, is_scalar_field: bool = False, shape: tuple[int, ...] = ()) -> bool:
    """Check whether zero-copy DLPack export is available for the current backend and data type.

    Args:
        is_field: ``True`` for SNode-backed Fields, ``False`` for Ndarrays.
        dtype: The Quadrants dtype (e.g. ``qd.f32``).  Types not in the C++ DLPack whitelist (f16, u8, …) return False.
        is_scalar_field: ``True`` when the source is a ``ScalarField`` (0-dim DLPack edge-case).
        shape: Batch shape of the field/ndarray.

    Returns:
        ``True`` if zero-copy via DLPack is supported.
    """
    if dtype is not None and dtype not in _DLPACK_SUPPORTED_DTYPES:
        return False
    arch = impl.current_cfg().arch
    if arch == _ARCH_VULKAN:
        return False
    if is_field:
        if arch == _ARCH_METAL and not _torch_mps_supports_dlpack_bytes_offset():
            return False
        # 0-dim ScalarFields lack DLPack bytes_offset support in current PyTorch
        if is_scalar_field and not shape:
            return False
    return True


def dlpack_to_torch(obj):
    """Return a cached zero-copy ``torch.Tensor`` view of a Field or Ndarray via DLPack.

    On the first call the DLPack capsule is created and wrapped into a ``torch.Tensor`` which is then stored on the
    source object as ``_qd_dlpack_tc``.  Subsequent calls return the cached tensor directly (O(1)).

    On Apple Metal an explicit ``sync()`` is performed after the initial export to ensure data visibility.
    """
    try:
        return obj._qd_dlpack_tc
    except AttributeError:
        import torch  # pylint: disable=C0415

        tc = torch.from_dlpack(obj.to_dlpack())
        obj._qd_dlpack_tc = tc
        if impl.current_cfg().arch == _ARCH_METAL:
            impl.get_runtime().sync()
        return tc


def invalidate_zerocopy_cache(obj) -> None:
    """Remove the cached DLPack torch tensor, if any."""
    try:
        del obj._qd_dlpack_tc
    except AttributeError:
        pass
