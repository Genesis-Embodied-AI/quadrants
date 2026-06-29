"""Tile slice-syntax dispatch for impl.subscript().

Intercepts array subscripts during compilation and returns deferred proxy objects
(_TileSliceProxy, _VecSliceProxy, _TileRefProxy) that execute tile load/store
when consumed by an assignment. Works for any registered tile size (currently 16 and 32).
"""

from quadrants.lang.exception import QuadrantsSyntaxError
from quadrants.lang.simt._tile16 import (
    _tile16_cache,
    _TileRefProxy,
    _TileSliceProxy,
    _VecSliceProxy,
)
from quadrants.lang.simt._tile32 import _tile32_cache
from quadrants.lang.struct import Struct


def _is_tile(value):
    """Return True if ``value`` is an instance of any registered tile dataclass."""
    return any(isinstance(value, t) for t in _tile16_cache.values()) or any(
        isinstance(value, t) for t in _tile32_cache.values()
    )


def _any_tile_built():
    """Return True if any tile dataclass has been built (i.e. at least one tile is in use)."""
    return bool(_tile16_cache) or bool(_tile32_cache)


def try_tile_ref(value, _indices):
    """Handle tile[:] → _TileRefProxy.

    Returns (True, proxy) if value is a tile struct subscripted with [:], otherwise (False, None).
    """
    if len(_indices) != 1 or not isinstance(_indices[0], slice) or _indices[0] != slice(None):
        return False, None

    if not isinstance(value, Struct):
        return False, None

    if _is_tile(value):
        return True, _TileRefProxy(value)
    return False, None


def _check_slice(s, name):
    if s.start is None or s.stop is None:
        raise QuadrantsSyntaxError(f"Tile {name} slice: both start and stop indices are required")


def try_tile_slice(value, indices):
    """Handle arr[r:r2, c:c2] and variants → _TileSliceProxy / _VecSliceProxy.

    Returns (True, proxy) if the subscript matches a tile slice pattern, otherwise (False, None).
    Raises QuadrantsSyntaxError if tile types exist but the pattern is invalid.
    """
    if not _any_tile_built():
        return False, None

    is_slice = [isinstance(i, slice) for i in indices]

    # arr[r:r2, c:c2]
    if is_slice == [True, True]:
        _check_slice(indices[0], "row")
        _check_slice(indices[1], "col")
        return True, _TileSliceProxy(value, indices[0].start, indices[0].stop, indices[1].start, indices[1].stop)

    # arr[batch, r:r2, c:c2]
    if is_slice == [False, True, True]:
        _check_slice(indices[1], "row")
        _check_slice(indices[2], "col")
        return True, _TileSliceProxy(
            value, indices[1].start, indices[1].stop, indices[2].start, indices[2].stop, indices[0]
        )

    # arr[r:r2, col]
    if is_slice == [True, False]:
        _check_slice(indices[0], "row")
        return True, _VecSliceProxy(value, indices[0].start, indices[0].stop, indices[1])

    # arr[batch, r:r2, col]
    if is_slice == [False, True, False]:
        _check_slice(indices[1], "row")
        return True, _VecSliceProxy(value, indices[1].start, indices[1].stop, indices[2], indices[0])

    return False, None
