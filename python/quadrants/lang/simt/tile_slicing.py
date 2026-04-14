"""Tile16x16 slice-syntax dispatch for impl.subscript().

Intercepts array subscripts during compilation and returns deferred proxy objects
(_TileSliceProxy, _VecSliceProxy, _TileRefProxy) that execute tile load/store
when consumed by an assignment.
"""

from quadrants.lang.exception import QuadrantsSyntaxError


def try_tile_ref(value, _indices):
    """Handle tile[:] → _TileRefProxy.

    Returns (True, proxy) if value is a Tile16x16 struct subscripted with [:], otherwise (False, None).
    """
    if len(_indices) != 1 or not isinstance(_indices[0], slice) or _indices[0] != slice(None):
        return False, None

    from quadrants.lang.struct import Struct  # pylint: disable=import-outside-toplevel  # noqa: I001

    if not isinstance(value, Struct):
        return False, None

    # pylint: disable-next=import-outside-toplevel
    from quadrants.lang.simt._tile16 import _tile16_cache, _TileRefProxy

    if any(isinstance(value, t) for t in _tile16_cache.values()):
        return True, _TileRefProxy(value)
    return False, None


def try_tile_slice(value, indices):
    """Handle arr[r:r2, c:c2] and variants → _TileSliceProxy / _VecSliceProxy.

    Returns (True, proxy) if the subscript matches a tile slice pattern, otherwise (False, None).
    Raises QuadrantsSyntaxError if tile types exist but the pattern is invalid.
    """
    from quadrants.lang.simt._tile16 import _tile16_cache  # pylint: disable=import-outside-toplevel  # noqa: I001

    if not _tile16_cache:
        return False, None

    def _check_slice(s, name):
        if s.start is None or s.stop is None:
            raise QuadrantsSyntaxError(f"Tile16x16 {name} slice: both start and stop indices are required")

    is_slice = [isinstance(i, slice) for i in indices]

    # arr[r:r2, c:c2]
    if is_slice == [True, True]:
        from quadrants.lang.simt._tile16 import _TileSliceProxy  # pylint: disable=import-outside-toplevel  # noqa: I001

        _check_slice(indices[0], "row")
        _check_slice(indices[1], "col")
        return True, _TileSliceProxy(value, indices[0].start, indices[0].stop, indices[1].start, indices[1].stop)

    # arr[batch, r:r2, c:c2]
    if is_slice == [False, True, True]:
        from quadrants.lang.simt._tile16 import _TileSliceProxy  # pylint: disable=import-outside-toplevel  # noqa: I001

        _check_slice(indices[1], "row")
        _check_slice(indices[2], "col")
        return True, _TileSliceProxy(
            value, indices[1].start, indices[1].stop, indices[2].start, indices[2].stop, indices[0]
        )

    # arr[r:r2, col]
    if is_slice == [True, False]:
        from quadrants.lang.simt._tile16 import _VecSliceProxy  # pylint: disable=import-outside-toplevel  # noqa: I001

        _check_slice(indices[0], "row")
        return True, _VecSliceProxy(value, indices[0].start, indices[0].stop, indices[1])

    # arr[batch, r:r2, col]
    if is_slice == [False, True, False]:
        from quadrants.lang.simt._tile16 import _VecSliceProxy  # pylint: disable=import-outside-toplevel  # noqa: I001

        _check_slice(indices[1], "row")
        return True, _VecSliceProxy(value, indices[1].start, indices[1].stop, indices[2], indices[0])

    return False, None
