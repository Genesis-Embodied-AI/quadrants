# type: ignore

from quadrants.lang.simt import block, grid, subgroup, warp

__all__ = ["warp", "subgroup", "block", "grid", "tile16", "Tile16x16"]


def __getattr__(name):
    if name == "tile16":
        from quadrants.lang.simt import tile16  # noqa: I001  # pylint: disable=import-outside-toplevel

        return tile16
    if name == "Tile16x16":
        from quadrants.lang.simt.tile16 import Tile16x16Proxy  # pylint: disable=import-outside-toplevel

        globals()["Tile16x16"] = Tile16x16Proxy
        return Tile16x16Proxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
