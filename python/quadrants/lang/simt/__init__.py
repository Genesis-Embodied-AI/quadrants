# type: ignore

from quadrants.lang.simt import block, grid, subgroup, warp

__all__ = ["warp", "subgroup", "block", "grid", "tile16"]


def __getattr__(name):
    if name == "tile16":
        from quadrants.lang.simt import tile16
        return tile16
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
