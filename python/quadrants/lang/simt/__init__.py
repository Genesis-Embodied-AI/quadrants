from typing import TYPE_CHECKING

from quadrants.lang.simt import block, grid, subgroup, warp

if TYPE_CHECKING:
    from quadrants.lang.simt._tile16 import Tile16x16Proxy as Tile16x16

__all__ = ["warp", "subgroup", "block", "grid", "Tile16x16"]


def __getattr__(name):
    if name == "Tile16x16":
        from quadrants.lang.simt._tile16 import (  # pylint: disable=import-outside-toplevel
            Tile16x16Proxy,
        )

        globals()["Tile16x16"] = Tile16x16Proxy
        return Tile16x16Proxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
