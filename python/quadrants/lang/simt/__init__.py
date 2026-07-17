from typing import TYPE_CHECKING

from quadrants.lang.simt import block, grid, subgroup, warp

if TYPE_CHECKING:
    from quadrants.lang.simt._tile import Tile16x16Proxy as Tile16x16
    from quadrants.lang.simt._tile import Tile32x32Proxy as Tile32x32

__all__ = ["warp", "subgroup", "block", "grid", "Tile16x16", "Tile32x32"]


def __getattr__(name):
    if name in ("Tile16x16", "Tile32x32"):
        from quadrants.lang.simt._tile import (  # pylint: disable=import-outside-toplevel
            Tile16x16Proxy,
            Tile32x32Proxy,
        )

        proxy = Tile16x16Proxy if name == "Tile16x16" else Tile32x32Proxy
        globals()[name] = proxy
        return proxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
