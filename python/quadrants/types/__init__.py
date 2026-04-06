"""
This module defines data types in Quadrants:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- quant: for quantized types, see "https://yuanming.quadrants.graphics/publication/2021-quanquadrants/quanquadrants.pdf"
"""

from quadrants.types import quant
from quadrants.types.annotations import *  # type: ignore
from quadrants.types.compound_types import *  # type: ignore
from quadrants.types.ndarray_type import *  # type: ignore
from quadrants.types.primitive_types import *  # type: ignore
from quadrants.types.utils import *  # type: ignore

__all__ = ["Tile16x16", "quant"]


def __getattr__(name):
    if name == "Tile16x16":
        from quadrants.lang.simt.tile16 import make_tile16x16  # pylint: disable=import-outside-toplevel

        globals()["Tile16x16"] = make_tile16x16
        return make_tile16x16
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
