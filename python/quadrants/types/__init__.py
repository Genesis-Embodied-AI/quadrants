"""
This module defines data types in Quadrants:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- quant: for quantized types, see "https://yuanming.quadrants.graphics/publication/2021-quanquadrants/quanquadrants.pdf"
"""

# ruff: noqa: I001  -- the import order in this file is load-bearing (see comment on UnpackedVector below).

from quadrants.types import quant
from quadrants.types.annotations import *  # type: ignore
from quadrants.types.buffer_view_type import *  # type: ignore
from quadrants.types.compound_types import *  # type: ignore
from quadrants.types.ndarray_type import *  # type: ignore
from quadrants.types.primitive_types import *  # type: ignore
from quadrants.types.utils import *  # type: ignore

# Must stay at the bottom: importing ``quadrants.lang.unpacked_vector`` indirectly pulls in ``quadrants.lang.util``,
# which imports ``Template`` from this module. Doing the import after the rest of the module has loaded ensures
# ``Template`` is already defined when that re-entry happens.
from quadrants.lang.unpacked_vector import UnpackedVector  # type: ignore  # noqa: E402

__all__ = ["quant", "UnpackedVector"]
