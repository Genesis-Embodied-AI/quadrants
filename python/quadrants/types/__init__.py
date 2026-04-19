"""
This module defines data types in Quadrants:

- primitive: int, float, etc.
- compound: matrix, vector, struct.
- template: for reference types.
- ndarray: for arbitrary arrays.
- tensor: layout-aware ndarray (logical-to-physical permutation).
- quant: for quantized types, see "https://yuanming.quadrants.graphics/publication/2021-quanquadrants/quanquadrants.pdf"
"""

from quadrants.types import quant
from quadrants.types.annotations import *  # type: ignore
from quadrants.types.compound_types import *  # type: ignore
from quadrants.types.ndarray_type import *  # type: ignore
from quadrants.types.primitive_types import *  # type: ignore
from quadrants.types.utils import *  # type: ignore

# isort: split
# Layout-aware tensor: re-export the Tensor class as the type-annotation alias
# for kernel arguments. Mirrors the qd.types.ndarray naming convention. The
# underlying machinery is the dataclass kernel-arg bridge — see
# quadrants.lang._tensor for details.
#
# This import deliberately lives *after* the type re-exports above, because
# quadrants.lang depends on quadrants.types and we would otherwise create a
# circular import.
from quadrants.lang._tensor import Tensor as tensor  # noqa: F401

__all__ = ["quant", "tensor"]
