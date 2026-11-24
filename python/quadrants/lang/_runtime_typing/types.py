"""Protocols and type aliases shared by runtime dataclass processing."""

import dataclasses
from typing import Any, ClassVar, Protocol, TypeAlias

from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants.lang.matrix import MatrixType
from quadrants.lang.struct import StructType
from quadrants.types import buffer_view_type, ndarray_type


class DataclassInstance(Protocol):
    """Describe the field metadata shared by Python dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


class RuntimeDataclassSchema(DataclassInstance, Protocol):
    """Describe generated schema metadata inferred from an arriving dataclass instance."""

    # Original dataclass class described by this generated schema.
    _qd_underlying_dataclass: ClassVar[type[DataclassInstance]]
    # Inferred member types can be reused without guards; frozen=True on the original class alone is insufficient.
    _qd_frozen: ClassVar[bool]
    # This schema or a nested schema contains errors raised only if compilation uses the affected members.
    _qd_has_deferred_errors: ClassVar[bool]
    # Original class identity and ordered member type signatures, used to select a compiled kernel.
    _qd_type_signature: ClassVar[tuple[int, tuple[tuple[str, Any], ...]]]


# Object identity maps to the retained object and its type-tagged Final specialization key.
FinalValueKeyCache: TypeAlias = dict[int, tuple[Any, tuple[Any, ...]]]

InferredMemberType: TypeAlias = (
    type | DataTypeCxx | ndarray_type.NdarrayType | buffer_view_type.BufferViewType | MatrixType | StructType
)

# Class identity, numeric-default identities, and mutable-ancestor context.
InferenceContextKey: TypeAlias = tuple[int, int, int, str | None]
