from typing import ClassVar, Union

from quadrants._lib import core as qd_python_core
from quadrants._lib.core.quadrants_python import DataTypeCxx

# ========================================
# Raw C++ DataType instances (internal use)
# ========================================

f16_cxx = qd_python_core.DataType_f16
f32_cxx = qd_python_core.DataType_f32
f64_cxx = qd_python_core.DataType_f64

i8_cxx = qd_python_core.DataType_i8
i16_cxx = qd_python_core.DataType_i16
i32_cxx = qd_python_core.DataType_i32
i64_cxx = qd_python_core.DataType_i64

u1_cxx = qd_python_core.DataType_u1
u8_cxx = qd_python_core.DataType_u8
u16_cxx = qd_python_core.DataType_u16
u32_cxx = qd_python_core.DataType_u32
u64_cxx = qd_python_core.DataType_u64


# ========================================
# Metaclass and base class for Python dtype wrappers
# ========================================


class PrimitiveMeta(type):
    """Metaclass that makes dtype classes behave like DataTypeCxx objects.

    Delegates attribute access and comparisons to the underlying .cxx object,
    allowing existing code that does e.g. dtype.to_string() to keep working.
    """

    def __eq__(cls, other):
        if isinstance(other, PrimitiveMeta):
            return cls is other
        if isinstance(other, DataTypeCxx):
            return cls.cxx == other
        return NotImplemented

    def __ne__(cls, other):
        if isinstance(other, PrimitiveMeta):
            return cls is not other
        if isinstance(other, DataTypeCxx):
            return cls.cxx != other
        return NotImplemented

    def __hash__(cls):
        return hash(cls.cxx)

    def __repr__(cls):
        return cls.cxx.to_string()

    def __getattr__(cls, name):
        # Delegate unknown attributes to the underlying DataTypeCxx
        try:
            return getattr(cls.cxx, name)
        except AttributeError:
            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'") from None


class PrimitiveBase(metaclass=PrimitiveMeta):
    """Base class for all primitive dtype classes.

    Each subclass has a `cxx` class variable holding the corresponding DataTypeCxx instance.
    Subclasses auto-register themselves in the _registry for reverse lookup (DataTypeCxx → Python class).
    """

    cxx: ClassVar[DataTypeCxx]
    _registry: ClassVar[dict[DataTypeCxx, "type[PrimitiveBase]"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "cxx"):
            PrimitiveBase._registry[cls.cxx] = cls


def cxx_to_py(dtype_cxx: DataTypeCxx) -> "type[PrimitiveBase]":
    """Convert a DataTypeCxx to its corresponding Python dtype class."""
    return PrimitiveBase._registry[dtype_cxx]


# ========================================
# Floating point types
# ========================================


class f16(PrimitiveBase):
    """16-bit precision floating point data type."""

    cxx = f16_cxx


class f32(PrimitiveBase):
    """32-bit single precision floating point data type."""

    cxx = f32_cxx


class f64(PrimitiveBase):
    """64-bit double precision floating point data type."""

    cxx = f64_cxx


float16 = f16
float32 = f32
float64 = f64

# ========================================
# Signed integer types
# ========================================


class i8(PrimitiveBase):
    """8-bit signed integer data type."""

    cxx = i8_cxx


class i16(PrimitiveBase):
    """16-bit signed integer data type."""

    cxx = i16_cxx


class i32(PrimitiveBase):
    """32-bit signed integer data type."""

    cxx = i32_cxx


class i64(PrimitiveBase):
    """64-bit signed integer data type."""

    cxx = i64_cxx


int8 = i8
int16 = i16
int32 = i32
int64 = i64

# ========================================
# Unsigned integer types
# ========================================


class u1(PrimitiveBase):
    """1-bit unsigned integer data type. Same as booleans."""

    cxx = u1_cxx


class u8(PrimitiveBase):
    """8-bit unsigned integer data type."""

    cxx = u8_cxx


class u16(PrimitiveBase):
    """16-bit unsigned integer data type."""

    cxx = u16_cxx


class u32(PrimitiveBase):
    """32-bit unsigned integer data type."""

    cxx = u32_cxx


class u64(PrimitiveBase):
    """64-bit unsigned integer data type."""

    cxx = u64_cxx


uint1 = u1
uint8 = u8
uint16 = u16
uint32 = u32
uint64 = u64

# ========================================
# Ref type (unchanged)
# ========================================


class RefType:
    def __init__(self, tp):
        self.tp = tp


def ref(tp):
    return RefType(tp)


# ========================================
# Type sets for fast lookup
# ========================================

real_types = {f16, f32, f64, float}
real_type_ids = {id(t) for t in real_types}

integer_types = {i8, i16, i32, i64, u1, u8, u16, u32, u64, int, bool}
integer_type_ids = {id(t) for t in integer_types}

all_types = real_types | integer_types
_py_type_ids = {id(t) for t in all_types}

_all_cxx = {f16_cxx, f32_cxx, f64_cxx, i8_cxx, i16_cxx, i32_cxx, i64_cxx, u1_cxx, u8_cxx, u16_cxx, u32_cxx, u64_cxx}
cxx_type_ids = {id(t) for t in _all_cxx}

# Combined set: matches both Python classes and DataTypeCxx instances
type_ids = _py_type_ids | cxx_type_ids

_python_primitive_types = Union[int, float, bool, str, None]

__all__ = [
    "float32",
    "f32",
    "float64",
    "f64",
    "float16",
    "f16",
    "int8",
    "i8",
    "int16",
    "i16",
    "int32",
    "i32",
    "int64",
    "i64",
    "uint1",
    "u1",
    "uint8",
    "u8",
    "uint16",
    "u16",
    "uint32",
    "u32",
    "uint64",
    "u64",
    "ref",
    "_python_primitive_types",
]
