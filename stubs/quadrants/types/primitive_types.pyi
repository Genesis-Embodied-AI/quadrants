from typing import Any, ClassVar, Union

from quadrants._lib.core.quadrants_python import DataTypeCxx

class PrimitiveMeta(type):
    cxx: DataTypeCxx
    def __eq__(cls, other: object) -> bool: ...
    def __ne__(cls, other: object) -> bool: ...
    def __hash__(cls) -> int: ...
    def __repr__(cls) -> str: ...
    def __getattr__(cls, name: str) -> Any: ...

class PrimitiveBase(metaclass=PrimitiveMeta):
    cxx: ClassVar[DataTypeCxx]

class f16(PrimitiveBase): ...
class f32(PrimitiveBase): ...
class f64(PrimitiveBase): ...
class i8(PrimitiveBase): ...
class i16(PrimitiveBase): ...
class i32(PrimitiveBase): ...
class i64(PrimitiveBase): ...
class u1(PrimitiveBase): ...
class u8(PrimitiveBase): ...
class u16(PrimitiveBase): ...
class u32(PrimitiveBase): ...
class u64(PrimitiveBase): ...

float16 = f16
float32 = f32
float64 = f64
int8 = i8
int16 = i16
int32 = i32
int64 = i64
uint1 = u1
uint8 = u8
uint16 = u16
uint32 = u32
uint64 = u64

# Raw C++ DataType instances (internal use)
f16_cxx: DataTypeCxx
f32_cxx: DataTypeCxx
f64_cxx: DataTypeCxx
i8_cxx: DataTypeCxx
i16_cxx: DataTypeCxx
i32_cxx: DataTypeCxx
i64_cxx: DataTypeCxx
u1_cxx: DataTypeCxx
u8_cxx: DataTypeCxx
u16_cxx: DataTypeCxx
u32_cxx: DataTypeCxx
u64_cxx: DataTypeCxx

class RefType:
    tp: Any
    def __init__(self, tp: Any) -> None: ...

def ref(tp: Any) -> RefType: ...

real_types: set[type[PrimitiveBase] | type]
real_type_ids: set[int]
integer_types: set[type[PrimitiveBase] | type]
integer_type_ids: set[int]
all_types: set[type[PrimitiveBase] | type]
cxx_type_ids: set[int]
type_ids: set[int]
_python_primitive_types = Union[int, float, bool, str, None]
