from typing import Any

from quadrants._lib import core as qd_python_core
from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants.types.primitive_types import PrimitiveBase

_is_signed_cxx = qd_python_core.is_signed
_is_integral_cxx = qd_python_core.is_integral
_is_real_cxx = qd_python_core.is_real
_is_tensor_cxx = qd_python_core.is_tensor


def _cook_if_needed(dt: Any) -> DataTypeCxx:
    if isinstance(dt, type) and issubclass(dt, PrimitiveBase):
        return dt.cxx
    return dt  # type: ignore[return-value]


def is_signed(dt: Any) -> bool:
    return _is_signed_cxx(_cook_if_needed(dt))  # type: ignore[arg-type]


def is_integral(dt: Any) -> bool:
    return _is_integral_cxx(_cook_if_needed(dt))  # type: ignore[arg-type]


def is_real(dt: Any) -> bool:
    return _is_real_cxx(_cook_if_needed(dt))  # type: ignore[arg-type]


def is_tensor(dt: Any) -> bool:
    return _is_tensor_cxx(_cook_if_needed(dt))  # type: ignore[arg-type]


__all__ = ["is_signed", "is_integral", "is_real", "is_tensor"]
