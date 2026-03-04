from quadrants._lib import core as qd_python_core
from quadrants.types.primitive_types import PrimitiveBase

_is_signed_cxx = qd_python_core.is_signed
_is_integral_cxx = qd_python_core.is_integral
_is_real_cxx = qd_python_core.is_real
_is_tensor_cxx = qd_python_core.is_tensor


def _cook_if_needed(dt):
    if isinstance(dt, type) and issubclass(dt, PrimitiveBase):
        return dt.cxx
    return dt


def is_signed(dt):
    return _is_signed_cxx(_cook_if_needed(dt))


def is_integral(dt):
    return _is_integral_cxx(_cook_if_needed(dt))


def is_real(dt):
    return _is_real_cxx(_cook_if_needed(dt))


def is_tensor(dt):
    return _is_tensor_cxx(_cook_if_needed(dt))


__all__ = ["is_signed", "is_integral", "is_real", "is_tensor"]
