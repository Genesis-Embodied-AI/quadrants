import functools
import os
import traceback
import warnings
from typing import Any

import numpy as np
from colorama import Fore, Style

from quadrants._lib import core as _qd_core
from quadrants._logging import is_logging_effective
from quadrants.lang import impl
from quadrants.types import Template
from quadrants.types.primitive_types import (
    PrimitiveBase,
    all_types,
    f16,
    f16_cxx,
    f32,
    f32_cxx,
    f64,
    f64_cxx,
    i8,
    i8_cxx,
    i16,
    i16_cxx,
    i32,
    i32_cxx,
    i64,
    i64_cxx,
    u1,
    u1_cxx,
    u8,
    u8_cxx,
    u16,
    u16_cxx,
    u32,
    u32_cxx,
    u64,
    u64_cxx,
)

MAP_TYPE_IDS: dict[int, Any] = {id(dtype): dtype for dtype in all_types}
_all_cxx_objs = (
    f16_cxx,
    f32_cxx,
    f64_cxx,
    i8_cxx,
    i16_cxx,
    i32_cxx,
    i64_cxx,
    u1_cxx,
    u8_cxx,
    u16_cxx,
    u32_cxx,
    u64_cxx,
)
for _cxx in _all_cxx_objs:
    MAP_TYPE_IDS[id(_cxx)] = _cxx

# Pre-computed id-based cache for cook_dtype hot path.
# Maps id(Python class) and id(DataTypeCxx) to the DataTypeCxx result.
_cook_cache: dict[int, _qd_core.DataTypeCxx] = {}
for _cls in (f16, f32, f64, i8, i16, i32, i64, u1, u8, u16, u32, u64):
    _cook_cache[id(_cls)] = _cls.cxx
for _cxx in _all_cxx_objs:
    _cook_cache[id(_cxx)] = _cxx


def has_pytorch():
    """Whether has pytorch in the current Python environment.

    Returns:
        bool: True if has pytorch else False.

    """
    _has_pytorch = False
    _env_torch = os.environ.get("QD_ENABLE_TORCH", "1")
    if not _env_torch or int(_env_torch):
        try:
            import torch  # noqa: F401 pylint: disable=C0415

            _has_pytorch = True
        except ImportError:
            pass
    return _has_pytorch


def get_clangpp():
    from distutils.spawn import find_executable  # pylint: disable=C0415

    # Quadrants itself uses llvm-10.0.0 to compile.
    # There will be some issues compiling CUDA with other clang++ version.
    _clangpp_candidates = ["clang++-10"]
    for c in _clangpp_candidates:
        if find_executable(c) is not None:
            _clangpp_presence = find_executable(c)
            return _clangpp_presence
    return None


def has_clangpp():
    return get_clangpp() is not None


def is_matrix_class(rhs):
    matrix_class = False
    try:
        if rhs._is_matrix_class:
            matrix_class = True
    except:
        pass
    return matrix_class


def is_quadrants_class(rhs):
    quadrants_class = False
    try:
        if rhs._is_quadrants_class:
            quadrants_class = True
    except:
        pass
    return quadrants_class


def to_numpy_type(dt):
    """Convert quadrants data type to its counterpart in numpy.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in numpy.

    """
    if dt == f32:
        return np.float32
    if dt == f64:
        return np.float64
    if dt == i32:
        return np.int32
    if dt == i64:
        return np.int64
    if dt == i8:
        return np.int8
    if dt == i16:
        return np.int16
    if dt == u1:
        return np.bool_
    if dt == u8:
        return np.uint8
    if dt == u16:
        return np.uint16
    if dt == u32:
        return np.uint32
    if dt == u64:
        return np.uint64
    if dt == f16:
        return np.half
    assert False


def to_pytorch_type(dt):
    """Convert quadrants data type to its counterpart in torch.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataType: The counterpart data type in torch.

    """
    import torch  # pylint: disable=C0415

    # pylint: disable=E1101
    if dt == f32:
        return torch.float32
    if dt == f64:
        return torch.float64
    if dt == i32:
        return torch.int32
    if dt == i64:
        return torch.int64
    if dt == i8:
        return torch.int8
    if dt == i16:
        return torch.int16
    if dt == u1:
        return torch.bool
    if dt == u8:
        return torch.uint8
    if dt == f16:
        return torch.float16

    if dt in (u16, u32, u64):
        if hasattr(torch, "uint16"):
            if dt == u16:
                return torch.uint16
            if dt == u32:
                return torch.uint32
            if dt == u64:
                return torch.uint64
        raise RuntimeError(f"PyTorch doesn't support {dt.to_string()} data type before version 2.3.0.")

    if dt in {torch.float32, torch.int32, torch.bool}:
        return dt
    raise RuntimeError(f"PyTorch doesn't support {dt.to_string()} data type.")


def to_quadrants_type(dt):
    """Convert primitive type id, numpy or torch data type to its counterpart in quadrants.

    Args:
        dt (DataType): The desired data type to convert.

    Returns:
        DataTypeCxx: The counterpart data type in quadrants (always returns DataTypeCxx).

    """
    _type = type(dt)
    if _type is int:
        return cook_dtype(MAP_TYPE_IDS[dt])

    if isinstance(dt, type) and issubclass(dt, PrimitiveBase):
        return dt.cxx

    if issubclass(_type, _qd_core.DataTypeCxx):
        return dt

    if dt == np.float32:
        return f32.cxx
    if dt == np.float64:
        return f64.cxx
    if dt == np.int32:
        return i32.cxx
    if dt == np.int64:
        return i64.cxx
    if dt == np.int8:
        return i8.cxx
    if dt == np.int16:
        return i16.cxx
    if dt == np.bool_:
        return u1.cxx
    if dt == np.uint8:
        return u8.cxx
    if dt == np.uint16:
        return u16.cxx
    if dt == np.uint32:
        return u32.cxx
    if dt == np.uint64:
        return u64.cxx
    if dt == np.half:
        return f16.cxx

    if has_pytorch():
        import torch  # pylint: disable=C0415

        # pylint: disable=E1101
        if dt == torch.float32:
            return f32.cxx
        if dt == torch.float64:
            return f64.cxx
        if dt == torch.int32:
            return i32.cxx
        if dt == torch.int64:
            return i64.cxx
        if dt == torch.int8:
            return i8.cxx
        if dt == torch.int16:
            return i16.cxx
        if dt == torch.bool:
            return u1.cxx
        if dt == torch.uint8:
            return u8.cxx
        if dt == torch.float16:
            return f16.cxx

        if hasattr(torch, "uint16"):
            if dt == torch.uint16:
                return u16.cxx
            if dt == torch.uint32:
                return u32.cxx
            if dt == torch.uint64:
                return u64.cxx

        raise RuntimeError(f"PyTorch doesn't support {dt.to_string()} data type before version 2.3.0.")

    raise AssertionError(f"Unknown type {dt}")


class DataTypeCxxWrapper(_qd_core.DataTypeCxx):
    __slots__ = ("_hash",)

    def __init__(self, dtype: _qd_core.Type):
        super().__init__(dtype)
        try:
            self._hash = super().__hash__()
        except RuntimeError:
            # Hash may not be supported
            pass

    def __hash__(self):
        return self._hash


def cook_dtype(dtype: Any) -> _qd_core.DataTypeCxx:
    """Convert Python dtype to C++ DataTypeCxx.

    Handles PrimitiveBase classes, raw DataTypeCxx instances, Type instances,
    and Python builtins (float, int, bool). Uses id-based cache for hot paths.
    """
    cached = _cook_cache.get(id(dtype))
    if cached is not None:
        return cached
    _type = type(dtype)
    if isinstance(dtype, type) and issubclass(dtype, PrimitiveBase):
        return dtype.cxx
    if issubclass(_type, _qd_core.DataTypeCxx):
        return dtype
    if issubclass(_type, _qd_core.Type):
        return DataTypeCxxWrapper(dtype)
    if dtype is float:
        return impl.get_runtime().default_fp
    if dtype is int:
        return impl.get_runtime().default_ip
    if dtype is bool:
        return u1.cxx
    raise ValueError(f"Invalid data type {dtype}")


def dtype_to_torch_dtype(dtype: Any):
    import torch  # pylint: disable=C0415

    return {
        float: torch.float32,
        int: torch.int32,
        i32: torch.int32,
        f32: torch.float32,
        i64: torch.int64,
        f64: torch.float64,
        bool: torch.bool,
        u1: torch.bool,
    }[dtype]


def in_quadrants_scope():
    return impl.inside_kernel()


def in_python_scope():
    return not in_quadrants_scope()


def quadrants_scope(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if not impl.is_python_backend():
            assert in_quadrants_scope(), f"{func.__name__} cannot be called in Python-scope"
        return func(*args, **kwargs)

    return wrapped


def python_scope(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        assert in_python_scope(), f"{func.__name__} cannot be called in Quadrants-scope"
        return func(*args, **kwargs)

    return wrapped


def warning(msg, warning_type=UserWarning, stacklevel=1, print_stack=True):
    """Print a warning message. Note that the builtin `warnings` module is
    unreliable since it may be suppressed by other packages such as IPython.

    Args:
        msg (str): message to print.
        warning_type (Type[Warning]): type of warning.
        stacklevel (int): warning stack level from the caller.
        print_stack (bool): whether to print the stack
    """
    if not is_logging_effective("warn"):
        return
    if print_stack:
        msg += f"\n{get_traceback(stacklevel)}"
    warnings.warn(Fore.YELLOW + Style.BRIGHT + msg + Style.RESET_ALL, warning_type)


def get_traceback(stacklevel=1):
    s = traceback.extract_stack()[: -1 - stacklevel]
    return "".join(traceback.format_list(s))


def is_data_oriented(obj: Any) -> bool:
    # Use getattr on class instead of object to bypass custom __getattr__ method that is
    # overwritten at instance level and very slow.
    return getattr(type(obj), "_data_oriented", False)


def is_qd_template(annotation: Any) -> bool:
    return annotation is Template or type(annotation) is Template


__all__ = []
