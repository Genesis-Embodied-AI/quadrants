import os
from itertools import islice
from typing import TYPE_CHECKING, Iterable

from ..._test_tools import warnings_helper
from .._wrap_inspect import FunctionSourceInfo
from .fast_caching_types import HashedFunctionSourceInfo
from .hash_utils import hash_iterable_strings

if TYPE_CHECKING:
    from quadrants.lang.kernel_impl import QuadrantsCallable


def pure(fn: "QuadrantsCallable") -> "QuadrantsCallable":
    warnings_helper.warn_once(
        "Use of @qd.pure is deprecated. Please use @qd.kernel(fastcache=True). @qd.pure is intended to be removed in v4.0.0"
    )
    fn.is_pure = True
    return fn


def _read_file(function_info: FunctionSourceInfo) -> list[str]:
    try:
        with open(function_info.filepath, encoding="utf-8") as f:
            return list(islice(f, function_info.start_lineno, function_info.end_lineno + 1))
    except Exception as e:
        raise Exception(
            f"Couldnt read file {function_info.filepath} lines {function_info.start_lineno}-{function_info.end_lineno} {function_info} exception {e}"
        )


def _hash_function(function_info: FunctionSourceInfo) -> str:
    return hash_iterable_strings(_read_file(function_info))


def hash_functions(function_infos: Iterable[FunctionSourceInfo]) -> list[HashedFunctionSourceInfo]:
    results = []
    for f_info in function_infos:
        hash_ = _hash_function(f_info)
        results.append(HashedFunctionSourceInfo(function_source_info=f_info, hash=hash_))
    return results


def hash_kernel(kernel_info: FunctionSourceInfo) -> str:
    return _hash_function(kernel_info)


def dump_stats() -> None:
    print("function hasher dump stats")


def _validate_hashed_function_info(hashed_function_info: HashedFunctionSourceInfo) -> bool:
    """
    Checks the hash
    """
    if not os.path.isfile(hashed_function_info.function_source_info.filepath):
        return False
    _hash = _hash_function(hashed_function_info.function_source_info)
    return _hash == hashed_function_info.hash


def validate_hashed_function_infos(function_infos: Iterable[HashedFunctionSourceInfo]) -> bool:
    for function_info in function_infos:
        if not _validate_hashed_function_info(function_info):
            return False
    return True
