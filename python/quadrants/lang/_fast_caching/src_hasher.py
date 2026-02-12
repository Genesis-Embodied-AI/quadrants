import json
import warnings
from typing import Any, Iterable, Sequence

import pydantic
from pydantic import BaseModel

import quadrants
from quadrants import _logging

from .._wrap_inspect import FunctionSourceInfo
from ..kernel_arguments import ArgMetadata
from . import args_hasher, config_hasher, function_hasher
from .fast_caching_types import HashedFunctionSourceInfo
from .hash_utils import hash_iterable_strings
from .python_side_cache import PythonSideCache


def create_cache_key(
    raise_on_templated_floats: bool,
    kernel_source_info: FunctionSourceInfo,
    args: Sequence[Any],
    arg_metas: Sequence[ArgMetadata],
) -> str | None:
    """
    cache key takes into account:
    - arg types
    - cache value arg values
    - kernel function (but not sub functions)
    - compilation config (which includes arch, and debug)
    """
    args_hash = args_hasher.hash_args(raise_on_templated_floats, args, arg_metas)
    if args_hash is None:
        # the bit in caps at start should not be modified without modifying corresponding text
        # freetext bit can be freely modified
        _logging.warn(
            f"[FASTCACHE][INVALID_FUNC] The pure function {kernel_source_info.function_name} could not be "
            "fast cached, because one or more parameter types were invalid"
        )
        return None
    kernel_hash = function_hasher.hash_kernel(kernel_source_info)
    config_hash = config_hasher.hash_compile_config()
    cache_key = hash_iterable_strings(
        (
            quadrants.__version_str__,
            kernel_hash,
            args_hash,
            config_hash,
            kernel_source_info.filepath,
            str(kernel_source_info.start_lineno),
            "pruned",
        )
    )
    return cache_key


class CacheValue(BaseModel):
    frontend_cache_key: str
    hashed_function_source_infos: list[HashedFunctionSourceInfo]
    used_py_dataclass_parameters: set[str]


def store(
    frontend_cache_key: str,
    fast_cache_key: str,
    function_source_infos: Iterable[FunctionSourceInfo],
    used_py_dataclass_parameters: set[str],
) -> None:
    """
    Note that unlike other caches, this cache is not going to store the actual value we want.
    This cache is only used for verification that our cache key is valid. Big picture:
    - we have a cache key, based on args and top level kernel function
    - we want to use this to look up LLVM IR, in C++ side cache
    - however, before doing that, we first want to validate that the source code didn't change
        - i.e. is our cache key still valid?
    - the python side cache contains information we will use to verify that our cache key is valid
        - ie the list of function source infos

    Update! We are now going to store parameter pruning infomation, which is:
    - used_py_dataclass_parameters: set[str]

    Update 2: we are going to store the cache key used by the c++ kernel cache, so that we can use that
    to retrieve the immutable cached c++ kernel later, rather than, before, we were storing the c++
    cached kernel using the fast cache key, leading to bugs, when cached kernel file then had to be mutable.
    """
    if not fast_cache_key:
        return
    assert frontend_cache_key is not None
    cache = PythonSideCache()
    hashed_function_source_infos = function_hasher.hash_functions(function_source_infos)
    cache_value_obj = CacheValue(
        frontend_cache_key=frontend_cache_key,
        hashed_function_source_infos=list(hashed_function_source_infos),
        used_py_dataclass_parameters=used_py_dataclass_parameters,
    )
    cache.store(fast_cache_key, cache_value_obj.model_dump_json())


def _try_load(cache_key: str) -> CacheValue | None:
    cache = PythonSideCache()
    maybe_cache_value_json = cache.try_load(cache_key)
    if maybe_cache_value_json is None:
        return None
    try:
        cache_value_obj = CacheValue.model_validate_json(maybe_cache_value_json)
    except (pydantic.ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
        warnings.warn(f"Failed to parse cache file {e}")
        return None
    return cache_value_obj


def load(cache_key: str) -> tuple[set[str], str] | tuple[None, None]:
    """
    loads function source infos from cache, if available
    checks the hashes against the current source code
    """
    cache_value = _try_load(cache_key)
    if cache_value is None:
        return None, None
    if function_hasher.validate_hashed_function_infos(cache_value.hashed_function_source_infos):
        return cache_value.used_py_dataclass_parameters, cache_value.frontend_cache_key
    return None, None


def dump_stats() -> None:
    print("dump stats")
    args_hasher.dump_stats()
    function_hasher.dump_stats()
