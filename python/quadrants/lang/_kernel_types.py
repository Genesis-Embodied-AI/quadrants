from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, TypeAlias

from quadrants.types.enums import AutodiffMode


class KernelBatchedArgType(IntEnum):
    FLOAT = 0
    INT = 1
    UINT = 2
    TI_ARRAY = 3
    TI_ARRAY_WITH_GRAD = 4


@dataclass
class SrcLlCacheObservations:
    cache_key_generated: bool = False
    cache_validated: bool = False
    cache_loaded: bool = False
    cache_stored: bool = False


@dataclass
class FeLlCacheObservations:
    cache_hit: bool = False


@dataclass
class LaunchObservations:
    found_kernel_in_materialize_cache: bool = False


@dataclass
class LaunchStats:
    kernel_args_count_by_type: dict[KernelBatchedArgType, int]


CompiledKernelKeyType = tuple[Callable, int, AutodiffMode]
ArgsHash: TypeAlias = tuple[int, ...]
