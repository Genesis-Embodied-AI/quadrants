from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, TypeAlias

from quadrants.types.enums import AutodiffMode


class KernelBatchedArgType(IntEnum):
    FLOAT = 0
    INT = 1
    UINT = 2
    QD_ARRAY = 3
    QD_ARRAY_WITH_GRAD = 4


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
class PerOffloadCacheObservations:
    """Per-offloaded-task compilation cache stats for one kernel compile.

    Counts the per-task codegen cache (LLVM module reuse per offloaded task; always active on the LLVM backends). On
    a warm compile where only one task changed, the expected result is ``recompiled == 1`` and
    ``cache_hit == total - 1``. Counts (not wall time) so the behavior can be asserted deterministically in tests.
    """

    constructs_total: int = 0
    constructs_cache_hit: int = 0
    constructs_recompiled: int = 0


@dataclass
class LaunchObservations:
    found_kernel_in_materialize_cache: bool = False


@dataclass
class LaunchStats:
    kernel_args_count_by_type: dict[KernelBatchedArgType, int]


CompiledKernelKeyType = tuple[Callable, int, AutodiffMode]
ArgsHash: TypeAlias = tuple[int, ...]
