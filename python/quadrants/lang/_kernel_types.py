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
    """Per-construct FRONTEND and per-task BACKEND compilation cache stats for one kernel compile.

    ``frontend_constructs_*`` count the per-construct FRONTEND cache (reuse of the simplify / merge_global_ptrs /
    offload output per top-level construct) and are ``-1`` when the per-construct frontend split did not run for this
    compile (the whole-kernel path took it, e.g. autodiff or mesh). The frontend split itself ships no reuse tier, so
    when it runs ``frontend_constructs_recompiled == frontend_constructs_total`` and ``frontend_constructs_cache_hit ==
    0``.

    ``tasks_*`` count the cross-process per-task artifact cache (reuse of a fully-compiled offloaded task -- its
    backend code plus launch metadata) and are ``-1`` when that tier did not run (non-CUDA, or ``offline_cache=False``).
    A construct expands to one or more tasks, so ``tasks_*`` is a finer-grained view than the per-construct counts.

    Counts (not wall time) so the behavior can be asserted deterministically in tests.
    """

    frontend_constructs_total: int = -1
    frontend_constructs_cache_hit: int = -1
    frontend_constructs_recompiled: int = -1
    tasks_total: int = -1
    tasks_cache_hit: int = -1
    tasks_recompiled: int = -1


@dataclass
class LaunchObservations:
    found_kernel_in_materialize_cache: bool = False


@dataclass
class LaunchStats:
    kernel_args_count_by_type: dict[KernelBatchedArgType, int]


CompiledKernelKeyType = tuple[Callable, int, AutodiffMode]
ArgsHash: TypeAlias = tuple[int, ...]
