"""Normalize deprecated CPU scheduling options to their current names."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from quadrants._lib import core as _qd_core

if TYPE_CHECKING:
    from quadrants.lang.misc import _EnvironmentConfigurator


def normalize_deprecated_cpu_scheduling_aliases(env_comp: _EnvironmentConfigurator) -> None:
    kwargs = env_comp.kwargs
    if env_comp.is_supplied("make_cpu_multithreading_loop") and env_comp.is_supplied("cpu_work_scheduling"):
        raise ValueError(
            "Cannot specify both make_cpu_multithreading_loop and cpu_work_scheduling, including through environment variables"
        )
    if env_comp.is_supplied("default_cpu_block_dim") and env_comp.is_supplied("cpu_fixed_block_dim"):
        raise ValueError(
            "Cannot specify both default_cpu_block_dim and cpu_fixed_block_dim, including through environment variables"
        )

    if env_comp.is_supplied("make_cpu_multithreading_loop"):
        warnings.warn(
            "make_cpu_multithreading_loop is deprecated; use cpu_work_scheduling instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        value = env_comp.read("make_cpu_multithreading_loop")
        if not isinstance(value, (bool, int)) or value not in (False, True):
            raise TypeError("make_cpu_multithreading_loop must be a boolean")
        kwargs["cpu_work_scheduling"] = (
            _qd_core.CPUWorkScheduling.PER_WORKER if value else _qd_core.CPUWorkScheduling.FIXED_SIZE
        )

    if env_comp.is_supplied("default_cpu_block_dim"):
        warnings.warn(
            "default_cpu_block_dim is deprecated; use cpu_fixed_block_dim instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        kwargs["cpu_fixed_block_dim"] = env_comp.read("default_cpu_block_dim", int)
