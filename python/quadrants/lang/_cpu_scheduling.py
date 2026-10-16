"""Resolve CPU scheduling settings and their deprecated initialization aliases."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Any

from quadrants._lib import core as _qd_core


def _is_supplied(kwargs: dict[str, Any], name: str) -> bool:
    return name in kwargs or bool(os.environ.get("QD_" + name.upper()))


def _read(kwargs: dict[str, Any], name: str, cast: Callable[[str], Any]) -> Any:
    env_name = "QD_" + name.upper()
    env_value = os.environ.get(env_name, "")
    if name in kwargs:
        if env_value:
            _qd_core.warn(f'Environment variable {env_name}={env_value} overridden by qd.init argument "{name}"')
        return kwargs.pop(name)
    return cast(env_value)


def _mode_from_env(value: str) -> _qd_core.CPUWorkScheduling:
    try:
        return {
            "PER_WORKER": _qd_core.CPUWorkScheduling.PER_WORKER,
            "FIXED_SIZE": _qd_core.CPUWorkScheduling.FIXED_SIZE,
        }[value]
    except KeyError:
        raise ValueError("QD_CPU_WORK_SCHEDULING must be PER_WORKER or FIXED_SIZE") from None


def configure_cpu_scheduling(kwargs: dict[str, Any], cfg: _qd_core.CompileConfig) -> None:
    if _is_supplied(kwargs, "make_cpu_multithreading_loop") and _is_supplied(kwargs, "cpu_work_scheduling"):
        raise ValueError(
            "Cannot specify both make_cpu_multithreading_loop and cpu_work_scheduling, including through environment variables"
        )
    if _is_supplied(kwargs, "default_cpu_block_dim") and _is_supplied(kwargs, "cpu_fixed_block_dim"):
        raise ValueError(
            "Cannot specify both default_cpu_block_dim and cpu_fixed_block_dim, including through environment variables"
        )

    if _is_supplied(kwargs, "make_cpu_multithreading_loop"):
        warnings.warn(
            "make_cpu_multithreading_loop is deprecated; use cpu_work_scheduling instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        value = _read(kwargs, "make_cpu_multithreading_loop", lambda value: bool(int(value)))
        if not isinstance(value, (bool, int)) or value not in (False, True):
            raise TypeError("make_cpu_multithreading_loop must be a boolean")
        kwargs["cpu_work_scheduling"] = (
            _qd_core.CPUWorkScheduling.PER_WORKER if value else _qd_core.CPUWorkScheduling.FIXED_SIZE
        )

    if _is_supplied(kwargs, "default_cpu_block_dim"):
        warnings.warn(
            "default_cpu_block_dim is deprecated; use cpu_fixed_block_dim instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        kwargs["cpu_fixed_block_dim"] = _read(kwargs, "default_cpu_block_dim", int)

    if _is_supplied(kwargs, "cpu_work_scheduling"):
        value = _read(kwargs, "cpu_work_scheduling", _mode_from_env)
        if not isinstance(value, _qd_core.CPUWorkScheduling):
            raise TypeError("cpu_work_scheduling must be a qd.CPUWorkScheduling member")
        cfg.cpu_work_scheduling = value

    if _is_supplied(kwargs, "cpu_fixed_block_dim"):
        cfg.cpu_fixed_block_dim = _read(kwargs, "cpu_fixed_block_dim", int)
