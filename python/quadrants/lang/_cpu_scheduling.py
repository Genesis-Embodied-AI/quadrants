"""Resolve CPU scheduling settings and their deprecated initialization aliases."""

import os
import warnings

from quadrants._lib import core as _qd_core


def _is_supplied(kwargs, name):
    return name in kwargs or bool(os.environ.get("QD_" + name.upper()))


def _read(kwargs, name, cast):
    env_name = "QD_" + name.upper()
    env_value = os.environ.get(env_name, "")
    if name in kwargs:
        if env_value:
            _qd_core.warn(f'Environment variable {env_name}={env_value} overridden by qd.init argument "{name}"')
        return kwargs.pop(name)
    return cast(env_value)


def _mode_from_env(value):
    try:
        return {
            "PER_WORKER": _qd_core.CPUWorkScheduling.PER_WORKER,
            "FIXED_SIZE": _qd_core.CPUWorkScheduling.FIXED_SIZE,
        }[value]
    except KeyError:
        raise ValueError("QD_CPU_WORK_SCHEDULING must be PER_WORKER or FIXED_SIZE") from None


def configure_cpu_scheduling(kwargs, cfg):
    aliases = (
        ("make_cpu_multithreading_loop", "cpu_work_scheduling"),
        ("default_cpu_block_dim", "cpu_fixed_block_dim"),
    )
    # Check all pairs before consuming arguments. Even equal values are ambiguous.
    for old, new in aliases:
        if _is_supplied(kwargs, old) and _is_supplied(kwargs, new):
            raise ValueError(f"Cannot specify both {old} and {new}, including through environment variables")

    for old, new in aliases:
        if _is_supplied(kwargs, old):
            warnings.warn(f"{old} is deprecated; use {new} instead.", DeprecationWarning, stacklevel=3)
            if new == "cpu_work_scheduling":
                value = _read(kwargs, old, lambda value: bool(int(value)))
                if not isinstance(value, (bool, int)) or value not in (False, True):
                    raise TypeError("make_cpu_multithreading_loop must be a boolean")
                cfg.cpu_work_scheduling = (
                    _qd_core.CPUWorkScheduling.PER_WORKER if value else _qd_core.CPUWorkScheduling.FIXED_SIZE
                )
            else:
                cfg.cpu_fixed_block_dim = _read(kwargs, old, int)
        elif _is_supplied(kwargs, new):
            if new == "cpu_work_scheduling":
                value = _read(kwargs, new, _mode_from_env)
                if not isinstance(value, _qd_core.CPUWorkScheduling):
                    raise TypeError("cpu_work_scheduling must be a qd.CPUWorkScheduling member")
                cfg.cpu_work_scheduling = value
            else:
                cfg.cpu_fixed_block_dim = _read(kwargs, new, int)
