import os
from collections.abc import Callable
from enum import Enum
from typing import Any

from quadrants._lib import core as _qd_core


class _EnvironmentConfigurator:
    def __init__(self, kwargs, _cfg):
        self.cfg = _cfg
        self.kwargs = kwargs
        self.keys = []

    def configure(self, key, _cast=None):
        # QD_OFFLINE_CACHE=   : no effect
        # QD_OFFLINE_CACHE=0  : False
        # QD_OFFLINE_CACHE=1  : True
        self.keys.append(key)
        if not self.is_supplied(key):
            return

        value_type = type(self[key])
        if issubclass(value_type, Enum):

            def parse_enum(text: str) -> Enum:
                try:
                    return value_type[text]
                except KeyError:
                    choices = " or ".join(value_type.__members__)
                    raise ValueError(f"QD_{key.upper()} must be {choices}") from None

            value = self.read(key, parse_enum)
            if not isinstance(value, value_type):
                raise TypeError(f"{key} must be a {value_type.__name__} member")
            self[key] = value
        else:
            if _cast is None:
                _cast = self.bool_int if value_type is bool else value_type
            self[key] = self.read(key, _cast)

    def is_supplied(self, key: str) -> bool:
        return key in self.kwargs or bool(os.environ.get("QD_" + key.upper()))

    def read(self, key: str, env_parse_fn: Callable[[str], Any] | None = None) -> Any:
        """Read a supplied option, removing it from kwargs and parsing only environment-variable values."""
        env_parse_fn = env_parse_fn or self.bool_int
        name = "QD_" + key.upper()
        value = os.environ.get(name, "")
        if key in self.kwargs:
            if value:
                _qd_core.warn(f'Environment variable {name}={value} overridden by qd.init argument "{key}"')
            return self.kwargs.pop(key)
        return env_parse_fn(value)

    def __getitem__(self, key):
        return getattr(self.cfg, key)

    def __setitem__(self, key, value):
        setattr(self.cfg, key, value)

    @staticmethod
    def bool_int(x):
        return bool(int(x))
