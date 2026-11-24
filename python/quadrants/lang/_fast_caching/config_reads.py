"""Track Python reads of native config fields during kernel compilation."""

import struct
from contextlib import contextmanager
from typing import Any, Iterable

from quadrants._lib.core.quadrants_python import CompileConfig
from quadrants.lang import impl

ConfigField = tuple[str, str]


class ConfigReads:
    def __init__(self):
        self.configs = _config_sources()
        self.values: dict[ConfigField, str] = {}
        self.cacheable = True
        self.compiled_real_functions: set[tuple[int, int]] = set()

    def record(self, config, name, value):
        for source, known_config in self.configs.items():
            if config is known_config:
                field = (source, name)
                fingerprint = _fingerprint(value)
                if self.values.setdefault(field, fingerprint) != fingerprint:
                    self.cacheable = False
                return
        # Arbitrary config objects cannot be identified reliably at lookup time.
        self.cacheable = False


def _config_sources() -> dict[str, CompileConfig]:
    return {"program": impl.current_cfg(), "default": impl.default_cfg()}


def _fingerprint(value: Any) -> str:
    if isinstance(value, float):
        # Keep the sign and payload of a NaN.
        return struct.pack("!d", value).hex()
    return str(value)


def read_config_values(fields: Iterable[ConfigField]) -> dict[ConfigField, str] | None:
    if not fields:
        return {}
    configs = _config_sources()
    try:
        return {field: _fingerprint(getattr(configs[field[0]], field[1])) for field in fields}
    except (KeyError, AttributeError):
        return None


_active_reads: ConfigReads | None = None


def record_real_function(func_id: int, instance_id: int, already_compiled: bool) -> None:
    reads = _active_reads
    if reads is None:
        return
    key = (func_id, instance_id)
    if not already_compiled:
        reads.compiled_real_functions.add(key)
    elif key not in reads.compiled_real_functions:
        # Another kernel compiled this body. The caller cannot observe its config reads.
        reads.cacheable = False


def _tracked_getter(name, getter):
    def get(config):
        reads = _active_reads
        value = getter(config)
        if reads is not None:
            reads.record(config, name, value)
        return value

    return get


def _install_getters():
    # A Python helper can read an alias without going through the AST transformer.
    for name, field in vars(CompileConfig).items():
        if isinstance(field, property) and field.fget is not None:
            setattr(CompileConfig, name, field.getter(_tracked_getter(name, field.fget)))


@contextmanager
def collect_config_reads(reads: ConfigReads | None):
    # The runtime compilation lock serializes kernel compilation. A process-wide collector also sees
    # reads from worker threads inside a compile-time Python helper. Unrelated reads can cause extra misses.
    global _active_reads  # pylint: disable=global-statement
    previous = _active_reads
    _active_reads = reads
    try:
        yield
    finally:
        _active_reads = previous


_install_getters()
