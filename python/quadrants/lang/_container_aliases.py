"""Describe shared containers so cached kernels retain compatible member bindings."""

import dataclasses
from collections.abc import Sequence
from typing import Any

from quadrants.lang.util import is_data_oriented, is_dataclass_instance


def get_container_alias_key(args: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    """Record container sharing across arguments without including process-specific object addresses.

    Container references must remain fixed after the first kernel call. The template mapper caches this key with
    its existing argument-instance cache; scalar values may change without changing the alias layout.
    """
    serials: dict[int, int] = {}
    result: list[tuple[int, ...]] = []
    for arg in args:
        identities: list[int] = []
        pending = [arg]
        while pending:
            py_obj = pending.pop()
            is_dataclass = is_dataclass_instance(py_obj)
            if not is_dataclass and not is_data_oriented(py_obj):
                continue
            identity = id(py_obj)
            if identity in serials:
                identities.append(serials[identity])
                continue
            serials[identity] = len(serials)
            identities.append(serials[identity])
            if is_dataclass:
                pending.extend(getattr(py_obj, field.name) for field in dataclasses.fields(py_obj))
            else:
                members = py_obj._asdict() if hasattr(py_obj, "_asdict") else vars(py_obj)
                pending.extend(value for name, value in members.items() if not name.startswith("_qd_"))
        result.append(tuple(identities))
    return tuple(result)
