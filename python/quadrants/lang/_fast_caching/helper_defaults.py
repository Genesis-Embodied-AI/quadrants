"""Track evaluated defaults of helpers that fastcache can resolve without compiling."""

import inspect
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterable, cast

from .._quadrants_callable import QuadrantsCallable

if TYPE_CHECKING:
    from .._func_base import FuncBase

HelperPath = tuple[str, ...]
HelperSource = tuple[str, int, str]
HelperDefaults = tuple[tuple[HelperSource, ...], tuple[tuple[str, str], ...]]


def _source(func: "FuncBase") -> HelperSource:
    code = func.func.__code__
    return code.co_filename, code.co_firstlineno, code.co_qualname


def _value(value: Any) -> str | None:
    if value is None:
        return "none"
    value_type = type(value)
    if value_type is bool:
        return "true" if value else "false"
    if value_type is int:
        return "int:" + hex(value)
    if value_type is float:
        return "float:" + struct.pack("!d", value).hex()
    if value_type is str:
        return "str:" + value
    if value_type is bytes:
        return "bytes:" + value.hex()
    return None


def _defaults(func: "FuncBase") -> tuple[tuple[str, str], ...] | None:
    values = []
    for arg in func.arg_metas:
        if arg.default is inspect.Parameter.empty:
            continue
        value = _value(arg.default)
        if value is None:
            return None
        values.append((arg.name, value))
    return tuple(values)


def read_helper_defaults(root: "FuncBase", paths: Iterable[HelperPath]) -> dict[HelperPath, HelperDefaults] | None:
    values = {}
    for path in paths:
        func = root
        sources = []
        for name in path:
            fn = func.func
            try:
                if fn.__closure__ and name in fn.__code__.co_freevars:
                    helper = fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents
                else:
                    helper = fn.__globals__.get(name)
            except ValueError:  # Empty closure cell.
                return None
            if type(helper) is not QuadrantsCallable or not helper._is_quadrants_function:
                return None
            func = cast("FuncBase", helper.wrapper)
            sources.append(_source(func))
        defaults = _defaults(func)
        if defaults is None:
            return None
        values[path] = (tuple(sources), defaults)
    return values


class HelperDefaultReads:
    def __init__(self, root: "FuncBase") -> None:
        self.values: dict[HelperPath, HelperDefaults] = {}
        self.cacheable = True
        self.root = root
        self._binding: tuple[FuncBase | None, HelperPath, tuple[HelperSource, ...]] = (root, (), ())

    @contextmanager
    def bind(self, caller: "FuncBase", helper: Any, name: str | None = None):
        previous = self._binding
        bound_func, path, sources = previous
        self._binding = (None, (), ())
        if caller is bound_func and name is not None and type(helper) is QuadrantsCallable:
            if helper._is_quadrants_function:
                callee = cast("FuncBase", helper.wrapper)
                self._binding = (callee, (*path, name), (*sources, _source(callee)))
        try:
            yield
        finally:
            self._binding = previous

    def record(self, func: "FuncBase") -> None:
        defaults = _defaults(func)
        bound_func, path, sources = self._binding
        if func is not bound_func:
            # Do not guess which instance a dynamic call will select on a cache hit.
            if defaults != ():
                self.cacheable = False
            return
        # An empty default list also matters if this name later refers to a helper with defaults.
        if defaults is None or self.values.setdefault(path, (sources, defaults)) != (sources, defaults):
            self.cacheable = False
