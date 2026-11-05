"""Carry unsupported dataclass leaves through argument discovery until a kernel actually uses them."""

from functools import cache
from typing import NamedTuple


class DeferredMemberType(NamedTuple):
    """An interned type description, without the application value or an exception traceback."""

    error_type: type[Exception]
    message: str

    def bind(self, flat_name: str) -> "DeferredMemberValue":
        return DeferredMemberValue(self, flat_name.removeprefix("__qd_").replace("__qd_", "."))


class DeferredMemberValue(NamedTuple):
    """Compiler binding retaining the original kernel path across inlined function argument renaming."""

    member_type: DeferredMemberType
    path: str

    def reject(self):
        raise self.member_type.error_type(f"Dataclass member {self.path!r}: {self.member_type.message}")


@cache
def deferred_member(error_type: type[Exception], message: str) -> DeferredMemberType:
    return DeferredMemberType(error_type, message)
