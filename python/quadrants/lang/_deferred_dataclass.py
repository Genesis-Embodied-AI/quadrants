"""Carry unsupported dataclass leaves through argument discovery until a kernel actually uses them."""

from functools import cache
from typing import NamedTuple, NoReturn


class DeferredMemberType(NamedTuple):
    """Store what is wrong: the exception class, message, and optional debug type names.

    When created via _deferred_member, it is cached and can be shared by multiple members.

    An unsupported member in a schema gets this description instead of a normal inferred type. The schema cache key
    includes each member type's identity.
    """

    error_type: type[Exception]
    message: str
    runtime_type_names: tuple[str, str] | None = None

    def bind(self, flat_name: str) -> "DeferredMemberError":
        """Bind this description to the original member path for a compiler error."""
        return DeferredMemberError(self, flat_name.removeprefix("__qd_").replace("__qd_", "."))


class DeferredMemberError(NamedTuple):
    """Combine the description from DeferredMemberType with where it occurred: the member's path.

    An example path is "state.velocity".

    Its reject() method raises the error if compilation actually uses that member.
    """

    member_type: DeferredMemberType
    path: str

    def reject(self) -> NoReturn:
        """Raise the recorded error, formatting optional debug type names only when the member is used."""
        message = f"Dataclass member {self.path!r}: {self.member_type.message}"
        if self.member_type.runtime_type_names is not None:
            module, qualname = self.member_type.runtime_type_names
            message += f" (runtime type: {module}.{qualname})"
        raise self.member_type.error_type(message)


@cache
def _deferred_member(
    error_type: type[Exception],
    message: str,
    runtime_type_names: tuple[str, str] | None = None,
) -> DeferredMemberType:
    """Intern descriptions including debug metadata; schema keys include the resulting description's identity."""
    return DeferredMemberType(error_type, message, runtime_type_names)


def deferred_member_for_value(error_type: type[Exception], message: str, py_obj: object) -> DeferredMemberType:
    """Capture runtime type names only in debug mode, without retaining the application value or its class."""
    # Import lazily: impl also loads compiler modules that use deferred descriptions.
    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

    runtime_type_names = None
    if impl.current_cfg().debug:
        runtime_type = type(py_obj)
        runtime_type_names = (runtime_type.__module__, runtime_type.__qualname__)
    return _deferred_member(error_type, message, runtime_type_names)
