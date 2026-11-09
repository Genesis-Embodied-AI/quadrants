"""Store deferred errors and raise them if compilation uses the affected members."""

from functools import cache
from typing import NamedTuple, NoReturn


class DeferredMemberType(NamedTuple):
    """Store what is wrong: the exception class, message, and optional debug type names.

    When created via _create_deferred_member_type, it is cached and can be shared by multiple members.

    An unsupported member in a schema gets this description instead of a normal inferred type. The schema cache key
    includes each member type's identity.
    """

    error_type: type[Exception]
    message: str
    # (type(py_obj).__module__, type(py_obj).__qualname__), e.g. ("builtins", "set"). Captured only with debug=True;
    # otherwise None. Store strings instead of the class so cached diagnostics do not keep application classes alive.
    runtime_type_names: tuple[str, str] | None = None

    def bind(self, qualified_member_name: str) -> "DeferredMemberError":
        """Combine this description with a qualified member name for a compiler error.

        The name starts from the root argument or variable in the compiled code, followed by its member names.
        For example:

            qualified_member_name = "state.inner.velocity"

        Here, "state" is the root argument or variable name, and "inner.velocity" identifies the member relative to
        that object. "state" is not a module or dataclass class name.
        """
        return DeferredMemberError(self, qualified_member_name)


class DeferredMemberError(NamedTuple):
    """Combine the description from DeferredMemberType with where it occurred: the qualified member name.

    An example qualified member name is "state.velocity".

    Its reject() method raises the error if compilation actually uses that member.
    """

    member_type: DeferredMemberType
    qualified_member_name: str

    def reject(self) -> NoReturn:
        """Raise the recorded error, formatting optional debug type names only when the member is used."""
        message = f"Dataclass member {self.qualified_member_name!r}: {self.member_type.message}"
        if self.member_type.runtime_type_names is not None:
            module, qualname = self.member_type.runtime_type_names
            message += f" (runtime type: {module}.{qualname})"
        raise self.member_type.error_type(message)


@cache
def _create_deferred_member_type(
    error_type: type[Exception],
    message: str,
    # (type(py_obj).__module__, type(py_obj).__qualname__), e.g. ("builtins", "set"). Captured only with debug=True;
    # otherwise None. Store strings instead of the class so cached diagnostics do not keep application classes alive.
    runtime_type_names: tuple[str, str] | None = None,
) -> DeferredMemberType:
    """Intern descriptions including debug metadata; schema keys include the resulting description's identity."""
    return DeferredMemberType(error_type, message, runtime_type_names)


def create_deferred_member_error(error_type: type[Exception], message: str, py_obj: object) -> DeferredMemberType:
    """Return a cached description of an invalid dataclass member for later error reporting.

    Capture runtime type names only in debug mode, without retaining the application value or its class.
    """
    # Import lazily: impl also loads compiler modules that use deferred descriptions.
    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

    runtime_type_names = None
    if impl.current_cfg().debug:
        runtime_type = type(py_obj)
        runtime_type_names = (runtime_type.__module__, runtime_type.__qualname__)
    return _create_deferred_member_type(error_type, message, runtime_type_names)
