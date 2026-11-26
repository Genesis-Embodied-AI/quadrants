"""Create and cache deferred error descriptions for unsupported dataclass members."""

from functools import cache

from quadrants.lang._runtime_typing.types import DeferredMemberType


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


def create_deferred_member_type_for_invalid_dataclass_member(
    error_type: type[Exception], message: str, invalid_dataclass_member_object: object
) -> DeferredMemberType:
    """Return a cached description of an invalid dataclass member for later error reporting.

    Capture runtime type names only in debug mode, without retaining the application value or its class.
    """
    # Import lazily: impl also loads compiler modules that use deferred descriptions.
    from quadrants.lang import impl  # pylint: disable=import-outside-toplevel

    runtime_type_names = None
    if impl.current_cfg().debug:
        runtime_type = type(invalid_dataclass_member_object)
        runtime_type_names = (runtime_type.__module__, runtime_type.__qualname__)
    return _create_deferred_member_type(error_type, message, runtime_type_names)
