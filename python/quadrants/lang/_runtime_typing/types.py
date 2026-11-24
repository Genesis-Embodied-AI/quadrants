"""Types shared by runtime dataclass processing."""

import dataclasses
from typing import Any, ClassVar, NamedTuple, NoReturn, Protocol, TypeAlias

from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants.lang.matrix import MatrixType
from quadrants.lang.struct import StructType
from quadrants.types import buffer_view_type, ndarray_type


class DataclassInstance(Protocol):
    """Describe the field metadata shared by Python dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


class RuntimeDataclassSchema(DataclassInstance, Protocol):
    """Describe generated schema metadata inferred from an arriving dataclass instance."""

    # Original dataclass class described by this generated schema.
    _qd_underlying_dataclass: ClassVar[type[DataclassInstance]]
    # Inferred member types can be reused without guards; frozen=True on the original class alone is insufficient.
    _qd_frozen: ClassVar[bool]
    # This schema or a nested schema contains errors raised only if compilation uses the affected members.
    _qd_has_deferred_errors: ClassVar[bool]
    # Original class identity and ordered member type signatures, used to select a compiled kernel.
    _qd_type_signature: ClassVar[tuple[int, tuple[tuple[str, Any], ...]]]


# Object identity maps to the retained object and its type-tagged Final compilation key.
FinalValueKeyCache: TypeAlias = dict[int, tuple[Any, tuple[Any, ...]]]

InferredMemberType: TypeAlias = (
    type | DataTypeCxx | ndarray_type.NdarrayType | buffer_view_type.BufferViewType | MatrixType | StructType
)

# Class identity, numeric-default identities, and mutable-ancestor context.
InferenceContextKey: TypeAlias = tuple[int, int, int, str | None]


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

    def bind(self, dotted_argument_path: str) -> "DeferredMemberError":
        """Combine this description with a dotted argument path for a compiler error.

        The path starts with the argument name in the compiled code, followed by member names separated by dots.
        For example:

            dotted_argument_path = "state.inner.velocity"

        Here, "state" is the argument name, and "inner.velocity" identifies the member relative to that argument.
        """
        return DeferredMemberError(self, dotted_argument_path)


class DeferredMemberError(NamedTuple):
    """Combine the description from DeferredMemberType with where it occurred: the dotted argument path.

    An example dotted argument path is "state.velocity".

    Its reject() method raises the error if compilation actually uses that member.
    """

    member_type: DeferredMemberType
    dotted_argument_path: str

    def reject(self) -> NoReturn:
        """Raise the recorded error, formatting optional debug type names only when the member is used."""
        message = f"Dataclass member {self.dotted_argument_path!r}: {self.member_type.message}"
        if self.member_type.runtime_type_names is not None:
            module, qualname = self.member_type.runtime_type_names
            message += f" (runtime type: {module}.{qualname})"
        raise self.member_type.error_type(message)
