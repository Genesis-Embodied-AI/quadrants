"""Types shared by runtime dataclass processing."""

import dataclasses
from collections.abc import Callable
from types import SimpleNamespace
from typing import (
    Any,
    ClassVar,
    NamedTuple,
    NoReturn,
    Protocol,
    TypeAlias,
    TypeGuard,
    cast,
)

from quadrants._lib.core.quadrants_python import DataTypeCxx
from quadrants.lang.matrix import MatrixType
from quadrants.lang.struct import StructType
from quadrants.types import buffer_view_type, ndarray_type


class DataclassOptions(Protocol):
    """Describe the attributes we read from Python's existing __dataclass_params__ object."""

    frozen: bool
    unsafe_hash: bool


class DataclassInstance(Protocol):
    """Describe the field metadata and decorator options shared by Python dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]
    __dataclass_params__: ClassVar[DataclassOptions]


class RuntimeDataclassSchema(DataclassInstance, Protocol):
    """Describe generated schema metadata inferred from an arriving dataclass instance."""

    # Original dataclass class described by this generated schema.
    _qd_underlying_dataclass: ClassVar[type[DataclassInstance]]
    # Inferred member types need checking before reuse; a frozen dataclass can still contain mutable members.
    _qd_needs_member_check: ClassVar[bool]
    # This schema or a nested schema contains errors raised only if compilation uses the affected members.
    _qd_has_deferred_errors: ClassVar[bool]
    # Original class identity and ordered member type signatures, used to select a compiled kernel.
    _qd_type_signature: ClassVar[tuple[int, tuple[tuple[str, Any], ...]]]


def is_frozen_dataclass(dataclass_type: type[DataclassInstance]) -> bool:
    """Treat frozen=True or unsafe_hash=True as a promise against field reassignment."""
    return dataclass_type.__dataclass_params__.frozen or dataclass_type.__dataclass_params__.unsafe_hash


def is_runtime_dataclass_schema(py_obj: object) -> TypeGuard[type[RuntimeDataclassSchema]]:
    """Recognize a generated dataclass schema class by its schema metadata."""
    return isinstance(py_obj, type) and hasattr(py_obj, "_qd_underlying_dataclass")


def create_dataclass_schema(
    dataclass_type: type[DataclassInstance],
    fields: dict[str, dataclasses.Field[Any]],
    needs_member_check: bool,
    has_deferred_error: bool,
) -> type[RuntimeDataclassSchema]:
    """Create a schema class from inferred fields and cache-reuse flags."""
    # FIXME: Replace this fake dataclass with a schema object, and migrate the dataclass-processing code that
    # consumes it to explicitly expect schema objects rather than relying on the dataclass interface.
    return cast(
        type[RuntimeDataclassSchema],
        type(
            dataclass_type.__name__,
            (),
            {
                "__module__": dataclass_type.__module__,
                "__qualname__": dataclass_type.__qualname__,
                "__dataclass_fields__": fields,
                "__dataclass_params__": SimpleNamespace(
                    frozen=is_frozen_dataclass(dataclass_type),
                    unsafe_hash=False,
                ),
                "__hash__": dataclass_type.__hash__,
                "_qd_underlying_dataclass": dataclass_type,
                "_qd_needs_member_check": needs_member_check,
                "_qd_has_deferred_errors": has_deferred_error,
                "_qd_type_signature": (
                    id(dataclass_type),
                    tuple(
                        (name, getattr(field.type, "_qd_type_signature", id(field.type)))
                        for name, field in fields.items()
                    ),
                ),
            },
        ),
    )


# Object identity maps to the retained object and its type-tagged Final specialization signature.
FinalValueSpecializationSignatureCache: TypeAlias = dict[int, tuple[Any, tuple[Any, ...]]]

InferredMemberType: TypeAlias = (
    type | DataTypeCxx | ndarray_type.NdarrayType | buffer_view_type.BufferViewType | MatrixType | StructType
)

# Integer and float default identities, debug setting, and mutable-ancestor path.
InferenceContextKey: TypeAlias = tuple[int, int, bool, str | None]


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


class SchemaNeedingMemberChecks:
    """Store member names and types, never child objects that could retain the owning dataclass."""

    __slots__ = ("schema", "member_guards")

    def __init__(
        self,
        schema: "type[RuntimeDataclassSchema]",
        member_guards: "list[tuple[str, InferredMemberType | DeferredMemberType]]",
    ) -> None:
        """Store the member checks selected during runtime type inference."""
        self.schema = schema
        self.member_guards = member_guards

    def do_member_check(
        self,
        dataclass_instance: "DataclassInstance",
        path: str,
        ancestor_ids: frozenset[int],
        infer_member_type: "Callable[[Any, str, frozenset[int]], tuple[InferredMemberType | DeferredMemberType, bool]]",
    ) -> bool:
        """Check whether the current member types still match the cached schema."""
        for name, member_type in self.member_guards:
            child = getattr(dataclass_instance, name)
            current_member_type, _ = infer_member_type(child, f"{path}.{name}", ancestor_ids)
            if current_member_type is not member_type:
                return False
        return True


class RuntimeDataclassSpecializationSignature(tuple[Any, ...]):
    """Store field specialization signatures and the inferred schema for hashing and equality."""

    schema: type[RuntimeDataclassSchema]

    def __new__(
        cls, field_specialization_signatures: tuple[Any, ...], schema: type[RuntimeDataclassSchema]
    ) -> "RuntimeDataclassSpecializationSignature":
        """Attach the inferred schema to the field specialization signatures."""
        result = super().__new__(cls, field_specialization_signatures)
        result.schema = schema
        return result

    def __hash__(self) -> int:
        """Hash the inferred type signature and field specialization signatures."""
        return hash((self.schema._qd_type_signature, tuple(self)))

    def __eq__(self, other: object) -> bool:
        """Compare inferred type signatures and field specialization signatures."""
        return (
            type(other) is type(self)
            and self.schema._qd_type_signature == other.schema._qd_type_signature
            and tuple.__eq__(self, other)
        )

    def __ne__(self, other: object) -> bool:
        """Return whether the specialization signatures differ."""
        return not self == other
