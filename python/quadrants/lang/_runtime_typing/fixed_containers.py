"""Guard container attributes after their first kernel use."""

from typing import Any

from quadrants.lang._weak_identity_cache import WeakIdentityCache
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.util import is_data_oriented, is_dataclass_instance

_fixed_names: WeakIdentityCache[frozenset[str]] = WeakIdentityCache()


def _is_container(py_obj: Any) -> bool:
    """Recognize references whose replacement could change a dataclass access route."""
    return isinstance(py_obj, (tuple, list, dict, set)) or is_dataclass_instance(py_obj) or is_data_oriented(py_obj)


def _install_assignment_guards(cls: type) -> None:
    """Wrap ordinary assignments and deletions, preserving the class's existing implementations."""
    if vars(cls).get("_qd_fixed_reference_guards", False):
        return

    original_setattr = getattr(cls, "__setattr__")
    original_delattr = getattr(cls, "__delattr__")

    def guarded_setattr(self: Any, name: str, value: Any) -> None:
        """Reject replacement of fixed references and introduction of new container attributes."""
        names = _fixed_names.get(self)
        if names is not None and not name.startswith("_qd_"):
            if name in names:
                if getattr(self, name) is not value:
                    raise QuadrantsRuntimeTypeError(
                        f"Container reference {type(self).__name__}.{name} is fixed after first kernel use."
                    )
            elif _is_container(value):
                raise QuadrantsRuntimeTypeError(
                    f"Cannot add container reference {type(self).__name__}.{name} after first kernel use."
                )
        original_setattr(self, name, value)

    def guarded_delattr(self: Any, name: str) -> None:
        """Reject deletion of a fixed container reference."""
        names = _fixed_names.get(self)
        if names is not None and name in names:
            raise QuadrantsRuntimeTypeError(
                f"Container reference {type(self).__name__}.{name} is fixed after first kernel use."
            )
        original_delattr(self, name)

    setattr(cls, "__setattr__", guarded_setattr)
    setattr(cls, "__delattr__", guarded_delattr)
    cls._qd_fixed_reference_guards = True


def fix_container_references(owner: Any, fixed_names: frozenset[str]) -> None:
    """Guard the stored references of an owner encountered during alias discovery."""
    if isinstance(owner, tuple) or getattr(getattr(type(owner), "__dataclass_params__", None), "frozen", False):
        return
    try:
        _fixed_names[owner] = fixed_names
    except TypeError as error:
        if fixed_names:
            raise QuadrantsRuntimeTypeError(
                f"{type(owner).__name__} must support weak references to guard its container attributes. "
                "For a slotted dataclass, use weakref_slot=True."
            ) from error
        # Leaf arguments can still run uncached. The alias cache warns about those arguments.
        return
    _install_assignment_guards(type(owner))


def validate_immutable_container(value: Any) -> None:
    """Reject mutable collections under data-oriented attributes, including inside tuples."""
    if isinstance(value, (list, dict, set)):
        raise QuadrantsRuntimeTypeError(
            "Mutable Python containers in runtime container graphs are unsupported; use tuples."
        )
    if isinstance(value, tuple):
        for item in value:
            validate_immutable_container(item)
