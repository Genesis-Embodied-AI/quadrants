"""Build structural specialization keys for mutable tuple capture graphs."""

import dataclasses

import numpy as np

from quadrants._tensor_wrapper import Tensor
from quadrants.lang._final_dataclass_fields import final_scalar_key
from quadrants.lang._ndarray import Ndarray
from quadrants.lang._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from quadrants.lang._runtime_dataclass import infer_member_type, resolve_dataclass
from quadrants.lang._deferred_dataclass import DeferredMemberType
from quadrants.lang._final_dataclass_fields import is_final_annotation
from quadrants.lang._template_mapper_hotpath import _ExprCxx, _extract_arg
from quadrants.lang.exception import QuadrantsRuntimeTypeError
from quadrants.lang.expr import Expr
from quadrants.lang.field import Field
from quadrants.lang.snode import SNode
from quadrants.lang.util import is_data_oriented, wants_runtime_primitives
from quadrants.types.annotations import template
from quadrants.types.ndarray_type import NdarrayType

_ARRAY_ANNOTATION = NdarrayType()


class TuplePlan:
    def __init__(self, value, name, raise_on_templated_floats):
        self.arrays = []
        self.scalars = []
        self._seen = {}
        self._visiting = set()
        self.name = name
        self.raise_on_templated_floats = raise_on_templated_floats
        self.key = self._describe(value, ())
        del self._seen, self._visiting

    def _describe(self, value, path, runtime_scalar=False):
        if isinstance(value, Tensor):
            value = value._unwrap()
        if value is None:
            return ("none",)
        if isinstance(value, (bool, int, float, np.number, np.bool_)):
            if runtime_scalar:
                dtype = infer_member_type(value, self.name, frozenset())
                self.scalars.append((path, dtype))
                return ("scalar", dtype)
            if self.raise_on_templated_floats and isinstance(value, (float, np.floating)):
                raise ValueError("Floats not allowed as templated types.")
            return ("constant", final_scalar_key(value, live=True))
        if isinstance(value, (str, bytes)):
            return ("constant", type(value), value)
        if isinstance(value, QuadrantsCallable):
            if not value._is_quadrants_function:
                raise QuadrantsRuntimeTypeError(f"Tuple member {self.name}{path} must be a qd.func, not a kernel")
            return ("function", value.wrapper)
        if isinstance(value, BoundQuadrantsCallable):
            return ("bound_function", value.wrapper, self._describe(value.instance, (*path, "instance")))
        if isinstance(value, Field):
            return ("field", value)
        if isinstance(value, (Expr, _ExprCxx, SNode)):
            return ("native", type(value), _extract_arg(False, value, template, self.name))
        identity = id(value)
        if identity in self._visiting:
            raise QuadrantsRuntimeTypeError(f"Tuple member {self.name}{path} contains a cycle")
        if identity in self._seen:
            return ("alias", self._seen[identity])
        self._seen[identity] = len(self._seen)
        if isinstance(value, Ndarray):
            self.arrays.append((path, value))
            return ("array", _extract_arg(False, value, _ARRAY_ANNOTATION, self.name))
        self._visiting.add(identity)
        try:
            if isinstance(value, tuple):
                return ("tuple", type(value), tuple(self._describe(child, (*path, i)) for i, child in enumerate(value)))
            if hasattr(type(value), "__dataclass_fields__"):
                schema = resolve_dataclass(value, self.name)
                return (
                    "dataclass",
                    type(value),
                    tuple(
                        (
                            field.name,
                            (
                                field.type
                                if isinstance(field.type, DeferredMemberType)
                                else self._constant_key(getattr(value, field.name))
                                if is_final_annotation(field.type)
                                else self._describe(
                                    getattr(value, field.name), (*path, field.name), runtime_scalar=True
                                )
                            ),
                        )
                        for field in dataclasses.fields(schema)
                    ),
                )
            if is_data_oriented(value):
                # Bound Quadrants methods are cached in the instance dictionary by the descriptor. They are not
                # independent captures; describing them would walk straight back into the same instance.
                entries = tuple(
                    (key, child)
                    for key, child in vars(value).items()
                    if not key.startswith("_qd_") and not isinstance(getattr(type(value), key, None), QuadrantsCallable)
                )
                runtime_primitives = wants_runtime_primitives(value)
                return (
                    "object",
                    type(value),
                    tuple((key, self._describe(child, (*path, key), runtime_primitives)) for key, child in entries),
                )
        finally:
            self._visiting.remove(identity)
        raise QuadrantsRuntimeTypeError(
            f"Tuple member {self.name}{path} has unsupported runtime type {type(value).__name__}"
        )

    def _constant_key(self, value):
        if self.raise_on_templated_floats and isinstance(value, (float, np.floating)):
            raise ValueError("Floats not allowed as templated types.")
        return final_scalar_key(value, live=True)
