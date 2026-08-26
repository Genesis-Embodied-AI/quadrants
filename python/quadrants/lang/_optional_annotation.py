"""Unwrap the optional (``T | None``) spelling of a kernel-argument annotation."""

import types
import typing

from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang.exception import QuadrantsSyntaxError
from quadrants.types import ndarray_type, template

_NoneType = type(None)


class _OptionalAbsent:
    """Spec-key sentinel for an optional ndarray slot passed ``None``.

    A distinct, hashable singleton so the absent specialization gets its own cache key and the arg-declaration path
    recognizes it as "no runtime arg; bind the name to the injected ``None`` template var so the present-only body
    specializes away". Mirrors how a ``qd.Tensor`` slot handles ``None``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<optional-absent>"


OPTIONAL_ABSENT = _OptionalAbsent()


def _is_optional_capable(inner) -> bool:
    """The families with an absent-value path; ``T | None`` on anything else has no runtime meaning."""
    if inner is _TensorClass:
        return True
    if inner is template or isinstance(inner, template):
        return True
    # NDArray annotations are instances (``NDArray[dtype, ndim]``), so accept the bare class too.
    if inner is ndarray_type.NdarrayType or type(inner) is ndarray_type.NdarrayType:
        return True
    return False


def split_optional(annotation):
    """Return ``(inner, optional)``, stripping ``None`` from an optional ``T | None`` annotation.

    Rejecting optionality on unsupported families here surfaces a bad annotation at kernel registration
    instead of deep inside launch.
    """
    if isinstance(annotation, ndarray_type._OptionalNdarray):
        return annotation.inner, True
    if isinstance(annotation, types.UnionType) or typing.get_origin(annotation) is typing.Union:
        args = typing.get_args(annotation)
        non_none = tuple(a for a in args if a is not _NoneType)
        if _NoneType not in args or len(non_none) != 1:
            raise QuadrantsSyntaxError(
                f"Quadrants kernels only support optional annotations of the form 'T | None', got: {annotation}"
            )
        inner = non_none[0]
        if not _is_optional_capable(inner):
            raise QuadrantsSyntaxError(
                "Optional ('T | None') kernel arguments are only supported for qd.Tensor, qd.types.Template and "
                f"qd.types.NDArray, got: {inner}"
            )
        return inner, True
    return annotation, False
