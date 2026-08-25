"""Parsing/validation for the optional (``T | None``) kernel-argument spelling.

Self-contained helper used once by ``FuncBase.check_parameter_annotations`` to unwrap a ``T | None`` /
``typing.Optional[T]`` annotation into ``(inner, optional)``. Kept out of ``_func_base.py`` so this feature adds a
new module rather than growing the central kernel base class. See design.md W4.
"""

import types
import typing

from quadrants._tensor_wrapper import Tensor as _TensorClass
from quadrants.lang.exception import QuadrantsSyntaxError
from quadrants.types import ndarray_type, template

_NoneType = type(None)


def _is_optional_capable(inner) -> bool:
    """Whether ``inner`` is an annotation family that handles an absent (``None``) value.

    Optionality (``T | None``) is only meaningful for the three families in design.md: ``qd.Tensor`` and
    ``qd.types.Template`` accept ``None`` at runtime today, and ``qd.types.NDArray`` accepts the spelling now with
    runtime support deferred to the dual-nature-slot work (W5). Every other annotation family (dataclass, struct,
    matrix, primitive, buffer view, ...) has no absent-value path, so recording it as optional would only defer a
    confusing internal failure to launch time (e.g. ``getattr(None, field.name)`` for a dataclass slot). Reject those
    at registration instead.
    """
    if inner is _TensorClass:
        return True
    if inner is template or isinstance(inner, template):  # qd.types.Template, class or instance
        return True
    # qd.types.NDArray, either the bare class or a subscripted ``NDArray[dtype, ndim]`` instance.
    if inner is ndarray_type.NdarrayType or type(inner) is ndarray_type.NdarrayType:
        return True
    return False


def split_optional(annotation):
    """Normalize a kernel-arg annotation, unwrapping the optional (``T | None`` / ``Optional[T]``) form.

    Returns ``(inner, optional)``. For a non-optional annotation this is ``(annotation, False)``. For the optional
    form it is ``(T, True)`` with the ``None`` stripped, so the caller can run ``inner`` through the existing
    per-family validation unchanged and separately record that the slot accepts ``None``.

    The three annotation families reach this uniformly for the ``|`` spelling: ``qd.Tensor`` and ``qd.types.Template``
    are classes, so ``T | None`` (and ``typing.Optional[T]``) build a real union; ``qd.types.NDArray[...]`` is an
    instance whose ``__or__`` shim yields an ``_OptionalNdarray`` marker. See design.md W4 and quadrants#831.

    Only these three families may be made optional; ``T | None`` on any other family is rejected here rather than
    silently recording an optional slot that has no absent-value handling (see ``_is_optional_capable``).
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
