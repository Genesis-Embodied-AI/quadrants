import inspect
from typing import Callable

from quadrants.lang.exception import QuadrantsSyntaxError


def get_func_signature(func: Callable) -> inspect.Signature:
    """Call ``inspect.signature`` with ``eval_str=True``.

    ``eval_str=True`` resolves stringified annotations (PEP 563 /
    ``from __future__ import annotations``) to real type objects so downstream
    code can introspect them (e.g. ``dataclasses.is_dataclass``).

    Annotation-evaluation failures (``NameError`` / ``AttributeError`` for
    unresolved references, ``SyntaxError`` for malformed string annotations
    such as ``"NDArray["``) are re-raised as :class:`QuadrantsSyntaxError`
    with the offending function's qualified name, so users get a
    Quadrants-flavored error rather than a raw ``inspect`` traceback.

    Note: ``TypeError`` is intentionally not caught here, since
    ``inspect.signature`` itself raises ``TypeError`` for non-introspectable
    objects -- wrapping that as "invalid type annotation" would be
    misleading.
    """
    try:
        return inspect.signature(func, eval_str=True)
    except (NameError, AttributeError, SyntaxError) as e:
        qualname = getattr(func, "__qualname__", repr(func))
        raise QuadrantsSyntaxError(f"Invalid type annotation in `{qualname}`: {e}") from e
