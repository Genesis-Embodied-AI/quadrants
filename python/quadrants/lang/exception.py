# type: ignore

from quadrants._lib import core


class QuadrantsCompilationError(Exception):
    """Base class for all compilation exceptions."""

    pass


class QuadrantsSyntaxError(QuadrantsCompilationError, SyntaxError):
    """Thrown when a syntax error is found during compilation."""

    pass


class QuadrantsNameError(QuadrantsCompilationError, NameError):
    """Thrown when an undefine name is found during compilation."""

    pass


class QuadrantsIndexError(QuadrantsCompilationError, IndexError):
    """Thrown when an index error is found during compilation."""

    pass


class QuadrantsTypeError(QuadrantsCompilationError, TypeError):
    """Thrown when a type mismatch is found during compilation."""

    pass


class QuadrantsRuntimeError(RuntimeError):
    """Thrown when the compiled program cannot be executed due to unspecified reasons."""

    pass


class QuadrantsAssertionError(QuadrantsRuntimeError, AssertionError):
    """Thrown when assertion fails at runtime."""

    pass


class QuadrantsRuntimeTypeError(QuadrantsRuntimeError, TypeError):
    @staticmethod
    def get(pos, needed, provided):
        return QuadrantsRuntimeTypeError(
            f"Argument {pos} (type={provided}) cannot be converted into required type {needed}"
        )

    @staticmethod
    def get_ret(needed, provided):
        return QuadrantsRuntimeTypeError(f"Return (type={provided}) cannot be converted into required type {needed}")


def handle_exception_from_cpp(exc):
    if isinstance(exc, core.QuadrantsTypeError):
        return QuadrantsTypeError(str(exc))
    if isinstance(exc, core.QuadrantsSyntaxError):
        return QuadrantsSyntaxError(str(exc))
    if isinstance(exc, core.QuadrantsIndexError):
        return QuadrantsIndexError(str(exc))
    if isinstance(exc, core.QuadrantsAssertionError):
        return QuadrantsAssertionError(str(exc))
    return exc


__all__ = [
    "QuadrantsSyntaxError",
    "QuadrantsTypeError",
    "QuadrantsCompilationError",
    "QuadrantsNameError",
    "QuadrantsRuntimeError",
    "QuadrantsRuntimeTypeError",
    "QuadrantsAssertionError",
]
