import warnings

from quadrants.lang.kernel_impl import real_func as _real_func


def real_func(func):
    warnings.warn(
        "qd.experimental.real_func is deprecated because it is no longer experimental. " "Use qd.real_func instead.",
        DeprecationWarning,
    )
    return _real_func(func)


__all__ = ["real_func"]
