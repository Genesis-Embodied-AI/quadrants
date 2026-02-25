# type: ignore

from quadrants._lib import core as _qd_core


def print_scoped_profiler_info():
    """Print time elapsed on the host tasks in a hierarchical format.

    This profiler is automatically on.

    Call function imports from C++ : _qd_core.print_profile_info()

    Example::

            >>> import quadrants as qd
            >>> qd.init(arch=qd.cpu)
            >>> var = qd.field(qd.f32, shape=1)
            >>> @qd.kernel
            >>> def compute():
            >>>     var[0] = 1.0
            >>>     print("Setting var[0] =", var[0])
            >>> compute()
            >>> qd.profiler.print_scoped_profiler_info()
    """
    _qd_core.print_profile_info()


def clear_scoped_profiler_info():
    """Clear profiler's records about time elapsed on the host tasks.

    Call function imports from C++ : _qd_core.clear_profile_info()
    """
    _qd_core.clear_profile_info()


__all__ = ["print_scoped_profiler_info", "clear_scoped_profiler_info"]
