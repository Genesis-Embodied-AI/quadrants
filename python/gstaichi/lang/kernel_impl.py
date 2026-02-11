import inspect
import re
import sys
import typing
from functools import update_wrapper, wraps
from typing import Any, Callable, TypeVar, cast, overload

from gstaichi.lang import impl
from gstaichi.lang.exception import (
    GsTaichiCompilationError,
    GsTaichiRuntimeError,
    GsTaichiSyntaxError,
)
from gstaichi.types.enums import AutodiffMode

from .._test_tools import warnings_helper
from ._gstaichi_callable import BoundGsTaichiCallable, GsTaichiCallable
from .func import Func
from .kernel import Kernel

# Define proxies for fast lookup
_NONE, _REVERSE = (
    AutodiffMode.NONE,
    AutodiffMode.REVERSE,
)


F = TypeVar("F", bound=Callable)


def func(fn: F, is_real_function: bool = False) -> F:
    """Marks a function as callable in GsTaichi-scope.

    This decorator transforms a Python function into a GsTaichi one. GsTaichi
    will JIT compile it into native instructions.

    Args:
        fn (Callable): The Python function to be decorated
        is_real_function (bool): Whether the function is a real function

    Returns:
        Callable: The decorated function

    Example::

        >>> @ti.func
        >>> def foo(x):
        >>>     return x + 2
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     print(foo(40))  # 42
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3 + is_real_function)

    fun = Func(fn, _classfunc=is_classfunc, is_real_function=is_real_function)
    gstaichi_callable = GsTaichiCallable(fn, fun)
    gstaichi_callable._is_gstaichi_function = True
    gstaichi_callable._is_real_function = is_real_function

    update_wrapper(gstaichi_callable, fn)
    return cast(F, gstaichi_callable)


def real_func(fn: Callable) -> GsTaichiCallable:
    return func(fn, is_real_function=True)  # type: ignore


def pyfunc(fn: Callable) -> GsTaichiCallable:
    """Marks a function as callable in both GsTaichi and Python scopes.

    When called inside the GsTaichi scope, GsTaichi will JIT compile it into
    native instructions. Otherwise it will be invoked directly as a
    Python function.

    See also :func:`~gstaichi.lang.kernel_impl.func`.

    Args:
        fn (Callable): The Python function to be decorated

    Returns:
        Callable: The decorated function
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3)
    fun = Func(fn, _classfunc=is_classfunc, _pyfunc=True)
    gstaichi_callable = GsTaichiCallable(fn, fun)
    gstaichi_callable._is_gstaichi_function = True
    gstaichi_callable._is_real_function = False
    return gstaichi_callable


# For a GsTaichi class definition like below:
#
# @ti.data_oriented
# class X:
#   @ti.kernel
#   def foo(self):
#     ...
#
# When ti.kernel runs, the stackframe's |code_context| of Python 3.8(+) is
# different from that of Python 3.7 and below. In 3.8+, it is 'class X:',
# whereas in <=3.7, it is '@ti.data_oriented'. More interestingly, if the class
# inherits, i.e. class X(object):, then in both versions, |code_context| is
# 'class X(object):'...
_KERNEL_CLASS_STACKFRAME_STMT_RES = [
    re.compile(r"@(\w+\.)?data_oriented"),
    re.compile(r"class "),
]


def _inside_class(level_of_class_stackframe: int) -> bool:
    try:
        maybe_class_frame = sys._getframe(level_of_class_stackframe)
        statement_list = inspect.getframeinfo(maybe_class_frame)[3]
        if statement_list is None:
            return False
        first_statment = statement_list[0].strip()
        for pat in _KERNEL_CLASS_STACKFRAME_STMT_RES:
            if pat.match(first_statment):
                return True
    except:
        pass
    return False


def _kernel_impl(_func: Callable, level_of_class_stackframe: int, verbose: bool = False) -> GsTaichiCallable:
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(level_of_class_stackframe + 1)

    if verbose:
        print(f"kernel={_func.__name__} is_classkernel={is_classkernel}")
    primal = Kernel(_func, autodiff_mode=_NONE, _is_classkernel=is_classkernel)
    adjoint = Kernel(_func, autodiff_mode=_REVERSE, _is_classkernel=is_classkernel)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    @wraps(_func)
    def wrapped_func(*args, **kwargs):
        try:
            return primal(*args, **kwargs)
        except (GsTaichiCompilationError, GsTaichiRuntimeError) as e:
            if impl.get_runtime().print_full_traceback:
                raise e
            raise type(e)("\n" + str(e)) from None

    wrapped: GsTaichiCallable
    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed when the kernel is accessed via the
        # instance inside _BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance owning the kernel, which is not known
        # until the kernel is accessed.
        # See also: _BoundedDifferentiableMethod, data_oriented.
        @wraps(_func)
        def wrapped_classkernel(*args, **kwargs):
            if args and not getattr(args[0], "_data_oriented", False):
                raise GsTaichiSyntaxError(f"Please decorate class {type(args[0]).__name__} with @ti.data_oriented")
            return wrapped_func(*args, **kwargs)

        wrapped = GsTaichiCallable(_func, wrapped_classkernel)
    else:
        wrapped = GsTaichiCallable(_func, wrapped_func)
        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    primal.gstaichi_callable = wrapped
    return wrapped


@overload
# TODO: This callable should be Callable[[F], F].
# See comments below.
def kernel(_fn: None = None, *, pure: bool = False) -> Callable[[Any], Any]: ...


# TODO: This next overload should return F, but currently that will cause issues
# with ndarray type. We need to migrate ndarray type to be basically
# the actual Ndarray, with Generic types, rather than some other
# NdarrayType class. The _fn should also be F by the way.
# However, by making it return Any, we can make the pure parameter
# change now, without breaking pyright.
@overload
def kernel(_fn: Any, *, pure: bool = False) -> Any: ...


def kernel(_fn: Callable[..., typing.Any] | None = None, *, pure: bool | None = None, fastcache: bool = False):
    """
    Marks a function as a GsTaichi kernel.

    A GsTaichi kernel is a function written in Python, and gets JIT compiled by
    GsTaichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
    The top-level ``for`` loops are automatically parallelized, and distributed
    to either a CPU thread pool or massively parallel GPUs.

    Kernel's gradient kernel would be generated automatically by the AutoDiff system.

    Example::

        >>> x = ti.field(ti.i32, shape=(4, 8))
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     # Assigns all the elements of `x` in parallel.
        >>>     for i in x:
        >>>         x[i] = i
    """

    def decorator(fn: F, has_kernel_params: bool = True) -> F:
        # Adjust stack frame: +1 if called via decorator factory (@kernel()), else as-is (@kernel)
        if has_kernel_params:
            level = 3
        else:
            level = 4

        wrapped = _kernel_impl(fn, level_of_class_stackframe=level)
        wrapped.is_pure = pure is not None and pure or fastcache
        if pure is not None:
            warnings_helper.warn_once(
                "@ti.kernel parameter `pure` is deprecated. Please use parameter `fastcache`. "
                "`pure` parameter is intended to be removed in 4.0.0"
            )

        update_wrapper(wrapped, fn)
        return cast(F, wrapped)

    if _fn is None:
        # Called with @kernel() or @kernel(foo="bar")
        return decorator

    return decorator(_fn, has_kernel_params=False)


class _BoundedDifferentiableMethod:
    def __init__(self, kernel_owner: Any, wrapped_kernel_func: GsTaichiCallable | BoundGsTaichiCallable):
        clsobj = type(kernel_owner)
        if not getattr(clsobj, "_data_oriented", False):
            raise GsTaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint
        self.__name__: str | None = None

    def __call__(self, *args, **kwargs):
        try:
            assert self._primal is not None
            return self._primal(self._kernel_owner, *args, **kwargs)
        except (GsTaichiCompilationError, GsTaichiRuntimeError) as e:
            if impl.get_runtime().print_full_traceback:
                raise e
            raise type(e)("\n" + str(e)) from None

    def grad(self, *args, **kwargs) -> "Kernel":
        assert self._adjoint is not None
        return self._adjoint(self._kernel_owner, *args, **kwargs)


def data_oriented(cls):
    """Marks a class as GsTaichi compatible.

    To allow for modularized code, GsTaichi provides this decorator so that
    GsTaichi kernels can be defined inside a class.

    See also https://docs.taichi-lang.org/docs/odop

    Example::

        >>> @ti.data_oriented
        >>> class TiArray:
        >>>     def __init__(self, n):
        >>>         self.x = ti.field(ti.f32, shape=n)
        >>>
        >>>     @ti.kernel
        >>>     def inc(self):
        >>>         for i in self.x:
        >>>             self.x[i] += 1.0
        >>>
        >>> a = TiArray(32)
        >>> a.inc()

    Args:
        cls (Class): the class to be decorated

    Returns:
        The decorated class.
    """

    def make_kernel_indirect(fun, is_property):
        @wraps(fun)
        def _kernel_indirect(self, *args, **kwargs):
            nonlocal fun
            ret = _BoundedDifferentiableMethod(self, fun)
            ret.__name__ = fun.__name__  # type: ignore
            return ret(*args, **kwargs)

        ret = GsTaichiCallable(fun, _kernel_indirect)
        if is_property:
            ret = property(ret)
        return ret

    # Iterate over all the attributes of the class to wrap member kernels in a way to ensure that they will be called
    # through _BoundedDifferentiableMethod. This extra layer of indirection is necessary to transparently forward the
    # owning instance to the primal function and its adjoint for auto-differentiation gradient computation.
    # There is a special treatment for properties, as they may actually hide kernels under the hood. In such a case,
    # the underlying function is extracted, wrapped as any member function, then wrapped again as a new property.
    # Note that all the other attributes can be left untouched.
    for name, attr in cls.__dict__.items():
        attr_type = type(attr)
        is_property = attr_type is property
        fun = attr.fget if is_property else attr
        if isinstance(fun, (BoundGsTaichiCallable, GsTaichiCallable)):
            if fun._is_wrapped_kernel:
                if fun._is_classkernel and attr_type is not staticmethod:
                    setattr(cls, name, make_kernel_indirect(fun, is_property))
    cls._data_oriented = True

    return cls


__all__ = ["data_oriented", "func", "kernel", "pyfunc", "real_func"]
