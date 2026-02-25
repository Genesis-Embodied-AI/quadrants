# Must import 'partial' directly instead of the entire module to avoid attribute lookup overhead.
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .kernel import Kernel


class QuadrantsCallable:
    """
    BoundQuadrantsCallable is used to enable wrapping a bindable function with a class.

    Design requirements for QuadrantsCallable:
    - wrap/contain a reference to a class Func instance, and allow (the QuadrantsCallable) being passed around
      like normal function pointer
    - expose attributes of the wrapped class Func, such as `_if_real_function`, `_primal`, etc
    - allow for (now limited) strong typing, and enable type checkers, such as pyright/mypy
        - currently QuadrantsCallable is a shared type used for all functions marked with @qd.func, @qd.kernel,
          python functions (?)
        - note: current type-checking implementation does not distinguish between different type flavors of
          QuadrantsCallable, with different values of `_if_real_function`, `_primal`, etc
    - handle not only class-less functions, but also class-instance methods (where determining the `self`
      reference is a challenge)

    Let's take the following example:

    def test_ptr_class_func():
    @qd.data_oriented
    class MyClass:
        def __init__(self):
            self.a = qd.field(dtype=qd.f32, shape=(3))

        def add2numbers_py(self, x, y):
            return x + y

        @qd.func
        def add2numbers_func(self, x, y):
            return x + y

        @qd.kernel
        def func(self):
            a, add_py, add_func = qd.static(self.a, self.add2numbers_py, self.add2numbers_func)
            a[0] = add_py(2, 3)
            a[1] = add_func(3, 7)

    (taken from test_ptr_assign.py).

    When the @qd.func decorator is parsed, the function `add2numbers_func` exists, but there is not yet any `self`
    - it is not possible for the method to be bound, to a `self` instance
    - however, the @qd.func annotation, runs the kernel_imp.py::func function --- it is at this point
      that Quadrants's original code creates a class Func instance (that wraps the add2numbers_func)
      and immediately we create a QuadrantsCallable instance that wraps the Func instance.
    - effectively, we have two layers of wrapping QuadrantsCallable->Func->function pointer
      (actual function definition)
    - later on, when we call self.add2numbers_py, here:

            a, add_py, add_func = qd.static(self.a, self.add2numbers_py, self.add2numbers_func)

      ... we want to call the bound method, `self.add2numbers_py`.
    - an actual python function reference, created by doing somevar = MyClass.add2numbers, can automatically
      binds to self, when called from self in this way (however, add2numbers_py is actually a class
      Func instance, wrapping python function reference -- now also all wrapped by a QuadrantsCallable
      instance -- returned by the kernel_impl.py::func function, run by @qd.func)
    - however, in order to be able to add strongly typed attributes to the wrapped python function, we need
      to wrap the wrapped python function in a class
    - the wrapped python function, wrapped in a QuadrantsCallable class (which is callable, and will
      execute the underlying double-wrapped python function), will NOT automatically bind
    - when we invoke QuadrantsCallable, the wrapped function is invoked. The wrapped function is unbound, and
      so `self` is not automatically passed in, as an argument, and things break

    To address this we need to use the `__get__` method, in our function wrapper, ie QuadrantsCallable,
    and have the `__get__` method return the `BoundQuadrantsCallable` object. The `__get__` method handles
    running the binding for us, and effectively binds `BoundFunc` object to `self` object, by passing
    in the instance, as an argument into `BoundQuadrantsCallable.__init__`.

    `BoundFunc` can then be used as a normal bound func - even though it's just an object instance -
    using its `__call__` method. Effectively, at the time of actually invoking the underlying python
    function, we have 3 layers of wrapper instances:
        BoundQuadrantsCallabe -> QuadrantsCallable -> Func -> python function reference/definition
    """

    def __init__(self, fn: Callable, wrapper: Callable) -> None:
        self.fn: Callable = fn
        self.wrapper: Callable = wrapper
        self._is_real_function: bool = False
        self._is_quadrants_function: bool = False
        self._is_wrapped_kernel: bool = False
        self._is_classkernel: bool = False
        self._primal: "Kernel | None" = None
        self._adjoint: "Kernel | None" = None
        self.grad: "Kernel | None" = None
        self.is_pure: bool = False
        update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.wrapper.__call__(*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return BoundQuadrantsCallable(instance, self)


class BoundQuadrantsCallable:
    def __init__(self, instance: Any, quadrants_callable: QuadrantsCallable):
        self.wrapper = quadrants_callable.wrapper
        self.instance = instance
        self.quadrants_callable = quadrants_callable

    def __call__(self, *args, **kwargs):
        return self.wrapper(self.instance, *args, **kwargs)

    def __getattr__(self, k: str) -> Any:
        res = getattr(self.quadrants_callable, k)
        return res

    def __setattr__(self, k: str, v: Any) -> None:
        # Note: these have to match the name of any attributes on this class.
        if k in {"wrapper", "instance", "quadrants_callable"}:
            object.__setattr__(self, k, v)
        else:
            setattr(self.quadrants_callable, k, v)

    def grad(self, *args, **kwargs) -> "Kernel":
        assert self.quadrants_callable._adjoint is not None
        return self.quadrants_callable._adjoint(self.instance, *args, **kwargs)
