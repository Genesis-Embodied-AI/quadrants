import inspect
import os
import time
from collections import defaultdict
from typing import Any, Callable, Generic, ParamSpec, Type, TypeVar

from .. import _logging
from . import impl
from ._exceptions import raise_exception
from ._quadrants_callable import QuadrantsCallable
from .exception import QuadrantsRuntimeError, QuadrantsSyntaxError

NUM_WARMUP: int = 3
NUM_ACTIVE: int = 1
REPEAT_AFTER_COUNT: int = 0
REPEAT_AFTER_SECONDS: float = 1.0

TI_PERFDISPATCH_PRINT_DEBUG = os.environ.get("TI_PERFDISPATCH_PRINT_DEBUG", "0") == "1"


class DispatchImpl:
    def __init__(self, implementation1: Callable | QuadrantsCallable, is_compatible: Callable | None) -> None:
        """
        - underlying1 might be the actual python function, or it might be a python fucntion wrapped in a
        QuadrantsCallable or not.
        - underlying2 should always be the actual python function.
        """
        self.is_compatible: Callable | None = is_compatible
        self.__wrapped__: Callable = implementation1
        self._wrapped_type = type(implementation1)
        if self._wrapped_type is QuadrantsCallable:
            self.implementation2 = implementation1.fn  # type: ignore
        else:
            self.implementation2 = implementation1

    def __call__(self, *args, **kwargs) -> Any:
        return self.__wrapped__(*args, **kwargs)

    def get_implementation2(self) -> Callable:
        return self.implementation2


P = ParamSpec("P")
R = TypeVar("R")


class PerformanceDispatcher(Generic[P, R]):
    def __init__(
        self,
        get_geometry_hash: Callable[P, int],
        fn: Callable,
        num_warmup: int | None = None,
        num_active: int | None = None,
        repeat_after_count: int | None = None,
        repeat_after_seconds: float | None = None,
    ) -> None:
        self.num_warmup = num_warmup if num_warmup is not None else NUM_WARMUP
        self.num_active = num_active if num_active is not None else NUM_ACTIVE
        self.repeat_after_count = repeat_after_count if repeat_after_count is not None else REPEAT_AFTER_COUNT
        self.repeat_after_seconds = repeat_after_seconds if repeat_after_seconds is not None else REPEAT_AFTER_SECONDS
        sig = inspect.signature(fn)
        self._param_types: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            self._param_types[param_name] = param.annotation
        self._get_geometry_hash: Callable[P, int] = get_geometry_hash
        self._dispatch_impl_set: set[DispatchImpl] = set()
        self._trial_count_by_dispatch_impl_by_geometry_hash: dict[int, dict[DispatchImpl, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._fastest_dispatch_impl_by_geometry_hash: dict[int, DispatchImpl | None] = defaultdict(None)
        self._times_by_dispatch_impl_by_geometry_hash: dict[int, dict[DispatchImpl, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._calls_since_last_update_by_geometry_hash: dict[int, int] = defaultdict(int)
        self._last_check_time_by_geometry_hash: dict[int, float] = defaultdict(float)

    def register(
        self, implementation: Callable | None = None, *, is_compatible: Callable[[dict], bool] | None = None
    ) -> Callable[[Callable], Callable] | Type[DispatchImpl]:
        """
        Use register to register a function with a @qd.perf_dispatch meta function

        See @qd.perf_dispatch for documentation about using @qd.perf_dispatch meta function

        is_compatible is an optional function that will return whether the function being registered can
        run on the specific arguments being passed in. If there are circumstances where this function being
        registered cannot run, then is_compatible MUST be implemented, and MUST return False given the specific arguments
        or platform.

        is_compatible receives the exact same *args and **kwargs that were used to call the meta function.

        Examples of where you might need to implement is_compatible:
        - the function only runs on Metal => is_compatible should return False on any platform where Metal is not
          available (typically, any non-Darwin machine for example)
        - the function only runs for certain ranges of dimensions on one or more of the input arguments
            - in this case, check the shape of the argument in question, and return False if out of spec for this
              implementation
        """
        dispatch_impl_set = self._dispatch_impl_set

        def decorator(func: Callable | QuadrantsCallable) -> DispatchImpl:
            sig = inspect.signature(func)
            log_str = f"perf_dispatch registering {func.__name__}"  # type: ignore
            _logging.debug(log_str)
            if TI_PERFDISPATCH_PRINT_DEBUG:
                print(log_str)
            for param_name, _param in sig.parameters.items():
                if param_name not in self._param_types:
                    raise_exception(
                        QuadrantsSyntaxError,
                        msg=f"Signature parameter {param_name} of function not in perf_dispatch function prototype",
                        err_code="PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH",
                    )
            if len(sig.parameters) != len(self._param_types):
                raise_exception(
                    QuadrantsSyntaxError,
                    msg=f"Number of function parameters {len(sig.parameters)} doesn't match number of parameters in perf_dispatch function prototype {len(self._param_types)}",
                    err_code="PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH",
                )

            dispatch_impl = DispatchImpl(implementation1=func, is_compatible=is_compatible)
            dispatch_impl_set.add(dispatch_impl)
            return dispatch_impl

        if implementation is not None:
            return decorator(implementation)
        return decorator

    def _get_compatible_functions(self, *args, **kwargs) -> set[DispatchImpl]:
        compatible_set = set()
        for dispatch_impl in self._dispatch_impl_set:
            if dispatch_impl.is_compatible and not dispatch_impl.is_compatible(*args, **kwargs):
                continue
            compatible_set.add(dispatch_impl)
        return compatible_set

    def _get_next_dispatch_impl(
        self, compatible_set: set[DispatchImpl], geometry_hash: int
    ) -> tuple[int, DispatchImpl]:
        least_trials_dispatch_impl = None
        least_trials = None
        for dispatch_impl in compatible_set:
            trial_count = self._trial_count_by_dispatch_impl_by_geometry_hash[geometry_hash].get(dispatch_impl, 0)
            if least_trials is None or trial_count < least_trials:
                least_trials_dispatch_impl = dispatch_impl
                least_trials = trial_count
        assert least_trials_dispatch_impl is not None and least_trials is not None
        return least_trials, least_trials_dispatch_impl

    def _get_min_trials_finished(self, geometry_hash: int) -> int:
        return min(self._trial_count_by_dispatch_impl_by_geometry_hash[geometry_hash].values())

    def _compute_are_trials_finished(self, geometry_hash: int) -> bool:
        if len(self._trial_count_by_dispatch_impl_by_geometry_hash[geometry_hash]) == 0:
            return False

        min_trials = min(self._trial_count_by_dispatch_impl_by_geometry_hash[geometry_hash].values())
        res = min_trials >= self.num_warmup + self.num_active
        return res

    def _compute_and_update_fastest(self, geometry_hash: int) -> None:
        times_by_dispatch_impl = self._times_by_dispatch_impl_by_geometry_hash[geometry_hash]
        fastest_dispatch, _ = min(times_by_dispatch_impl.items(), key=lambda x: x[1])
        self._fastest_dispatch_impl_by_geometry_hash[geometry_hash] = fastest_dispatch
        underlying = fastest_dispatch.get_implementation2()
        log_str = (
            f"perf dispatch chose {underlying.__name__} out of {len(self._dispatch_impl_set)} registered functions."
        )
        _logging.debug(log_str)
        if TI_PERFDISPATCH_PRINT_DEBUG:
            print(log_str)

    def __call__(self, *args: P.args, **kwargs: P.kwargs):
        """
        We are going to run each function self.num_warmup times, to warm up, then run them each again,
        then choose the fastest function, based on the time of the last run.

        Each function must have identical behavior, including for side-effects.

        We only run a single function per call, so functions don't need to be idempotent.

        We call sync before and after, because kernels run async, so:
        - if we didn't sync after, we'd measure the time to queue the function, without waiting for it to finish.
        - if we didn't sync before, we'd be measuring also the time for all the existing gpu function that
          have already been queued up, are processing. So we sync to make sure those have finished first.

        We collect a single sample from each implementation, and compare that single sample with the samples from the
        other implementations.

        We are comparing algorithms based on empirical runtime.

        Note that for best results, sets of input arguments that have different runtimes should map to different
        geometries, otherwise the comparison between runtimes might not be fair, and an inappropriate implementation
        function might be selected.

        We are not implementing an epsilon-greedy algorithm to keep sampling non-fastest variants just in case the
        distribution is shifting over time.

        It is not possible for you to control exploration vs exploitation.
        """
        geometry_hash = self._get_geometry_hash(*args, **kwargs)
        fastest = self._fastest_dispatch_impl_by_geometry_hash.get(geometry_hash)
        if fastest:
            restart_measurements = False
            if self.repeat_after_count > 0:
                self._calls_since_last_update_by_geometry_hash[geometry_hash] += 1
                calls = self._calls_since_last_update_by_geometry_hash[geometry_hash]
                if calls >= self.repeat_after_count:
                    restart_measurements = True
            if self.repeat_after_seconds > 0:
                elapsed = time.time() - self._last_check_time_by_geometry_hash[geometry_hash]
                if elapsed >= self.repeat_after_seconds:
                    restart_measurements = True
            if not restart_measurements:
                return fastest(*args, **kwargs)

            self._times_by_dispatch_impl_by_geometry_hash[geometry_hash].clear()
            self._trial_count_by_dispatch_impl_by_geometry_hash[geometry_hash].clear()
            del self._fastest_dispatch_impl_by_geometry_hash[geometry_hash]
            self._calls_since_last_update_by_geometry_hash[geometry_hash] = 0

        res = None
        runtime = impl.get_runtime()
        compatible_set = self._get_compatible_functions(*args, **kwargs)
        if len(compatible_set) == 0:
            raise QuadrantsRuntimeError("No suitable functions were found.")

        elif len(compatible_set) == 1:
            self._fastest_dispatch_impl_by_geometry_hash[geometry_hash] = next(iter(compatible_set))
            self._last_check_time_by_geometry_hash[geometry_hash] = time.time()
            dispatch_impl_ = self._fastest_dispatch_impl_by_geometry_hash[geometry_hash]
            assert dispatch_impl_ is not None
            log_str = (
                f"perf dispatch chose {dispatch_impl_.get_implementation2().__name__} "
                f"out of {len(self._dispatch_impl_set)} registered functions. Only 1 was compatible."
            )
            _logging.debug(log_str)
            if TI_PERFDISPATCH_PRINT_DEBUG:
                print(log_str)
            return dispatch_impl_(*args, **kwargs)

        min_trial_count, dispatch_impl = self._get_next_dispatch_impl(
            compatible_set=compatible_set, geometry_hash=geometry_hash
        )
        trial_count_by_dispatch_impl = self._trial_count_by_dispatch_impl_by_geometry_hash[geometry_hash]
        trial_count_by_dispatch_impl[dispatch_impl] += 1
        in_warmup = min_trial_count < self.num_warmup
        start = 0
        if not in_warmup:
            runtime.sync()
            start = time.time()
        res = dispatch_impl(*args, **kwargs)
        if not in_warmup:
            runtime.sync()
            end = time.time()
            elapsed = end - start
            self._times_by_dispatch_impl_by_geometry_hash[geometry_hash][dispatch_impl].append(elapsed)
            if self._compute_are_trials_finished(geometry_hash=geometry_hash):
                self._compute_and_update_fastest(geometry_hash)
                self._last_check_time_by_geometry_hash[geometry_hash] = time.time()
        return res


def perf_dispatch(
    *,
    get_geometry_hash: Callable,
    warmup: int = NUM_WARMUP,
    active: int = NUM_ACTIVE,
    repeat_after_count: int = REPEAT_AFTER_COUNT,
    repeat_after_seconds: float = REPEAT_AFTER_SECONDS,
):
    """
    This annotation designates a meta-function that can have one or more functions registered with it.

    At runtime, gstaichi will try running each registered function in turn, and choose the fastest. Once
    chosen, the fastest function will systematically be used, for the lifetime of the process. This is
    aimed for use where there are multiple possible functions, and no clear heuristic to
    choose between them.

    Args:
        get_geometry_hash: A function that returns a geometry hash given the arguments.
        warmup: Number of warmup iterations to run for each implementation before measuring. Default 3.
        active: Number of active (timed) iterations to run for each implementation. Default 1.
        repeat_after_count: repeats the cycle of warmup and active from scratch after repeat_after_count
        additional calls.
        repeat_after_seconds: repeats the cycle of warmup and active from scratch after repeat_after_seconds
        seconds elapsed.

    Example usage:

    @qd.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape))
    def my_func1(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]): ...

    @qd.perf_dispatch(get_geometry_hash=lambda a, c: hash(a.shape + c.shape), warmup=5, active=2)
    def my_func2(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]): ...
        # note: this is intentionally empty. The function body will NEVER be called.

    @my_func1.register
    @qd.kernel
    def my_func1_impl1(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]) -> None:
        # implementation 1 here...

    @my_func1.register(is_compatible=lambda a, c: a.shape[0] < 2)
    @qd.kernel
    def my_func1_impl2(a: qd.types.NDArray[qd.i32, 1], c: qd.types.NDArray[qd.i32, 1]) -> None:
        # implementation 2 here...

    Then simply call the meta-function, just like any other function:

    my_func1(a, b)

    Note that the effect of each implementation must be identical, including side effects, otherwise subtle
    and hard to diagnose bugs are likely to occur. @qd.perf_dispatch does NOT check that the implementations have
    identical effects.

    ## Geometry

    Depending on certain characteristics of the input arguments to a call, different implementations might be
    relatively faster or slower. We denote such characteristics the 'geometry' of the call. An example of 'geometry'
    is the stride and padding to a call to a convolutional kernel, as well as the number of channels, the height
    and the width.

    The meta function @qd.perf_dispatch annotation MUST provide a function that returns a geometry hash
    given the arguments.

    You are free to return any valid hash.
    - In the simplest case, you could simply return a constant value, in which case all inputs will be considered to
      have identical 'geometry', and the same implemnetation function will systematically be called
    - Otherwise, if you are aware of key characteristics of the input arguments, then you can return a hash of these
      characteristics here

    Note that it is strongly recommended that any values used to create the geometry hash are NOT retrieved from data
    on the GPU, otherwise you are likely to create a GPU sync point, which would be likely to severely slow down
    performance.

    Examples of geometry could be simply the shapes of all input arguments:

    get_geometry_hash=lambda *args, **kwargs: hash(tuple([arg.shape for arg in args]))

    ### Advanced geometry

    You can simply hash the input arguments directly:

    get_geometry_hash=lambda *args, **kwargs: hash(tuple(*args, frozendict(kwargs)))
    """

    def decorator(fn: Callable | QuadrantsCallable):
        return PerformanceDispatcher(
            get_geometry_hash=get_geometry_hash,
            fn=fn,
            num_warmup=warmup,
            num_active=active,
            repeat_after_count=repeat_after_count,
            repeat_after_seconds=repeat_after_seconds,
        )

    return decorator


__all__ = ["perf_dispatch"]
