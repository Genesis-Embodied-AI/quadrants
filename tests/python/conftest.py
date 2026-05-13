import gc
import os
import sys
import time

import pytest

# rerunfailures use xdist version number to determine if it is compatible
# but we are using a forked version of xdist(with git hash as it's version),
# so we need to override it
import pytest_rerunfailures

import quadrants as qd

pytest_rerunfailures.works_with_current_xdist = lambda: True


@pytest.fixture(autouse=True)
def run_gc_after_test():
    """
    This is necessary to prevent random test failures when testing with ndarray.

    ndarray comprises two separate objects:
    - a c++ side ndarray, which then links the actual data, and contains metadata around
      the shape, and so on
      - let's call this 'ndarray-cpp'
    - a python side ndarray object, that represents the c++ side ndarray, from pybind11
      - let's call this 'ndarray-pybind'
    - a python side ndarray object, that is created independently of the pybind11-created
      python side ndarray object
      - let's call this 'ndarray-py'

    pybind11 is configured such that ownership of ndarray-cpp is NOT passed to the python side

    However, pybind-py has a __del__ method on it, which is called when pybind-py is garbage-
    collected
    - when pybind-py __del__ is called, it calls a c++ method, via pybind, to delete the
      underling ndarray-cpp

    When qd.init() or similar is called, during tests, ndarray-cpp is no longer considered allocated
    - however ndarray-cp has not yet been garbage collected, still exists, and still has a pointer
      to where the ndarray-cpp used to be
    - on mac os x, it regularly happens, as an artifact of how memory management works, that new
      ndarray-cpps are allocated with the exact same address as the old one
    - when garbage collection runs, __del__ is called on the old ndarray-py
        - causing the new ndarray-cpp to be deleted
        - at this point => crash bug

    By calling gc.collect after each test, we avoid this issue.
    """
    yield
    gc.collect()
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def _vulkan_debug_warmup():
    """Prime the Vulkan debugPrintf callback pipeline once per worker process.

    The validation layer's debugPrintf callback delivery has a race condition on the first kernel dispatch in a new
    process: vkQueueWaitIdle() can return before the callback fires. A full init/dispatch/sync/reset cycle here warms
    the driver-level debug infrastructure so subsequent inits work reliably.
    """
    if sys.platform == "darwin":
        return

    from tests import test_utils

    if qd.vulkan not in test_utils.expected_archs():
        return

    try:
        qd.init(arch=qd.vulkan, debug=True, enable_fallback=False, print_full_traceback=True)

        @qd.kernel
        def _warmup() -> qd.i8:
            return qd.i8(64) + qd.i8(64)

        _warmup()
        qd.sync()
        sys.stdout.flush()
    except Exception:
        pass
    finally:
        try:
            qd.reset()
        except Exception:
            pass


@pytest.fixture(autouse=True)
def wanted_arch(request, req_arch, req_options):
    if req_arch is not None:
        if req_arch in (qd.cuda, qd.amdgpu):
            if not request.node.get_closest_marker("run_in_serial"):
                # Optimization only apply to non-serial tests, since serial tests
                # are picked out exactly because of extensive resource consumption.
                # Separation of serial/non-serial tests is done by the test runner
                # through `-m run_in_serial` / `-m not run_in_serial`.
                req_options = {"device_memory_GB": 0.3, **req_options}
                if req_arch == qd.cuda:
                    req_options = {"cuda_stack_limit": 1024, **req_options}
            else:
                # Serial tests run without aggressive resource optimization
                req_options = {"device_memory_GB": 1, **req_options}
        if "print_full_traceback" not in req_options:
            req_options["print_full_traceback"] = True
        qd.init(arch=req_arch, enable_fallback=False, **req_options)
    yield
    if req_arch is not None:
        qd.reset()


def pytest_generate_tests(metafunc):
    if not getattr(metafunc.function, "__qd_test__", False):
        # For test functions not wrapped with @test_utils.test(),
        # fill with empty values to avoid undefined fixtures
        metafunc.parametrize("req_arch,req_options", [(None, None)], ids=["none"])


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """
    Retire test workers when a test fails, to avoid the failing test
    leaving a corrupted GPU state for the following tests.
    """

    interactor = getattr(sys, "xdist_interactor", None)
    if not interactor:
        return

    if report.outcome not in ("rerun", "error", "failed"):
        return

    layoff = False

    chain = getattr(getattr(report, "longrepr", None), "chain", None)
    if chain:
        for _, loc, _ in chain:
            msg = getattr(loc, "message", "") if loc else ""
            if "CUDA_ERROR_OUT_OF_MEMORY" in msg:
                layoff = True
                break

    # Don't call interactor.retire() - it uses os._exit(0) which kills
    # the process before execnet's IO thread can flush the channel buffer.
    # The test failure report (queued by xdist's own hook, which ran before
    # this trylast hook) would be lost, hiding all error messages.
    interactor.sendevent("workerretire", layoff=layoff)
    time.sleep(0.2)
    os._exit(0)


import importlib
import sys

import pytest


@pytest.fixture
def temporary_module():
    """
    Fixture to import and then unload a module after test

    Use like:

    def test_with_temporary_module(temporary_module):
        module = temporary_module('your_module')
    """
    modules_to_delete = []

    def _import(module_name):
        assert module_name not in sys.modules
        mod = importlib.import_module(module_name)
        modules_to_delete.append(module_name)
        return mod

    yield _import

    for name in modules_to_delete:
        del sys.modules[name]
