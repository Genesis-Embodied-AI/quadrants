import gc
import os
import sys
import tempfile

import pytest

import quadrants as qd


@pytest.fixture(scope="session", autouse=True)
def _offline_cache_dir(tmp_path_factory):
    """Enable the kernel compilation disk cache for the test session.

    Uses pytest's tmp_path_factory so the cache directory is managed by pytest's retention policy
    (tmp_path_retention_count / tmp_path_retention_policy) and cleaned up automatically. This avoids recompiling
    identical kernels after each qd.reset()/qd.init() cycle within a session.
    """
    cache_dir = tmp_path_factory.mktemp("qdcache")
    os.environ["QD_OFFLINE_CACHE"] = "1"
    os.environ["QD_OFFLINE_CACHE_FILE_PATH"] = str(cache_dir)
    os.environ.setdefault("QD_OFFLINE_CACHE_CLEANING_POLICY", "never")


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
def wanted_arch(request, req_arch, req_options, req_extensions):
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

        # Extension support can depend on the capabilities of the device that `qd.init` just created, so the test
        # requirements resolve against the initialized program rather than a static arch table.
        for ext in req_extensions:
            if not qd.is_extension_enabled(ext):
                qd.reset()
                pytest.skip(f"Extension '{ext.name}' is unsupported by the '{req_arch.name}' device on this machine.")
    yield
    if req_arch is not None:
        qd.reset()


def pytest_generate_tests(metafunc):
    if not getattr(metafunc.function, "__qd_test__", False):
        # For test functions not wrapped with @test_utils.test(),
        # fill with empty values to avoid undefined fixtures
        metafunc.parametrize("req_arch,req_options,req_extensions", [(None, None, ())], ids=["none"])


def _exit_marker_dir():
    """Temp directory shared between xdist controller and workers for intentional-exit markers."""
    return os.environ.get("_QD_XDIST_EXIT_MARKER_DIR")


def pytest_configure(config):
    """On the xdist controller, create a temp directory for intentional-exit markers.

    Workers inherit the ``_QD_XDIST_EXIT_MARKER_DIR`` env var and use the same directory.
    """
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    if os.environ.get("_QD_XDIST_EXIT_MARKER_DIR"):
        return
    d = os.path.join(tempfile.gettempdir(), f"qd_xdist_exits_{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    os.environ["_QD_XDIST_EXIT_MARKER_DIR"] = d


def pytest_unconfigure(config):
    """Clean up the marker directory at session end."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    d = _exit_marker_dir()
    if d and os.path.isdir(d):
        import shutil

        shutil.rmtree(d, ignore_errors=True)
    os.environ.pop("_QD_XDIST_EXIT_MARKER_DIR", None)


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Kill the xdist worker process after a test failure so it restarts with clean GPU state.

    Runs trylast so xdist's own hook sends the real test report over the channel first.  Before exiting, we write a
    marker file so the controller's pytest_handlecrashitem can distinguish this intentional exit from a genuine crash
    (segfault, OOM, etc.).
    """
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return

    if report.outcome not in ("error", "failed"):
        return

    d = _exit_marker_dir()
    if d:
        worker_id = os.environ["PYTEST_XDIST_WORKER"]
        try:
            with open(os.path.join(d, worker_id), "w") as f:
                f.write(report.nodeid)
        except OSError:
            pass
    os._exit(1)


def pytest_handlecrashitem(crashitem, report, sched):
    """Suppress the synthetic crash report only for intentional ``os._exit(1)`` exits.

    When a worker is killed intentionally (to reset GPU state after a failure), it writes a marker file before exiting.
    If the marker exists, we mutate the synthetic report so the terminal reporter drops it into the empty-string stats
    bucket (invisible in the summary).  Genuine crashes (segfaults, OOM, etc.) have no marker, so their reports pass
    through as failures.
    """
    d = _exit_marker_dir()
    if not d:
        return
    node = getattr(report, "node", None)
    if not node:
        return
    worker_id = node.gateway.id
    marker = os.path.join(d, worker_id)
    if not os.path.exists(marker):
        return
    try:
        os.unlink(marker)
    except OSError:
        pass
    report.outcome = "passed"
    report.when = "teardown"
    report.longrepr = None
    return True


import importlib


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
