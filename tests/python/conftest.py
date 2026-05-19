import gc
import os
import random
import sys
import time

import pytest

# rerunfailures use xdist version number to determine if it is compatible
# but we are using a forked version of xdist(with git hash as it's version),
# so we need to override it
import pytest_rerunfailures

import quadrants as qd

pytest_rerunfailures.works_with_current_xdist = lambda: True


# ---------------------------------------------------------------------------
# @pytest.mark.sample(...)  --  per-test stochastic parametrize subsampling
# ---------------------------------------------------------------------------
#
# Some tests parametrize so widely (test_tile16_load_store, test_tile16_cholesky, ...) that running every case on every
# CI run is wasteful: the parametrize axes are intentionally varied to cover corner cases, but most runs would get the
# same signal from a small random subset. ``@pytest.mark.sample(n=...)`` or ``@pytest.mark.sample(fraction=...)`` opts a
# *single* test into per-run random sub-selection. Over many runs, each parametrize case asymptotically gets covered
# (Pr[hit after k runs] = 1 - (1 - keep/total)^k).
#
# Reproducibility hooks:
#   - whole-suite: ``--sample-seed=<S>`` reproduces the exact same trimmed set (header prints the seed used).
#   - single failing case: paste the failing nodeid into ``pytest <nodeid>`` -- the sampler's ``len(group) <= 1``
#     short-circuit keeps it; no flags needed.
#   - exhaustive run (release gate / coverage audit): ``--no-sample`` skips the sampler entirely.
#
# Per-test RNG keyed on ``(seed, nodeid_prefix)``: adding / renaming a @sample-marked test does NOT shift any other
# test's sample. Routine refactors don't migrate failures.


def pytest_addoption(parser):
    parser.addoption(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for @pytest.mark.sample subsampling. If absent, a fresh seed is picked and printed "
        "in the report header so a failing run can be reproduced via --sample-seed=<S>.",
    )
    parser.addoption(
        "--no-sample",
        action="store_true",
        default=False,
        help="Disable @pytest.mark.sample subsampling -- run every parametrize case of every marked test. "
        "Use for exhaustive CI release gates / coverage-debt audits.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    # The marker is registered here (rather than only in pytest.ini) so callers that use
    # `--strict-markers` don't blow up if they happen to import this conftest in isolation.
    config.addinivalue_line(
        "markers",
        "sample(fraction=None, n=None): per-test stochastic parametrize subsampling. Pass exactly one of "
        "`fraction` (0..1) or `n` (>= 1). Seed printed in report header; rerun the same sample with "
        "--sample-seed=<S>; rerun every case with --no-sample; rerun a single failing case by pasting its nodeid.",
    )
    # Seed propagation contract: the seed must reach the controller AND every xdist worker as the same value, or
    # xdist's collection-consistency check fails with "Different tests were collected between gw0 and gwN". argv is
    # forwarded by xdist to every worker, so we require the seed to live on argv as ``--sample-seed=N``. ``tests/
    # run_tests.py`` picks a seed once per run and injects it; direct ``pytest`` invocations either pass
    # ``--sample-seed`` explicitly (reproducibility) or fall back to a single-process seed picked below. We do NOT
    # mutate ``os.environ`` here -- env-var inheritance into xdist worker subprocesses is not guaranteed for runtime
    # mutations, only for vars present when pytest itself was launched.
    if (
        not config.getoption("--no-sample")
        and config.getoption("--sample-seed") is None
        and not hasattr(config, "workerinput")  # single-process / non-xdist controller only.
    ):
        config.option.sample_seed = random.randrange(0, 2**31)


def pytest_report_header(config):
    if config.getoption("--no-sample"):
        return "sample: --no-sample (every @sample-marked test runs every parametrize case)"
    seed = config.getoption("--sample-seed")
    if seed is None:
        return None
    return (
        f"sample-seed={seed}  (reproduce the same sample: --sample-seed={seed}; "
        f"reproduce a single failure: paste its nodeid; run every case: --no-sample)"
    )


def _sample_keep_count(mark, group_size, group_key):
    """Resolve ``@pytest.mark.sample(fraction=..., n=...)`` for a group of ``group_size`` parametrize cases.

    Exactly one of ``fraction`` (0..1) or ``n`` (int >= 1) must be passed; ``UsageError`` otherwise. The result is
    clamped to ``[1, group_size]`` so every @sample-marked test runs at least one case per run (no silent zero-case
    runs even if e.g. ``fraction * group_size`` rounds to zero on a 1-case group).
    """
    fraction = mark.kwargs.get("fraction")
    n = mark.kwargs.get("n")
    if (fraction is None) == (n is None):
        raise pytest.UsageError(
            f"@pytest.mark.sample on {group_key!r}: pass exactly one of `fraction` or `n`, got "
            f"fraction={fraction!r}, n={n!r}"
        )
    if fraction is not None:
        return max(1, int(round(group_size * float(fraction))))
    return max(1, min(int(n), group_size))


def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-sample"):
        return
    seed = config.getoption("--sample-seed")
    sys.stderr.write(
        f"[QD_SAMPLE_DEBUG] pid={os.getpid()} workerinput={hasattr(config, 'workerinput')} "
        f"seed-opt={seed} argv={sys.argv}\n"
    )
    sys.stderr.flush()
    if seed is None:
        # Defensive: pytest_configure didn't run (e.g. someone imported this module manually). Nothing to do.
        return

    # Group items by test function (strip the parametrize bracket suffix). Per-function stratification is what
    # guarantees every @sample-marked test keeps at least one case per run -- uniform sampling across all items
    # could otherwise drop a 2-case marked test entirely.
    groups: dict[str, list] = {}
    for item in items:
        key = item.nodeid.split("[", 1)[0]
        groups.setdefault(key, []).append(item)

    keep, deselected = [], []
    # ``sorted(groups)`` so the iteration order (and therefore any incidental RNG advance) is reproducible across
    # Python versions / dict insertion orders. Per-test RNG is keyed below so this only matters for the (cheap)
    # bookkeeping order.
    for key in sorted(groups):
        group = groups[key]
        mark = group[0].get_closest_marker("sample")
        if mark is None or len(group) <= 1:
            # No sample mark -> every case runs. Also: a single-item group means either the test only had one
            # parametrize case to begin with, or pytest narrowed collection to a specific nodeid -- both cases
            # should run as-is. This is what makes "paste failing nodeid" work without --no-sample.
            keep.extend(group)
            continue
        keep_n = _sample_keep_count(mark, len(group), key)
        # Per-test RNG: keyed on (seed, key) so:
        #   - Independence: adding / renaming / tweaking the @sample mark on test_A does NOT shift the sample of test_B.
        #     Routine refactors don't cause failures to migrate file-wide.
        #   - Locality: when debugging, you can reason about one test's sample without simulating all the others' RNG
        #     advances.
        rng = random.Random((seed, key))
        kept_ids = {id(it) for it in rng.sample(group, k=keep_n)}
        for it in group:
            (keep if id(it) in kept_ids else deselected).append(it)

    if deselected:
        # ``pytest_deselected`` is the supported way to report filtered-out items so pytest's summary shows them as
        # deselected (not silently dropped). xdist also forwards this to the controller correctly.
        config.hook.pytest_deselected(items=deselected)
    items[:] = keep


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

    # Don't call interactor.retire() — it uses os._exit(0) which kills
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
