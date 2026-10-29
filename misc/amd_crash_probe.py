"""Experimental pytest diagnostics for the AMD worker exit; not a production plugin."""

import faulthandler
import json
import os
import resource
import time
from pathlib import Path

import pytest

_output = None


def _record(event, **details):
    _output.write(
        json.dumps(
            {
                "time": time.time(),
                "event": event,
                "pid": os.getpid(),
                "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                **details,
            }
        )
        + "\n"
    )
    _output.flush()


def pytest_configure(config):
    global _output
    root = Path(os.environ["AMD_CRASH_RESULTS"])
    root.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    _output = (root / f"{worker}-{os.getpid()}.log").open("w", buffering=1)
    faulthandler.enable(file=_output, all_threads=True)
    _record("configure", cache=os.environ["AMD_CRASH_CACHE"])
    import quadrants as qd

    original_init = qd.init

    def init(*args, **kwargs):
        kwargs["offline_cache"] = os.environ["AMD_CRASH_CACHE"] == "on"
        _record("init", offline_cache=kwargs["offline_cache"])
        return original_init(*args, **kwargs)

    qd.init = init


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    _record("setup_start", test=item.nodeid)
    yield
    _record("setup_end", test=item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _record("call_start", test=item.nodeid)
    yield
    _record("call_end", test=item.nodeid)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    _record("teardown_start", test=item.nodeid)
    yield
    _record("teardown_end", test=item.nodeid)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    _record("worker_down", worker=node.gateway.id, error=str(error))
