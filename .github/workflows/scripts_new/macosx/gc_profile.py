"""Record GC and test-phase wall/CPU time in each pytest worker without changing collection policy.

Enable with ``-p gc_profile`` and QD_GC_PROFILE_DIR. JSONL records identify the test and phase, generation,
collected/uncollectable counts, process CPU time, load average and peak RSS. No object-graph scans are added.
"""
import gc
import json
import os
from pathlib import Path
import resource
import time

import pytest


_stream = None
_node = None
_phase = 'session'
_start = None
_native_start = None


def _write(record):
    record.update(nodeid=_node, phase=_phase, timestamp=time.time())
    _stream.write(json.dumps(record) + '\n')


def _gc_callback(event, info):
    global _start, _native_start
    if event == 'start':
        if info['generation'] == 2 and os.environ.get('QD_GC_NATIVE') == '1':
            _write(dict(kind='gc_begin', generation=2))
            _native_start = (time.thread_time(), resource.getrusage(resource.RUSAGE_SELF))
        else:
            _native_start = None
        _start = (time.perf_counter(), time.process_time())
    elif event == 'stop' and _start is not None:
        wall, cpu = time.perf_counter() - _start[0], time.process_time() - _start[1]
        native = {}
        if _native_start is not None:
            thread, before = _native_start
            after = resource.getrusage(resource.RUSAGE_SELF)
            native = dict(thread_cpu_seconds=time.thread_time() - thread,
                          resource_delta={name: getattr(after, name) - getattr(before, name)
                                          for name in ('ru_utime', 'ru_stime', 'ru_minflt', 'ru_majflt',
                                                       'ru_nvcsw', 'ru_nivcsw', 'ru_inblock', 'ru_oublock')})
        if info['generation'] == 2 or wall >= 0.1:
            _write(dict(kind='gc', wall_seconds=wall, cpu_seconds=cpu, **info,
                        load1=os.getloadavg()[0], maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        **native))
        _start = None
        _native_start = None


def pytest_configure(config):
    global _stream
    if not os.environ.get('PYTEST_XDIST_WORKER'):
        return
    folder = Path(os.environ['QD_GC_PROFILE_DIR'])
    folder.mkdir(parents=True, exist_ok=True)
    _stream = (folder / f'gc-{os.environ["PYTEST_XDIST_WORKER"]}-{os.getpid()}.jsonl').open('w', buffering=1)
    _write(dict(kind='start', pid=os.getpid(), worker=os.environ['PYTEST_XDIST_WORKER']))
    gc.callbacks.append(_gc_callback)


def pytest_unconfigure(config):
    if _stream is not None:
        gc.callbacks.remove(_gc_callback)
        _write(dict(kind='end', stats=gc.get_stats()))
        _stream.close()


def _measure(item, phase):
    global _node, _phase
    if _stream is None:
        yield
        return
    previous_node, previous_phase = _node, _phase
    _node, _phase = item.nodeid, phase
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        _write(dict(kind='phase', wall_seconds=time.perf_counter() - wall,
                    cpu_seconds=time.process_time() - cpu))
        _node, _phase = previous_node, previous_phase


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    yield from _measure(item, 'setup')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    yield from _measure(item, 'call')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    yield from _measure(item, 'teardown')
