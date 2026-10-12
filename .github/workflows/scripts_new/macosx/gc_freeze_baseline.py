"""Freeze the import/collection object graph while preserving full GC for objects created by tests.

This diagnostic refuses to freeze an initialized runtime or any registered ndarray. Frozen objects are returned
to normal collection at session end. Both existing per-test gc.collect() calls remain unchanged.
"""
import gc
import json
import os
from pathlib import Path
import time

import pytest


_frozen = False
_metadata = {}


def _save():
    folder = Path(os.environ['QD_GC_PROFILE_DIR'])
    folder.mkdir(parents=True, exist_ok=True)
    name = f'freeze-{os.environ["PYTEST_XDIST_WORKER"]}-{os.getpid()}.json'
    (folder / name).write_text(json.dumps(_metadata, indent=2) + '\n')


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    global _frozen
    if not os.environ.get('PYTEST_XDIST_WORKER'):
        return
    from quadrants.lang import impl

    if gc.get_freeze_count():
        raise RuntimeError('Another component has already frozen GC objects')
    gc.collect()
    gc.collect()
    runtime = impl.get_runtime()
    if runtime._prog is not None or len(runtime.ndarrays):
        raise RuntimeError('Cannot freeze a baseline containing an initialized runtime or ndarrays')
    gc.freeze()
    _frozen = True
    _metadata.update(worker=os.environ['PYTEST_XDIST_WORKER'], pid=os.getpid(),
                     frozen_objects=gc.get_freeze_count(), collection_items=len(session.items),
                     automatic_gc_enabled=gc.isenabled(), started_at=time.time())
    _save()


def pytest_sessionfinish(session, exitstatus):
    global _frozen
    if not _frozen:
        return
    start = time.perf_counter()
    gc.unfreeze()
    _frozen = False
    collected = [gc.collect(), gc.collect()]
    _metadata.update(frozen_objects_after=gc.get_freeze_count(), cleanup_collected=collected,
                     cleanup_wall_seconds=time.perf_counter() - start, exitstatus=int(exitstatus))
    _save()
