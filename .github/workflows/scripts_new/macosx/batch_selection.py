"""Restrict collection before importing test modules, for the fresh-process batching experiment."""
import json
import os
from pathlib import Path


_selected = None


def pytest_configure(config):
    global _selected
    manifest = Path(os.environ['QD_BATCH_SELECTION'])
    _selected = {Path(path) for path in json.loads(manifest.read_text())}
    if not _selected or any(not path.is_file() for path in _selected):
        raise RuntimeError('Batch selection is empty or references missing files')


def pytest_ignore_collect(collection_path, config):
    if collection_path.is_file() and collection_path.suffix in ('.py', '.ipynb'):
        return collection_path.resolve() not in _selected
    return None
