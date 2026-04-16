"""Pytest plugin that auto-enables kernel coverage when pytest-cov is active.

Registered via the ``pytest11`` entry point so it loads automatically when quadrants is installed.
Opt out by setting ``QD_KERNEL_COVERAGE=0`` explicitly.
"""

import os


def pytest_configure(config):
    if config.pluginmanager.hasplugin("_cov"):
        os.environ.setdefault("QD_KERNEL_COVERAGE", "1")
