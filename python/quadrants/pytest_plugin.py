"""Pytest plugin that auto-enables kernel coverage when pytest-cov is active.

Registered via the ``pytest11`` entry point so it loads automatically when quadrants is installed.
Opt out by setting ``QD_KERNEL_COVERAGE=0`` explicitly.
"""

import os


def pytest_configure(config):
    if config.pluginmanager.hasplugin("_cov"):
        os.environ.setdefault("QD_KERNEL_COVERAGE", "1")
        # Kernel coverage always writes arc-format data; ensure pytest-cov matches to avoid
        # "Can not mix line and arc data" errors during coverage combine.
        if not config.option.__dict__.get("cov_branch", False):
            config.option.cov_branch = True
