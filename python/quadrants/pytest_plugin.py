"""Pytest plugin that auto-enables kernel coverage when pytest-cov is active.

Registered via the ``pytest11`` entry point so it loads automatically when quadrants is installed.
Opt out by setting ``QD_KERNEL_COVERAGE=0`` explicitly.
"""

import os


def pytest_configure(config):
    if not config.pluginmanager.hasplugin("_cov"):
        return
    os.environ.setdefault("QD_KERNEL_COVERAGE", "1")
    if os.environ.get("QD_KERNEL_COVERAGE") != "1":
        return
    # Tell the kernel coverage module whether pytest-cov is running in branch (arc) mode,
    # so it writes the matching format and avoids "Can not mix line and arc data" at combine time.
    # We read config.option.cov_branch which pytest-cov has already populated by this point.
    cov_branch = getattr(config.option, "cov_branch", False) or False
    os.environ["_QD_KCOV_ARC"] = "1" if cov_branch else "0"
