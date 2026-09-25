import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_entry_point_does_not_import_runtime(tmp_path):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from importlib.metadata import entry_points; import sys; "
            # entry_points(group='pytest11') lists installed pytest plugin registrations.
            "plugin = next(ep for ep in entry_points(group='pytest11') if ep.name == 'quadrants'); "
            "plugin.load(); "
            "assert 'quadrants' not in sys.modules; assert 'torch' not in sys.modules",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )


def test_pytest_autoload_does_not_import_runtime(tmp_path):
    (tmp_path / "conftest.py").write_text(
        "import sys\n"
        "def pytest_sessionstart(session):\n"
        "    assert session.config.pluginmanager.hasplugin('quadrants')\n"
        "    assert 'quadrants' not in sys.modules\n"
        "    assert 'torch' not in sys.modules\n"
    )
    (tmp_path / "test_startup.py").write_text("def test_startup():\n    pass\n")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=tmp_path, env=env, check=True)


@pytest.mark.parametrize(
    "has_cov, opt_in, branch, expected_coverage, expected_arc",
    [
        (False, None, False, None, None),
        (True, None, False, "1", "0"),
        (True, None, True, "1", "1"),
        (True, "0", True, "0", None),
        (True, "1", True, "1", "1"),
    ],
)
def test_kernel_coverage_configuration(monkeypatch, has_cov, opt_in, branch, expected_coverage, expected_arc):
    import quadrants_pytest

    monkeypatch.delenv("QD_KERNEL_COVERAGE", raising=False)
    monkeypatch.delenv("_QD_KCOV_ARC", raising=False)
    if opt_in is not None:
        monkeypatch.setenv("QD_KERNEL_COVERAGE", opt_in)
    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(hasplugin=lambda name: has_cov if name == "_cov" else False),
        option=SimpleNamespace(cov_branch=branch),
    )
    quadrants_pytest.pytest_configure(config)
    assert os.environ.get("QD_KERNEL_COVERAGE") == expected_coverage
    assert os.environ.get("_QD_KCOV_ARC") == expected_arc
