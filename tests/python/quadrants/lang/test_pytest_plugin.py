import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

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
        "    # Check that pytest registered the plugin named quadrants, not that the package was imported.\n"
        "    # This prevents the test from passing merely because plugin loading was disabled.\n"
        "    assert session.config.pluginmanager.hasplugin('quadrants')\n"
        "    assert 'quadrants' not in sys.modules\n"
        "    assert 'torch' not in sys.modules\n"
    )
    (tmp_path / "test_startup.py").write_text("def test_startup():\n    pass\n")
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(key, None)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=tmp_path, env=env, check=True)


# Each case specifies:
# has_cov: whether pytest-cov's internal _cov plugin is registered.
# opt_in: initial QD_KERNEL_COVERAGE value (None means unset; "0" disables; "1" enables).
# branch: whether pytest-cov requests branch coverage via --cov-branch.
# expected_coverage: QD_KERNEL_COVERAGE value after the hook (None means unset).
# expected_arc: _QD_KCOV_ARC value after the hook ("1" for branch data, "0" for line data, None for unset).
@pytest.mark.parametrize(
    "has_cov, opt_in, branch, expected_coverage, expected_arc",
    [
        (False, None, False, None, None),
        (True, None, False, "1", "0"),
        (True, None, True, "1", "1"),
        (True, "0", True, "0", None),
        (True, "1", False, "1", "0"),
        (True, "1", True, "1", "1"),
    ],
)
def test_kernel_coverage_configuration(has_cov, opt_in, branch, expected_coverage, expected_arc):
    import quadrants_pytest

    # patch.dict restores os.environ on exit, including variables the hook creates. Otherwise QD_KERNEL_COVERAGE=1 leaks
    # into later tests on this xdist worker and compiles their kernels with coverage probes.
    with mock.patch.dict(os.environ):
        os.environ.pop("QD_KERNEL_COVERAGE", None)
        os.environ.pop("_QD_KCOV_ARC", None)
        if opt_in is not None:
            os.environ["QD_KERNEL_COVERAGE"] = opt_in
        config = SimpleNamespace(
            pluginmanager=SimpleNamespace(hasplugin=lambda name: has_cov if name == "_cov" else False),
            option=SimpleNamespace(cov_branch=branch),
        )
        # Call the startup hook directly with this test configuration to set the kernel coverage environment variables.
        quadrants_pytest.pytest_configure(config)
        assert os.environ.get("QD_KERNEL_COVERAGE") == expected_coverage
        assert os.environ.get("_QD_KCOV_ARC") == expected_arc
