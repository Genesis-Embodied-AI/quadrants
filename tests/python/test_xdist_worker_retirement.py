"""Tests for the xdist worker retirement hooks in conftest.py.

Verifies that when a worker is killed via os._exit(1) after a test failure: 1. Failures are not double-counted (no
synthetic "worker crashed" report)  2. The session completes even with many failures (--max-worker-restart cap does not
trigger premature shutdown)

These tests use pytester to run pytest-xdist in a subprocess, so they do not require GPU hardware.
"""

import pytest

pytest_plugins = ["pytester"]

SUBPROCESS_ARGS = [
    "-p",
    "no:retry",
    "-p",
    "no:rerunfailures",
    "-p",
    "no:nbmake",
    "-p",
    "no:timeout",
    "-p",
    "no:cacheprovider",
    "-o",
    "addopts=",
]


@pytest.fixture
def xdist_project(pytester, monkeypatch):
    """Write a minimal conftest that reproduces our worker-retirement hooks."""
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    pytester.makeconftest(
        """
        import os
        import tempfile
        import pytest

        def _exit_marker_dir():
            return os.environ.get("_QD_XDIST_EXIT_MARKER_DIR")

        def pytest_configure(config):
            if os.environ.get("PYTEST_XDIST_WORKER"):
                return
            if os.environ.get("_QD_XDIST_EXIT_MARKER_DIR"):
                return
            d = os.path.join(tempfile.gettempdir(), f"qd_xdist_exits_{os.getpid()}")
            os.makedirs(d, exist_ok=True)
            os.environ["_QD_XDIST_EXIT_MARKER_DIR"] = d

        @pytest.hookimpl(trylast=True)
        def pytest_runtest_logreport(report):
            if not os.environ.get("PYTEST_XDIST_WORKER"):
                return
            if report.outcome not in ("error", "failed"):
                return
            d = _exit_marker_dir()
            if d:
                worker_id = os.environ["PYTEST_XDIST_WORKER"]
                try:
                    with open(os.path.join(d, worker_id), "w") as f:
                        f.write(report.nodeid)
                except OSError:
                    pass
            os._exit(1)

        def pytest_handlecrashitem(crashitem, report, sched):
            d = _exit_marker_dir()
            if not d:
                return
            node = getattr(report, "node", None)
            if not node:
                return
            worker_id = node.gateway.id
            marker = os.path.join(d, worker_id)
            if not os.path.exists(marker):
                return
            try:
                os.unlink(marker)
            except OSError:
                pass
            report.outcome = "passed"
            report.when = "teardown"
            report.longrepr = None
            return True
        """
    )
    return pytester


class TestNoDuplicateFailures:
    def test_single_failure_counted_once(self, xdist_project):
        """A single failing test should appear exactly once in the summary."""
        xdist_project.makepyfile(
            """
            def test_pass():
                pass

            def test_fail():
                assert False, "intentional failure"
            """
        )
        result = xdist_project.runpytest_subprocess("-n", "2", "--dist=worksteal", *SUBPROCESS_ARGS, "-v")
        result.assert_outcomes(passed=1, failed=1)

    def test_multiple_failures_counted_correctly(self, xdist_project):
        """Each failing test should be counted exactly once."""
        xdist_project.makepyfile(
            """
            import pytest

            @pytest.mark.parametrize("i", range(4))
            def test_fail(i):
                assert False, f"failure {i}"

            def test_pass():
                pass
            """
        )
        result = xdist_project.runpytest_subprocess(
            "-n",
            "2",
            "--dist=worksteal",
            "--max-worker-restart=999999",
            *SUBPROCESS_ARGS,
            "-v",
        )
        result.assert_outcomes(passed=1, failed=4)


class TestSessionCompletesWithManyFailures:
    def test_no_premature_shutdown(self, xdist_project):
        """With a high --max-worker-restart, all tests should run even if many fail."""
        xdist_project.makepyfile(
            """
            import pytest

            @pytest.mark.parametrize("i", range(20))
            def test_fail(i):
                assert False, f"failure {i}"

            @pytest.mark.parametrize("i", range(5))
            def test_pass(i):
                pass
            """
        )
        result = xdist_project.runpytest_subprocess(
            "-n",
            "2",
            "--dist=worksteal",
            "--max-worker-restart=999999",
            *SUBPROCESS_ARGS,
            "-v",
        )
        result.assert_outcomes(passed=5, failed=20)
        assert "maximum crashed workers reached" not in result.stdout.str()
