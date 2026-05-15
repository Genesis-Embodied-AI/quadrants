import argparse
import importlib.util
import os


def _test_python(args, default_dir="python"):
    print("\nRunning Python tests...\n")

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(curr_dir, default_dir)
    pytest_args = []

    # TODO: use pathlib to deal with suffix and stem name manipulation
    if args.files:
        # run individual tests
        for f in args.files:
            # auto-complete file names
            if not f.startswith("test_"):
                f = "test_" + f
            if not (f.endswith(".py") or f.endswith(".ipynb")):
                f = f + ".py"
            file = os.path.join(test_dir, f)
            has_tests = False
            if os.path.exists(file):
                pytest_args.append(file)
                has_tests = True
            assert has_tests, f"Test {f} does not exist."
    else:
        # run all the tests
        pytest_args = [test_dir]
    pytest_args += ["--nbmake"]
    if args.verbose:
        pytest_args += ["-v"]
    if args.rerun:
        pytest_args += ["--reruns", args.rerun]
    try:
        if args.coverage:
            os.environ.setdefault("QD_KERNEL_COVERAGE", "1")
            os.environ["_QD_KCOV_ARC"] = "1"
            _spec = importlib.util.find_spec("quadrants")
            assert _spec is not None and _spec.origin is not None, "quadrants package not found"
            _cov_src = os.path.dirname(_spec.origin)
            # Disable the quadrants pytest plugin (pytest11 entry point) during coverage runs.  Loading the plugin
            # forces Python to import the quadrants parent package before pytest-cov starts measuring, so module-level
            # code (imports, def lines, etc.) appears uncovered.  The plugin is still useful for *external* users who
            # run `pytest --cov` on their own code — they don't measure quadrants' own coverage.  We replicate the
            # plugin's env-var setup (QD_KERNEL_COVERAGE, _QD_KCOV_ARC) above.
            pytest_args += [
                "--cov-branch",
                f"--cov={_cov_src}",
                f"--cov={test_dir}",
                "-p",
                "no:quadrants",
            ]
        if args.cov_append:
            pytest_args += ["--cov-append"]
        if args.keys:
            pytest_args += ["-k", args.keys]
        if args.marks:
            pytest_args += ["-m", args.marks]
        if args.failed_first:
            pytest_args += ["--failed-first"]
        if args.fail_fast:
            pytest_args += ["--exitfirst"]
        elif args.maxfail > 0:
            pytest_args += [f"--maxfail={args.maxfail}"]
        if args.timeout > 0:
            pytest_args += [
                "--durations=15",
                # Suppress stock pytest-timeout if installed — it conflicts
                # with pytest_hardtle (both register the same hook specs).
                "-p",
                "no:timeout",
                # pytest_hardtle uses a CFFI-compiled C watchdog that calls
                # _exit(1) from a native signal handler, so it can kill tests
                # hung in native GPU calls even when the GIL is held.
                # Stock pytest-timeout's signal method cannot do this.
                "-p",
                "pytest_hardtle",
                f"--timeout={args.timeout}",
            ]
    except AttributeError:
        pass

    try:
        from multiprocessing import cpu_count  # pylint: disable=C0415

        threads = min(8, cpu_count())  # To prevent running out of memory
    except NotImplementedError:
        threads = 2

    env_threads = os.environ.get("QD_TEST_THREADS", "")
    threads = args.threads or env_threads or threads
    print(f"Starting {threads} testing thread(s)...")
    if args.show_output:
        pytest_args += ["-s"]
        print(f"Due to how pytest-xdist is implemented, the -s option does not work with multiple thread...")
    else:
        if int(threads) > 1:
            pytest_args += ["-n", str(threads), "--dist=worksteal"]
    if os.environ.get("QD_FILE_TIMING", "0") == "1":
        import sys as _sys

        if test_dir not in _sys.path:
            _sys.path.insert(0, test_dir)
        pytest_args += ["-p", "pytest_file_timing"]

    import pytest  # pylint: disable=C0415

    return int(pytest.main(pytest_args))


def test():
    """Run the tests"""
    parser = argparse.ArgumentParser(description=f"Run quadrants python test")
    parser.add_argument("files", nargs="*", help='Test name(s) to be run, e.g. "cli"')
    parser.add_argument(
        "-c",
        "--cpp",
        dest="cpp",
        default=False,
        action="store_true",
        help="Only run the C++ tests",
    )
    parser.add_argument(
        "-s",
        "--show",
        dest="show_output",
        action="store_true",
        help="Show output (do not capture)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Run with verbose outputs",
    )
    parser.add_argument(
        "-r",
        "--rerun",
        required=False,
        default=None,
        dest="rerun",
        type=str,
        help="Rerun failed tests for given times",
    )
    parser.add_argument(
        "-k",
        "--keys",
        required=False,
        default=None,
        dest="keys",
        type=str,
        help="Only run tests that match the keys",
    )
    parser.add_argument(
        "-m",
        "--marks",
        required=False,
        default=None,
        dest="marks",
        type=str,
        help="Only run tests with specific marks",
    )
    parser.add_argument(
        "-f",
        "--failed-first",
        required=False,
        default=None,
        dest="failed_first",
        action="store_true",
        help="Run the previously failed test first",
    )
    parser.add_argument(
        "-x",
        "--fail-fast",
        required=False,
        default=None,
        dest="fail_fast",
        action="store_true",
        help="Exit instantly on the first failed test",
    )
    parser.add_argument(
        "--maxfail",
        required=False,
        default=20,
        type=int,
        dest="maxfail",
        help="Stop after this many test failures (default: 20, 0 = no limit)",
    )
    parser.add_argument(
        "-C",
        "--coverage",
        required=False,
        default=None,
        dest="coverage",
        action="store_true",
        help="Run tests and record the coverage result",
    )
    parser.add_argument(
        "-T",
        "--timeout",
        required=False,
        default=600,
        type=int,
        dest="timeout",
        help="Per test timeout (only apply to python tests)",
    )
    parser.add_argument(
        "-A",
        "--cov-append",
        required=False,
        default=None,
        dest="cov_append",
        action="store_true",
        help="Append coverage result to existing one instead of overriding it",
    )
    parser.add_argument(
        "-t",
        "--threads",
        required=False,
        default=None,
        dest="threads",
        type=str,
        help="Custom number of threads for parallel testing",
    )
    parser.add_argument(
        "-a",
        "--arch",
        required=False,
        default=None,
        dest="arch",
        type=str,
        help="Custom the arch(s) (backend) to run tests on",
    )
    parser.add_argument(
        "-n",
        "--exclusive",
        required=False,
        default=False,
        dest="exclusive",
        action="store_true",
        help="Exclude arch(s) from test instead of include them, together with -a",
    )
    run_count = 1
    args = parser.parse_args()
    print(args)

    if args.arch:
        arch = args.arch
        if args.exclusive:
            arch = "^" + arch
        print(f"Running on Arch={arch}")
        os.environ["QD_WANTED_ARCHS"] = arch

    if args.cpp:
        # C++ tests are now handled by pytest too,
        # so we can use `_test_python` to run them,
        # though they are not really python tests.
        exit(_test_python(args, "cpp"))

    for _ in range(run_count):
        ret = _test_python(args)
        if ret == 5:
            # treat 'no tests collected' as success
            ret = 0
        if ret != 0:
            exit(ret)


if __name__ == "__main__":
    test()
