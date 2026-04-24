#!/bin/bash

set -ex

TEST_EXIT=0

# Disable kernel-level coverage on CUDA: it changes field memory layout and breaks dlpack tests
# (ValueError: Expected zero byte_offset).  Python code coverage (--cov) still runs.
QD_KERNEL_COVERAGE=0 python tests/run_tests.py -v -r 1 --arch cuda --coverage -m "not needs_torch" || TEST_EXIT=$?

pip install torch --index-url https://download.pytorch.org/whl/cu128
QD_KERNEL_COVERAGE=0 python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append -m needs_torch || TEST_EXIT=$?

# Run kernel coverage tests on CUDA with coverage enabled — these are skipped by the phases above
# (QD_KERNEL_COVERAGE=0) and include GPU-only tests like test_kernel_coverage_simt_e2e.
QD_KERNEL_COVERAGE=1 python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append test_kernel_coverage.py || TEST_EXIT=$?

python tests/coverage_report.py --collect-only

exit $TEST_EXIT
