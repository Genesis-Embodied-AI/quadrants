#!/bin/bash

set -ex

TEST_EXIT=0

# Disable kernel-level coverage on CUDA: it changes field memory layout and
# breaks dlpack tests (ValueError: Expected zero byte_offset).  Python code
# coverage (--cov) still runs.
QD_KERNEL_COVERAGE=0 python tests/run_tests.py -v -r 1 --arch cuda --coverage -m "not needs_torch" || TEST_EXIT=$?

pip install torch --index-url https://download.pytorch.org/whl/cu128
QD_KERNEL_COVERAGE=0 python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append -m needs_torch || TEST_EXIT=$?

python tests/coverage_report.py --collect-only

exit $TEST_EXIT
