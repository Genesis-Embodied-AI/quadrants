#!/bin/bash

# errexit: the first failing phase (or a setup command like `pip install torch`) aborts the step, so the failing
# phase's pytest output is the last thing in the log and the red step is this test step (not coverage).
set -ex

# Disable kernel-level coverage on CUDA: it changes field memory layout and breaks dlpack tests
# (ValueError: Expected zero byte_offset).  Python code coverage (--cov) still runs.
QD_KERNEL_COVERAGE=0 python tests/run_tests.py -v -r 1 --arch cuda --coverage -m "not needs_torch"

pip install torch --index-url https://download.pytorch.org/whl/cu128
QD_KERNEL_COVERAGE=0 python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append -m needs_torch

# Run kernel coverage tests on CUDA with coverage enabled — these are skipped by the phases above
# (QD_KERNEL_COVERAGE=0) and include GPU-only tests like test_kernel_coverage_simt_e2e.
QD_KERNEL_COVERAGE=1 python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append test_kernel_coverage.py

# Coverage data is combined/uploaded by the dedicated "Collect CUDA coverage" workflow step (if: always()).
