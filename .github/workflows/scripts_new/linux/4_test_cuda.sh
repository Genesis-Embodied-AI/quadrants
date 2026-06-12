#!/bin/bash

# errexit so setup commands (notably `pip install torch ...`) fail the step; run_phase is intentionally
# non-aborting so pytest failures in one phase don't prevent later phases from running (see _run_test_phases.sh).
set -ex

source "$(dirname "$0")/_run_test_phases.sh"

# Disable kernel-level coverage on CUDA: it changes field memory layout and breaks dlpack tests
# (ValueError: Expected zero byte_offset).  Python code coverage (--cov) still runs.
run_phase "not-needs-torch" env QD_KERNEL_COVERAGE=0 \
  python tests/run_tests.py -v -r 1 --arch cuda --coverage -m "not needs_torch"

pip install torch --index-url https://download.pytorch.org/whl/cu128
run_phase "needs-torch" env QD_KERNEL_COVERAGE=0 \
  python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append -m needs_torch

# Run kernel coverage tests on CUDA with coverage enabled — these are skipped by the phases above
# (QD_KERNEL_COVERAGE=0) and include GPU-only tests like test_kernel_coverage_simt_e2e.
run_phase "kernel-coverage" env QD_KERNEL_COVERAGE=1 \
  python tests/run_tests.py -v -r 1 --arch cuda --coverage --cov-append test_kernel_coverage.py

# Coverage data is combined/uploaded by the dedicated "Collect CUDA coverage" workflow step (if: always()).
report_failed_phases
