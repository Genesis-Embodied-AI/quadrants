#!/bin/bash

set -ex

source "$(dirname "$0")/_run_test_phases.sh"

pip install --group test
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/quadrants_cpp_tests  --gtest_filter=-AMDGPU.*

# Phase 1: run all tests except torch-dependent ones
run_phase "not-needs-torch" \
  python tests/run_tests.py -v -r 3 --coverage -m "not needs_torch"

pip install torch --index-url https://download.pytorch.org/whl/cpu
run_phase "needs-torch" env QD_KERNEL_COVERAGE=0 \
  python tests/run_tests.py -v -r 3 --coverage --cov-append -m needs_torch

# Phase 3: run tests that are skipped under kernel coverage (offline cache, snode layout, FE-LL observations,
# etc.) without --coverage so QD_KERNEL_COVERAGE stays 0.
run_phase "no-coverage-extras" env QD_KERNEL_COVERAGE=0 \
  python tests/run_tests.py -v -r 3 -m "not needs_torch"

# Coverage data is combined/uploaded by the dedicated "Collect CPU coverage" workflow step (if: always()).
