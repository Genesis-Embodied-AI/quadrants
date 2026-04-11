#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/quadrants_cpp_tests  --gtest_filter=-AMDGPU.*

# Tests incompatible with kernel coverage instrumentation:
#  - test_offline_cache: coverage field creates internal kernels breaking cache counts
#  - test_concurrent_kernels: coverage field triggers add_struct_module from worker thread
#  - test_src_ll_cache_with_corruption: recompile after corruption yields different LLVM IR
NO_KCOV="test_offline_cache or test_concurrent_kernels or test_src_ll_cache_with_corruption or test_fe_ll_observations"
python tests/run_tests.py -v -r 3 -k "$NO_KCOV" --coverage

# Enable kernel coverage instrumentation (writes .coverage.kernel at exit)
export QD_KERNEL_COVERAGE=1

# Phase 1: run all tests except torch-dependent and coverage-incompatible ones
python tests/run_tests.py -v -r 3 -m "not needs_torch" -k "not ($NO_KCOV)" --coverage --cov-append

# Phase 2: install torch, run only torch tests
pip install torch --index-url https://download.pytorch.org/whl/cpu
python tests/run_tests.py -v -r 3 -m needs_torch --coverage --cov-append

# Merge per-worker kernel coverage data into the main .coverage produced by pytest-cov
# Each xdist worker writes _qd_kcov.<pid> (not .coverage.* to avoid pytest-cov conflicts)
coverage combine --append _qd_kcov.* 2>/dev/null || true

# Generate coverage reports (--ignore-errors skips temp-file sources from kernel probes)
coverage xml -o coverage.xml --ignore-errors
coverage report --show-missing --skip-covered --ignore-errors > pytest-coverage.txt
