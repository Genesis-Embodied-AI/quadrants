#!/bin/bash

set -ex

# Tests incompatible with kernel coverage instrumentation
NO_KCOV="test_offline_cache or test_concurrent_kernels or test_src_ll_cache_with_corruption or test_fe_ll_observations"

# Run coverage-incompatible tests first, without kernel coverage
python tests/run_tests.py -v -r 1 --arch cuda -k "$NO_KCOV" --coverage

# Enable kernel coverage instrumentation (writes _qd_kcov.<pid> at exit)
export QD_KERNEL_COVERAGE=1

# Run all CUDA tests except torch-dependent and coverage-incompatible ones
python tests/run_tests.py -v -r 1 --arch cuda -m "not needs_torch" -k "not ($NO_KCOV)" --coverage --cov-append

# Install torch and run torch tests
# Pin to torch cu128 until we update the driver on the github runner gpu nodes
pip install torch --index-url https://download.pytorch.org/whl/cu128
python tests/run_tests.py -v -r 1 --arch cuda -m needs_torch --coverage --cov-append

# Merge per-worker kernel coverage data into the main .coverage
coverage combine --append _qd_kcov.* 2>/dev/null || true

# Generate coverage XML for merging (--ignore-errors skips temp-file sources from kernel probes)
coverage xml -o coverage.xml --ignore-errors
