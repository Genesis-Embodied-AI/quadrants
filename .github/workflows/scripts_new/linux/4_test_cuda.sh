#!/bin/bash

set -ex

# TEMP: restricted to test_kernel_coverage for faster CI iteration
python tests/run_tests.py -v -r 1 --arch cuda --coverage -k test_kernel_coverage

python tests/coverage_report.py --collect-only
