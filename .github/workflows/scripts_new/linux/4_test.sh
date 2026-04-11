#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/quadrants_cpp_tests  --gtest_filter=-AMDGPU.*

python tests/run_tests.py -v -r 3 --coverage -m "not needs_torch"

pip install torch --index-url https://download.pytorch.org/whl/cpu
python tests/run_tests.py -v -r 3 --coverage --cov-append -m needs_torch

python tests/coverage_report.py --collect-only
