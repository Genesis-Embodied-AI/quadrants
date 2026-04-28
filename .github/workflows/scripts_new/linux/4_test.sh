#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/quadrants_cpp_tests

# The test_reduction_single* tests put the GPU under a lot of stress and can run
# out of time during the test run with lots of GPU workers. We should run them separately.
python tests/run_tests.py -v -r 3 -a amdgpu -t 16 -k "not test_reduction_single"
python tests/run_tests.py -v -r 3 -a amdgpu -t 16 -k "test_reduction_single"
