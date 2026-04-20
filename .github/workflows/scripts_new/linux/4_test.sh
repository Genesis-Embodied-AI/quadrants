#!/bin/bash

set -x

pip install --group test
pip install -r requirements_test_xdist.txt
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
./build/quadrants_cpp_tests

# Phase 1: CPU tests (parallel, non-torch)
python tests/run_tests.py -v -r 3 -m "not needs_torch" -a cpu

# Phase 2: AMDGPU tests (serial, non-torch)
python tests/run_tests.py -v -r 3 -m "not needs_torch" -a amdgpu -t 1

# Phase 3: CPU torch tests (parallel)
python tests/run_tests.py -v -r 3 -m needs_torch -a cpu

# Phase 4: AMDGPU torch tests (serial)
python tests/run_tests.py -v -r 3 -m needs_torch -a amdgpu -t 1
