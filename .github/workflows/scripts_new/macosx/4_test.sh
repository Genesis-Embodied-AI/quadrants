#!/bin/bash

set -ex

pip install --prefer-binary --group test
pip install -r requirements_test_xdist.txt
find . -name '*.bc'
ls -lh build/
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
chmod +x ./build/quadrants_cpp_tests
./build/quadrants_cpp_tests

# Phase 1: run all tests except torch-dependent ones
python tests/run_tests.py -v -r 3 --arch metal,vulkan,cpu -m "not needs_torch"

# Phase 2: install torch, run only torch tests
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
python tests/run_tests.py -v -r 3 --arch metal,vulkan,cpu -m needs_torch
