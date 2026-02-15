#!/bin/bash

set -ex

pip install --prefer-binary --group test
pip install -r requirements_test_xdist.txt
# TODO: revert to stable torch after 2.9.2 release
pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cpu
find . -name '*.bc'
ls -lh build/
export QD_LIB_DIR="$(python -c 'import quadrants as ti; print(ti.__path__[0])' | tail -n 1)/_lib/runtime"
chmod +x ./build/quadrants_cpp_tests
./build/quadrants_cpp_tests
python tests/run_tests.py -v -r 3 --arch metal,vulkan,cpu
