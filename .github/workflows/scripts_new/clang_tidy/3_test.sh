#!/bin/bash

set -ex

python ./scripts/run_clang_tidy.py $PWD/quadrants -clang-tidy-binary clang-tidy-14 -header-filter=$PWD/quadrants -j2
