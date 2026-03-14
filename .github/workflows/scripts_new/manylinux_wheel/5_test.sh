#!/bin/bash

set -ex

pip install --group test
pip install -r requirements_test_xdist.txt

# Phase 1: run all tests except torch-dependent ones
python tests/run_tests.py -v -r 3 -m "not needs_torch"

# Phase 2: install torch, run only torch tests
pip install torch --index-url https://download.pytorch.org/whl/cpu
python tests/run_tests.py -v -r 3 -m needs_torch
